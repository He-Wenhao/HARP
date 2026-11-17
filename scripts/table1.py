"""Reproduce Table 1 of the paper: MAE of BP86 / B3LYP / DSD-PBEP86 and HARP.

HARP column: from the model's output/test/test.json (959 test molecules).
DFT columns: from the property-matched-basis recompute
(data/table1_dft_recompute.tar.gz, also on Zenodo), where each functional is
evaluated at the basis of the property's coupled-cluster reference:
  cc-pVTZ for E (as an atomization energy), |mu|, ||Q||, q and bond order;
  cc-pVDZ for alpha_iso and E_gap.
(compare_methods.py instead uses the def2-SVP/def2-TZVP baseline observables
of the test-set archive, so its DFT columns differ from Table 1.)

Metrics follow the paper: |mu| is the norm of the dipole-error vector in
Debye, ||Q|| the Frobenius norm of the quadrupole-error tensor, q per atom,
bond orders over the truth-listed bonded pairs.

Molecule sets (--set):
  paper (default)  DFT columns over 962 molecules -- the 959 test molecules plus
                   3 (325, 772, 889) removed from the test set by the outlier
                   screen -- which is how the printed Table 1 was aggregated.
  test             DFT columns over the 959 test molecules only.
The HARP column is always over the 959 test molecules.

Usage:
    python scripts/table1.py --model-dir models/HARP [--set paper|test] [--csv out.csv]
"""
import argparse
import configparser
import glob
import io
import json
import os
import tarfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DFT = os.path.join(REPO, 'data', 'table1_dft_recompute.tar.gz')
ESINGLE = os.path.join(REPO, 'src', 'data', 'E_single.json')

HA_KCAL = 627.5094740631
HA_EV = 27.211386245988
CM_INV_HA = 4.55633528e-6
AU_DEBYE = 2.541746
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16, "Cl": 17}

METHODS = ["BP86", "B3LYP", "DSD_PBEP86"]
PROPS = [("E", "E (kcal/mol)"), ("E_gap", "E_gap (eV)"), ("dipole", "|mu| (Debye)"),
         ("quadrupole", "||Q|| (a.u.)"), ("alpha_iso", "alpha_iso (a.u.)"),
         ("atomic_charge", "q (e)"), ("bond_order", "bond order")]
# Table 1 as printed (arXiv:2608.18346v2): BP86, B3LYP, DSD-PBEP86, HARP
PAPER = {"E": ("63.0", "13.2", "15.1", "0.23"), "E_gap": ("1.32", "0.61", "4.84", "0.067"),
         "dipole": ("0.160", "0.159", "0.109", "0.022"),
         "quadrupole": ("0.389", "0.347", "0.233", "0.044"),
         "alpha_iso": ("4.71", "2.79", "1.18", "0.121"),
         "atomic_charge": ("0.0205", "0.0256", "0.0220", "0.0053"),
         "bond_order": ("0.0556", "0.0588", "0.0251", "0.0066")}


def matches_printed(value, printed, slack=0.01):
    """True if value is within half a unit of the printed value's last digit.

    `slack` (1%) widens that by a hair so that GPU run-to-run variation of the
    HARP column (fourth significant figure) does not flip a cell that sits on
    a rounding boundary."""
    nd = len(printed.split(".")[1]) if "." in printed else 0
    return abs(value - float(printed)) <= 0.5 * 10 ** (-nd) * (1 + slack)
OUTLIERS = ("325", "772", "889")


class DFTData:
    """Read the recompute package from a .tar.gz or an unpacked directory."""

    def __init__(self, path):
        self.files = {}
        if os.path.isdir(path):
            root = path.rstrip('/')
            for f in glob.glob(os.path.join(root, '**', '*.json'), recursive=True):
                self.files[os.path.relpath(f, root)] = f
            self._read = lambda k: open(self.files[k]).read()
        else:
            tf = tarfile.open(path)
            members = {m.name: m for m in tf.getmembers() if m.isfile()}
            prefix = os.path.commonpath(list(members)) if len(members) > 1 else ''
            for name, m in members.items():
                self.files[os.path.relpath(name, prefix) if prefix else name] = m
            self._read = lambda k: tf.extractfile(self.files[k]).read().decode()

    def get(self, key):
        return json.loads(self._read(key)) if key in self.files else None


def qmat(o6):
    xx, yy, zz, xy, xz, yz = o6
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def iso(t):
    return float(np.trace(np.array(t))) / 3.0


def mae(chunks):
    if not chunks:
        return float('nan')
    flat = np.concatenate([np.atleast_1d(np.asarray(c, float)).ravel() for c in chunks])
    return float(np.mean(np.abs(flat)))


def truth_terms(t):
    dip = np.array([t["x"], t["y"], t["z"]])
    quad = [t["xx"], t["yy"], t["zz"], t["xy"], t["xz"], t["yz"]]
    bo = {(min(a, b), max(a, b)): v for a, b, v in t["bond_order"]}
    return dip, quad, bo


def dft_errors(mol, dft, ecc, eatom, method):
    """Per-molecule errors of one functional, or {} if its data are missing."""
    idx, els, t = mol
    tz = dft.get(f"obs_pvtz/{method}/{idx}.json")
    dz = dft.get(f"obs_pvdz/{method}/{idx}.json")
    dip_t, quad_t, bo_t = truth_terms(t)
    e = {}
    if tz is not None and tz.get("E_total") is not None:
        shift = sum(ecc[x] - eatom[method][x] for x in els)
        e["E"] = (tz["E_total"] + shift - t["energy"]) * HA_KCAL
        if tz.get("dipole") is not None:
            e["dipole"] = np.linalg.norm(np.array(tz["dipole"]) - dip_t) * AU_DEBYE
        if tz.get("quadrupole"):
            e["quadrupole"] = np.linalg.norm(qmat(tz["quadrupole"]) - qmat(quad_t))
        if tz.get("atomic_charge") and len(tz["atomic_charge"]) == len(t["atomic_charge"]):
            e["atomic_charge"] = np.array(tz["atomic_charge"]) - np.array(t["atomic_charge"])
        bo_m = {(min(a, b), max(a, b)): v for a, b, v in tz.get("bond_order", [])}
        e["bond_order"] = np.array([bo_m.get(k, 0.0) - v for k, v in bo_t.items()])
    if dz is not None:
        if dz.get("alpha") is not None:
            e["alpha_iso"] = iso(dz["alpha"]) - iso(t["alpha"])
        if dz.get("Ee") is not None:
            e["E_gap"] = dz["Ee"] - CM_INV_HA * HA_EV * t["Ee"][0]
    return e


def model_errors(test, group_dir, elements):
    """HARP errors from test.json, with the paper's metrics."""
    err = {p: [] for p, _ in PROPS}
    for i, (bdir, fname) in enumerate(test["file_names"]):
        idx = fname.split("_")[0]
        t = json.load(open(os.path.join(group_dir, "obs", f"{idx}.json")))
        els = elements[idx]
        dip_t, quad_t, bo_t = truth_terms(t)
        g = lambda k: test[k]["pred"][i]
        err["E"].append((g("E") - (t["energy"] - t["E_nn"])) * HA_KCAL)
        err["E_gap"].append((g("E_gap") - CM_INV_HA * t["Ee"][0]) * HA_EV)
        err["dipole"].append(np.linalg.norm(np.array([g(k) for k in "xyz"]) - dip_t) * AU_DEBYE)
        quad_p = [g(k) for k in ("xx", "yy", "zz", "xy", "xz", "yz")]
        err["quadrupole"].append(np.linalg.norm(qmat(quad_p) - qmat(quad_t)))
        err["alpha_iso"].append(iso(g("alpha")) - iso(t["alpha"]))
        zarr = np.array([Z[x] for x in els], float)
        err["atomic_charge"].append(zarr - np.array(g("atomic_charge")) - np.array(t["atomic_charge"]))
        bo_p = np.array(g("bond_order"))
        err["bond_order"].append(np.array([bo_p[a, b] - v for (a, b), v in bo_t.items()]))
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dft", default=DEFAULT_DFT,
                    help="table1_dft_recompute .tar.gz or unpacked directory")
    ap.add_argument("--set", choices=["paper", "test"], default="paper")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(model_dir, "config.ini"))
    group = cfg["test"]["datagroups"].split(",")[0].strip()
    group_dir = os.path.join(REPO, "dataset", group)
    test = json.load(open(os.path.join(model_dir, "output", "test", "test.json")))

    dft = DFTData(args.dft)
    ecc = json.load(open(ESINGLE))["CCSDt"]
    eatom = dft.get("single_atom_ccpvtz.json")["energies"]

    # molecule set for the DFT columns
    mols = []
    for bf in sorted(glob.glob(os.path.join(group_dir, "basic", "*.json"))):
        idx = os.path.basename(bf).split("_")[0]
        op = os.path.join(group_dir, "obs", f"{idx}.json")
        if os.path.exists(op):
            mols.append((idx, json.load(open(bf))["elements"], json.load(open(op))))
    n_test = len(mols)
    if args.set == "paper":
        for idx in OUTLIERS:
            s = [k for k in dft.files if k.startswith(f"outliers/structures/{idx}_")]
            mols.append((idx, dft.get(s[0])["elements"], dft.get(f"outliers/obs/{idx}.json")))

    rows = {p: [] for p, _ in PROPS}
    n_used = {}
    for m in METHODS:
        acc = {p: [] for p, _ in PROPS}
        for mol in mols:
            for p, v in dft_errors(mol, dft, ecc, eatom, m).items():
                acc[p].append(v)
        n_used[m] = len(acc["E"])
        for p, _ in PROPS:
            rows[p].append(mae(acc[p]))
    merr = model_errors(test, group_dir, {idx: els for idx, els, _ in mols})
    for p, _ in PROPS:
        rows[p].append(mae(merr[p]))

    print(f"Table 1 (MAE). DFT columns: {len(mols)} molecules (--set {args.set}; "
          f"{n_test} test molecules{' + outliers ' + ', '.join(OUTLIERS) if args.set == 'paper' else ''}). "
          f"HARP column: {len(test['file_names'])} test molecules.")
    hdr = ["property", "BP86", "B3LYP", "DSD-PBEP86", "HARP"]
    print("| " + " | ".join(hdr) + " | printed in Table 1 |")
    print("|" + "---|" * (len(hdr) + 1))
    n_match, n_cell = 0, 0
    for p, label in PROPS:
        cells = [f"{v:.4g}" for v in rows[p]]
        ref = PAPER[p]
        print(f"| {label} | " + " | ".join(cells) + " | " + " / ".join(ref) + " |")
        for v, r in zip(rows[p], ref):
            n_cell += 1
            n_match += matches_printed(v, r)
    print(f"\n{n_match}/{n_cell} cells agree with the printed Table 1 value to its printed precision.")
    if args.set == "test":
        print("The printed DFT columns include outliers 325, 772, 889; "
              "run with --set paper to reproduce them.")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            for p, label in PROPS:
                w.writerow([label] + [f"{v:.6g}" for v in rows[p]])
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
