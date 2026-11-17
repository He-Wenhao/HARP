"""Per-label MAE/RMSE: NN model vs DFT methods (B3LYP / BP86 / DSD-PBEP86).

Universal driver — point it at any trained model directory that contains
`config.ini` and `output/test/test.json`. The dataset path used for the
comparison is read from `[test] datagroups` in the model's config; for each
listed datagroup the script expects the standard sub-folders:
  basic/        per-molecule geometry JSONs (named <idx>_<formula>.json)
  obs/          high-level reference labels (named <idx>.json)
  obs_B3LYP/    B3LYP reference labels  (named <idx>.json)  [optional]
  obs_bp86/     BP86 reference labels   (named <idx>.json)  [optional]
  obs_DSD_PBEP86/ DSD-PBEP86 reference labels (named <idx>.json) [optional]

The DFT folders above are the def2-SVP (BP86, B3LYP) and def2-TZVP
(DSD-PBEP86) observables of the test-set archive, so the DFT columns here are
not those of the paper's Table 1; scripts/table1.py reproduces Table 1 from
the property-matched-basis recompute.

Each method's per-property MAE/RMSE is aggregated across all 959 (or N) test
molecules, with bond-order errors masked to truth-listed bonded pairs only
(zero-fill on non-bonded entries gives DFT methods an unfair advantage).

Usage:
    python compare_methods.py --model-dir /path/to/models/<name>
    python compare_methods.py --model-dir <dir> --dataset-root <root>
"""
import argparse
import configparser
import json
import os
import numpy as np

CM_INV_TO_HARTREE = 4.55633528e-6
EV_TO_HARTREE = 1.0 / 27.2114
HARTREE_TO_KCAL_MOL = 627.5094740631   # E reported in kcal/mol
HARTREE_TO_EV       = 27.211386245988  # E_gap reported in eV

# Periodic table — universal. compare_methods filters to whatever elements
# are actually used by the model (read from config.ini [element] list).
PERIODIC_TABLE = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Br": 35, "I": 53,
}


def load_json(p):
    with open(p) as f:
        return json.load(f)


def load_config(path):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(path)
    return cfg


def agg(diffs):
    flat = np.concatenate([np.asarray(d, dtype=float).ravel() for d in diffs])
    mae = float(np.mean(np.abs(flat)))
    rmse = float(np.sqrt(np.mean(flat ** 2)))
    return mae, rmse, len(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="Trained model directory (contains config.ini, "
                         "output/test/test.json).")
    ap.add_argument("--dataset-root",
                    default=None,
                    help="Optional override for dataset root. By default, "
                         "use the model's `[test] datagroups` and the parent "
                         "of `output/...` to locate dataset/. Most users can "
                         "leave this unset.")
    ap.add_argument("--report", default=None,
                    help="Output path for report.md (default: "
                         "<model-dir>/output/test/report.md).")
    ap.add_argument("--csv", default=None,
                    help="If set, also write a CSV table to this path. "
                         "Columns: property,unit,<method>_MAE/RMSE,… "
                         "with methods ordered model, BP86, B3LYP, DSD_PBEP86.")
    args = ap.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    test_json = os.path.join(model_dir, "output", "test", "test.json")
    if not os.path.exists(test_json):
        raise SystemExit(f"test.json not found at {test_json}; run test.py first.")
    report_path = args.report or os.path.join(model_dir, "output", "test", "report.md")

    # Element list → nuclear-charge map for THIS model only
    cfg = load_config(os.path.join(model_dir, "config.ini"))
    element_list = [e.strip() for e in cfg["element"]["list"].split(",")]
    try:
        nuc_charge = {e: PERIODIC_TABLE[e] for e in element_list}
    except KeyError as missing:
        raise SystemExit(f"Element {missing} not in PERIODIC_TABLE — please add it.")

    test = load_json(test_json)
    file_names = test["file_names"]
    n = len(file_names)
    print(f"Loaded {n} test molecules from {test_json}")

    methods = ["model", "BP86", "B3LYP", "DSD_PBEP86"]
    method_dirs = {"BP86": "obs_bp86", "B3LYP": "obs_B3LYP",
                   "DSD_PBEP86": "obs_DSD_PBEP86"}
    props = ["E", "E_gap", "dipole", "quadrupole",
             "alpha_iso", "atomic_charge", "bond_order"]
    diffs = {m: {p: [] for p in props} for m in methods}

    skipped = 0
    for i, (basic_dir, fname) in enumerate(file_names):
        idx = fname.split("_")[0]
        # basic_dir ends in the loader's basic subdir, which is `basic` for a
        # BP86-baseline model but `basic_B3LYP` for a B3LYP-baseline one; strip
        # whatever it actually is rather than a fixed-width "/basic/".
        group_dir = os.path.dirname(basic_dir.rstrip("/"))
        obs_path = os.path.join(group_dir, "obs", f"{idx}.json")
        if not os.path.exists(obs_path):
            skipped += 1
            continue
        truth = load_json(obs_path)
        basic = load_json(os.path.join(basic_dir, fname))
        elements = basic["elements"]
        zarr = np.array([nuc_charge[e] for e in elements], dtype=float)

        # ---- Truth ----
        E_true = truth["energy"] - truth["E_nn"]
        Egap_true = CM_INV_TO_HARTREE * truth["Ee"][0]
        dip_true = np.array([truth["x"], truth["y"], truth["z"]])
        quad_true = np.array([truth["xx"], truth["yy"], truth["zz"],
                              truth["xy"], truth["yz"], truth["xz"]])
        alpha_iso_true = float(np.trace(np.array(truth["alpha"]))) / 3.0
        ac_true = np.array(truth["atomic_charge"])

        natom = len(elements)
        bo_true = np.zeros((natom, natom))
        bonded_idx = []
        for a, b, v in truth["bond_order"]:
            bo_true[a, b] = v
            bo_true[b, a] = v
            ii, jj = (a, b) if a < b else (b, a)
            bonded_idx.append((ii, jj))
        bi = np.array([p[0] for p in bonded_idx], dtype=int)
        bj = np.array([p[1] for p in bonded_idx], dtype=int)

        # ---- Model predictions ----
        E_pred_m = test["E"]["pred"][i]
        Egap_pred_m = test["E_gap"]["pred"][i]
        dip_pred_m = np.array([test["x"]["pred"][i],
                               test["y"]["pred"][i],
                               test["z"]["pred"][i]])
        quad_pred_m = np.array([test["xx"]["pred"][i], test["yy"]["pred"][i],
                                test["zz"]["pred"][i], test["xy"]["pred"][i],
                                test["yz"]["pred"][i], test["xz"]["pred"][i]])
        alpha_iso_pred_m = float(np.trace(np.array(test["alpha"]["pred"][i]))) / 3.0
        ac_pred_m = zarr - np.array(test["atomic_charge"]["pred"][i])
        bo_pred_m = np.array(test["bond_order"]["pred"][i])

        diffs["model"]["E"].append(E_pred_m - E_true)
        diffs["model"]["E_gap"].append(Egap_pred_m - Egap_true)
        diffs["model"]["dipole"].append(dip_pred_m - dip_true)
        diffs["model"]["quadrupole"].append(quad_pred_m - quad_true)
        diffs["model"]["alpha_iso"].append(alpha_iso_pred_m - alpha_iso_true)
        diffs["model"]["atomic_charge"].append(ac_pred_m - ac_true)
        diffs["model"]["bond_order"].append(
            bo_pred_m[bi, bj] - bo_true[bi, bj])

        # ---- DFT-method predictions ----
        for mname, sub in method_dirs.items():
            mpath = os.path.join(group_dir, sub, f"{idx}.json")
            if not os.path.exists(mpath):
                continue
            mo = load_json(mpath)
            E_pred = mo["energy"] - mo["E_nn"]
            try:
                Egap_pred = float(mo["Ee"]) * EV_TO_HARTREE
            except (TypeError, ValueError):
                Egap_pred = CM_INV_TO_HARTREE * mo["Ee"][0]
            dip_pred = np.array([mo["x"], mo["y"], mo["z"]])
            quad_pred = np.array([mo["xx"], mo["yy"], mo["zz"],
                                  mo["xy"], mo["yz"], mo["xz"]])
            alpha_iso_pred = float(np.trace(np.array(mo["alpha"]))) / 3.0
            ac_pred = np.array(mo["atomic_charge"])
            bo_pred = np.zeros((natom, natom))
            for a, b, v in mo["bond_order"]:
                bo_pred[a, b] = v
                bo_pred[b, a] = v

            diffs[mname]["E"].append(E_pred - E_true)
            diffs[mname]["E_gap"].append(Egap_pred - Egap_true)
            diffs[mname]["dipole"].append(dip_pred - dip_true)
            diffs[mname]["quadrupole"].append(quad_pred - quad_true)
            diffs[mname]["alpha_iso"].append(alpha_iso_pred - alpha_iso_true)
            diffs[mname]["atomic_charge"].append(ac_pred - ac_true)
            diffs[mname]["bond_order"].append(
                bo_pred[bi, bj] - bo_true[bi, bj])

    print(f"Skipped {skipped} mols (missing obs file)")

    # Render markdown
    units = {"E": "kcal/mol", "E_gap": "eV",
             "dipole": "a.u., per-component",
             "quadrupole": "a.u., per-component",
             "alpha_iso": "a.u.",
             "atomic_charge": "e, per-atom",
             "bond_order": "per-pair, bonded only"}
    # Per-property unit-conversion factor applied to the raw diffs (which are
    # method − truth in the property's native unit, Hartree for energies).
    unit_scale = {p: 1.0 for p in props}
    unit_scale["E"]     = HARTREE_TO_KCAL_MOL
    unit_scale["E_gap"] = HARTREE_TO_EV

    # One combined "MAE/RMSE" cell per method.
    header = ["Property (unit)"] + [f"{m} (MAE/RMSE)" for m in methods] + ["N"]
    rows = []
    # Also collect numeric values for CSV (single cell per method, "mae/rmse" string).
    num_rows = []   # list of dicts: {property, unit, <m>_MAE/RMSE, …, N}
    for p in props:
        row_cells = []
        n_ref = None
        scale = unit_scale[p]
        num_row = {"property": p, "unit": units[p]}
        for m in methods:
            if not diffs[m][p]:
                row_cells.append("—")
                num_row[f"{m}_MAE/RMSE"] = ""
                continue
            mae, rmse, k = agg(diffs[m][p])
            mae *= scale; rmse *= scale
            row_cells.append(f"{mae:.4g} / {rmse:.4g}")
            num_row[f"{m}_MAE/RMSE"] = f"{mae:.6g}/{rmse:.6g}"
            n_ref = k
        num_row["N"] = n_ref if n_ref is not None else ""
        num_rows.append(num_row)
        rows.append([f"{p} ({units[p]})"] + row_cells + [str(n_ref)])

    lines = []
    lines.append(f"# Per-label MAE / RMSE: NN model vs DFT methods\n")
    lines.append(f"Model dir: `{model_dir}`. Test set size: {n - skipped}.\n")
    lines.append("Errors are method − truth, aggregated over flattened "
                 "per-element entries (bond_order masked to truth-listed "
                 "bonded pairs).\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {report_path}")
    print("\n".join(lines))

    if args.csv:
        import csv
        csv_path = os.path.abspath(args.csv)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        cols = ["property", "unit"] + [f"{m}_MAE/RMSE" for m in methods]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in num_rows:
                w.writerow(r)
        print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
