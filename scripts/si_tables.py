#!/usr/bin/env python3
"""Numbers for the SI accuracy tables, from one model's test.json.

Same conventions as compare_methods.py (truth = composite CCSD(T)/cc-pVTZ obs/,
bond orders masked to truth-listed bonded pairs, charges = Z - population).
Prints:
  * per-property MAE / RMSE of the model (SI Table "per-property error"),
  * absolute-error distribution (median, p90, p99, max) per molecule /
    per atom / per bonded pair (SI Table "error distributions"),
  * polarizability-tensor channels (iso, diagonal, off-diagonal, anisotropy)
    for the model and the def2-SVP BP86 / B3LYP baselines,
  * Mulliken-charge MAE per element.

Usage:  python scripts/si_tables.py --model-dir models/HARP
"""
import argparse, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_methods import (CM_INV_TO_HARTREE, HARTREE_TO_KCAL_MOL, HARTREE_TO_EV,
                             PERIODIC_TABLE, load_json, load_config)

ELEMENTS = ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl"]


def aniso(a):
    a = np.asarray(a)
    return np.sqrt(0.5 * ((a[0, 0] - a[1, 1]) ** 2 + (a[1, 1] - a[2, 2]) ** 2
                          + (a[2, 2] - a[0, 0]) ** 2
                          + 6 * (a[0, 1] ** 2 + a[0, 2] ** 2 + a[1, 2] ** 2)))


def quad_mat(q):  # xx, yy, zz, xy, yz, xz -> symmetric 3x3
    xx, yy, zz, xy, yz, xz = q
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def channels(a):
    a = np.asarray(a)
    return dict(iso=[np.trace(a) / 3], diag=list(np.diag(a)),
                off=[a[0, 1], a[0, 2], a[1, 2]], aniso=[aniso(a)])


def mae_rmse(x):
    x = np.abs(np.concatenate([np.atleast_1d(v) for v in x]))
    return x.mean(), np.sqrt((x ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    md = os.path.abspath(args.model_dir)
    cfg = load_config(os.path.join(md, "config.ini"))
    nuc = {e.strip(): PERIODIC_TABLE[e.strip()] for e in cfg["element"]["list"].split(",")}
    test = load_json(os.path.join(md, "output", "test", "test.json"))
    print(f"model dir: {md}\ntest.json: {len(test['file_names'])} molecules")

    err = defaultdict(list)          # per-molecule / per-entry signed errors
    dist = defaultdict(list)         # absolute errors for the distribution table
    ten = {m: defaultdict(list) for m in ("model", "B3LYP", "BP86")}
    by_el = defaultdict(list)
    for i, (basic_dir, fname) in enumerate(test["file_names"]):
        idx = fname.split("_")[0]
        gdir = os.path.dirname(basic_dir.rstrip("/"))
        truth = load_json(os.path.join(gdir, "obs", f"{idx}.json"))
        els = load_json(os.path.join(basic_dir, fname))["elements"]
        z = np.array([nuc[e] for e in els], float)

        dE = (test["E"]["pred"][i] - (truth["energy"] - truth["E_nn"])) * HARTREE_TO_KCAL_MOL
        dg = (test["E_gap"]["pred"][i] - CM_INV_TO_HARTREE * truth["Ee"][0]) * HARTREE_TO_EV
        dmu = np.array([test[k]["pred"][i] - truth[k] for k in "xyz"])
        dq = np.array([test[k]["pred"][i] - truth[k] for k in ("xx", "yy", "zz", "xy", "yz", "xz")])
        a_t = np.array(truth["alpha"]); a_p = np.array(test["alpha"]["pred"][i])
        dc = (z - np.array(test["atomic_charge"]["pred"][i])) - np.array(truth["atomic_charge"])
        n = len(els); bt = np.zeros((n, n)); pairs = []
        for a, b, v in truth["bond_order"]:
            bt[a, b] = bt[b, a] = v; pairs.append((min(a, b), max(a, b)))
        bi, bj = np.array(pairs).T
        db = np.array(test["bond_order"]["pred"][i])[bi, bj] - bt[bi, bj]

        err["E"].append(dE); err["E_gap"].append(dg); err["dipole"].append(dmu)
        err["quadrupole"].append(dq); err["alpha_iso"].append(np.trace(a_p - a_t) / 3)
        err["atomic_charge"].append(dc); err["bond_order"].append(db)

        dist["E"].append(abs(dE)); dist["E_gap"].append(abs(dg))
        dist["dipole"].append(np.linalg.norm(dmu))
        dist["quadrupole"].append(np.linalg.norm(quad_mat(dq)))   # Frobenius
        dist["alpha_iso"].append(abs(np.trace(a_p - a_t) / 3))
        dist["atomic_charge"].extend(np.abs(dc)); dist["bond_order"].extend(np.abs(db))
        for e, x in zip(els, np.abs(dc)):
            by_el[e].append(x)

        ct = channels(a_t)
        preds = {"model": a_p}
        for m, sub in (("B3LYP", "obs_B3LYP"), ("BP86", "obs_bp86")):
            p = os.path.join(gdir, sub, f"{idx}.json")
            if os.path.exists(p):
                preds[m] = np.array(load_json(p)["alpha"])
        for m, a in preds.items():
            cp = channels(a)
            for k in ct:
                ten[m][k].append(np.array(cp[k]) - np.array(ct[k]))

    print("\n== per-property MAE / RMSE (model) ==")
    for k in ("E", "E_gap", "dipole", "quadrupole", "alpha_iso", "atomic_charge", "bond_order"):
        print(f"  {k:14s} %.4g / %.4g" % mae_rmse(err[k]))
    print("  |dmu| mean (D) %.4f   ||dQ||_F mean (au) %.4f"
          % (np.mean(dist["dipole"]) * 2.541746, np.mean(dist["quadrupole"])))

    print("\n== absolute-error distribution: median / p90 / p99 / max ==")
    for k, v in dist.items():
        v = np.asarray(v)
        print(f"  {k:14s} " + " / ".join("%.4g" % x for x in
              (np.median(v), np.percentile(v, 90), np.percentile(v, 99), v.max())))

    print("\n== polarizability tensor channels MAE / RMSE (a.u.) ==")
    for m in ("model", "B3LYP", "BP86"):
        print(f"  {m:6s} " + "  ".join(f"{k} %.3f/%.3f" % mae_rmse(ten[m][k])
                                     for k in ("iso", "diag", "off", "aniso")))

    print("\n== Mulliken-charge MAE per element (e) / N atoms ==")
    print("  " + "  ".join(f"{e} {np.mean(by_el[e]):.4f}/{len(by_el[e])}" for e in ELEMENTS))


if __name__ == "__main__":
    main()
