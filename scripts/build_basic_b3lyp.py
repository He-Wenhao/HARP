"""Build the B3LYP model inputs `basic_B3LYP/` for one or more dataset groups.

The released HARP model starts from a B3LYP/def2-SVP Hamiltonian, so
`models/HARP/config.ini` reads its per-molecule inputs from `basic_B3LYP/`
(`basic_subdir = basic_B3LYP`). The Zenodo test-set archive ships `basic/`,
which holds the **BP86**/def2-SVP Fock/overlap/polarizability. Do not feed
`basic/` to the model (and do not symlink it to `basic_B3LYP/`): it runs
without error but every Hamiltonian-derived property comes out wrong.

`basic_B3LYP/` needs no new quantum-chemistry run. Each file merges
  - geometry and composition (charge, multiplicity, electron counts,
    coordinates, elements, name, header) from `basic/<id>_<formula>.json`, and
  - the B3LYP Fock `h`, overlap `S`, polarizability `alpha`, total energy
    (`HF` <- `energy`), nuclear repulsion (`Enn` <- `E_nn`) and `E_shift`
    from `obs_B3LYP/<id>.json`,
both of which are in the Zenodo archive. `obs_B3LYP` stores `h` in the same
convention as `basic` (shifted AO Fock, ORCA ordering).

Usage:
    python scripts/build_basic_b3lyp.py [GROUP_DIR ...]

GROUP_DIR is a directory holding `basic/` and `obs_B3LYP/`. The default is
the test group, dataset/P_S_Cl_dataset_v3/P_S_Cl_data_1_1000.
"""
import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GROUP = os.path.join(REPO_ROOT, 'dataset', 'P_S_Cl_dataset_v3',
                             'P_S_Cl_data_1_1000')

# fields taken from basic/ (functional-independent)
GEOM_KEYS = ['charge', 'multiplicity', 'ne', 'ne0', 'ne1',
             'coordinates', 'elements', 'name', 'header']
# preprocessing cache written by the loader from basic_B3LYP/
CACHE_SUBDIR = 'tmp_basic_b3lyp'


def build_group(gdir):
    bdir = os.path.join(gdir, 'basic')
    odir = os.path.join(gdir, 'obs_B3LYP')
    outdir = os.path.join(gdir, 'basic_B3LYP')
    for d in (bdir, odir):
        if not os.path.isdir(d):
            sys.exit(f"error: {d} not found")
    if os.path.islink(outdir):
        sys.exit(f"error: {outdir} is a symlink (-> {os.readlink(outdir)}). "
                 "Remove it first; writing through it would overwrite the "
                 "files it points to.")
    os.makedirs(outdir, exist_ok=True)

    n_ok = n_skip = 0
    for bf in sorted(glob.glob(os.path.join(bdir, '*.json'))):
        fn = os.path.basename(bf)                 # e.g. 143_H1Cl2O1P1.json
        of = os.path.join(odir, fn.split('_')[0] + '.json')
        if not os.path.exists(of):
            n_skip += 1
            continue
        with open(bf) as fh:
            basic = json.load(fh)
        with open(of) as fh:
            ob3 = json.load(fh)
        out = {k: basic.get(k) for k in GEOM_KEYS}
        out['h'] = ob3['h']
        out['S'] = ob3['S']
        out['alpha'] = ob3['alpha']
        out['HF'] = ob3['energy']
        out['Enn'] = ob3['E_nn']
        out['E_shift'] = ob3.get('E_shift')
        with open(os.path.join(outdir, fn), 'w') as fh:   # keep <id>_<formula>.json
            json.dump(out, fh)
        n_ok += 1
    print(f"{gdir}: wrote {n_ok} files to basic_B3LYP/, "
          f"skipped {n_skip} (no obs_B3LYP)")

    cache = os.path.join(gdir, CACHE_SUBDIR)
    if os.path.isdir(cache) and os.listdir(cache):
        print(f"  note: {cache} already exists. The loader reuses cached .pt "
              "files without re-reading basic_B3LYP/; delete that directory "
              "if it was built from other inputs.")


def main():
    ap = argparse.ArgumentParser(
        description="Build basic_B3LYP/ from basic/ + obs_B3LYP/.")
    ap.add_argument('groups', nargs='*', default=[DEFAULT_GROUP],
                    help="group directories containing basic/ and obs_B3LYP/ "
                         f"(default: {os.path.relpath(DEFAULT_GROUP, REPO_ROOT)})")
    args = ap.parse_args()
    for g in args.groups:
        build_group(os.path.abspath(g))


if __name__ == '__main__':
    main()
