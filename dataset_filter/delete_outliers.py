"""Delete molecules flagged as outliers in outliers.json from the dataset.

For every (datagroup, mol_id) pair flagged at z >= Z_THRESHOLD (default 10),
removes any file in these subdirs whose name starts with `<mol_id>` followed
by '.', '_', or end-of-name:

  basic/   obs/   obs_B3LYP/   obs_bp86/   obs_DSD_PBEP86/
  tmp/     tmp_baseline_b3lyp/   tmp_baseline_bp86/

Writes a per-file log to dataset_filter/deleted_files.log.
"""
import os, sys, json, re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
OUTLIERS_PATH = os.path.join(SCRIPT_DIR, 'outliers.json')
LOG_PATH = os.path.join(SCRIPT_DIR, 'deleted_files.log')

DATA_ROOT = os.path.join(ROOT, 'dataset')
SUBDIRS = ['basic', 'obs', 'obs_B3LYP', 'obs_bp86', 'obs_DSD_PBEP86',
           'tmp', 'tmp_baseline_b3lyp', 'tmp_baseline_bp86']

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"


def matches_mol(name, mol_id):
    """A filename belongs to mol_id iff it starts with `<mol_id>` followed by
    a delimiter ('.', '_') — guards against 1 matching 11, 100, 1234, etc."""
    if not name.startswith(mol_id):
        return False
    rest = name[len(mol_id):]
    return rest == '' or rest[0] in '._'


def main():
    with open(OUTLIERS_PATH) as f:
        out = json.load(f)
    z_thresh = float(os.environ.get("Z_THRESHOLD", "10.0"))
    flagged = [m for m in out['union'] if m['max_z'] >= z_thresh]
    print(f"Outliers loaded with z >= {z_thresh}: {len(flagged)} molecules")
    if DRY_RUN:
        print("DRY_RUN=1 — no files will be deleted")

    log = open(LOG_PATH, 'w')
    log.write(f"# Deletion run (z >= {z_thresh})\n")
    log.write(f"# {len(flagged)} molecules flagged\n\n")

    total_files = 0
    for m in sorted(flagged, key=lambda x: -x['max_z']):
        mol_id = str(m['mol'])
        group = m['datagroup']
        group_dir = os.path.join(DATA_ROOT, group)
        if not os.path.isdir(group_dir):
            print(f"  [skip] missing {group_dir}")
            continue

        log.write(f"\n## mol {mol_id}  in  {group}  (z_max={m['max_z']:.2f})\n")
        flagged_props = ", ".join(f"{p}={z['z']:.2f}" for p, z in m['flagged'].items())
        log.write(f"#  flagged: {flagged_props}\n")

        per_mol = 0
        for sub in SUBDIRS:
            subdir = os.path.join(group_dir, sub)
            if not os.path.isdir(subdir):
                continue
            for entry in os.listdir(subdir):
                if matches_mol(entry, mol_id):
                    fpath = os.path.join(subdir, entry)
                    log.write(f"  delete  {os.path.relpath(fpath, ROOT)}\n")
                    if not DRY_RUN:
                        os.remove(fpath)
                    per_mol += 1
        log.write(f"#  {per_mol} files\n")
        total_files += per_mol
        print(f"  mol {mol_id:>6} in {group}: {per_mol} files {'(dry)' if DRY_RUN else 'removed'}")

    log.write(f"\n# Total: {total_files} files across {len(flagged)} molecules\n")
    log.close()
    print(f"\n{'Would delete' if DRY_RUN else 'Deleted'} {total_files} files from "
          f"{len(flagged)} molecules. Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
