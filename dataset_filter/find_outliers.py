"""Identify outlier molecules from per-molecule per-property errors.

Reads:  dataset_filter/per_mol_errors.json
Writes: dataset_filter/outliers.json
        dataset_filter/error_distributions.png
        dataset_filter/outliers_summary.txt
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ERR_PATH = os.path.join(SCRIPT_DIR, 'per_mol_errors.json')

# Robust z-score threshold above which a property is flagged
Z_THRESHOLD = float(os.environ.get("Z_THRESHOLD", "5.0"))

# Properties to inspect (in display order)
PROPS = ['E', 'E_gap', 'dipole', 'quadrupole', 'atomic_charge', 'alpha', 'bond_order']


def robust_z(values):
    """Median-based z-score on log10(error). Returns z and the log values."""
    log_v = np.log10(values + 1e-30)
    med = np.median(log_v)
    mad = np.median(np.abs(log_v - med))
    sigma = 1.4826 * mad if mad > 0 else log_v.std()
    if sigma == 0:
        sigma = 1.0
    return (log_v - med) / sigma, log_v, med, sigma


def main():
    with open(ERR_PATH) as f:
        data = json.load(f)

    # Flatten: [(group, mol_id, errors_dict), ...]
    all_mols = []
    for group, mols in data.items():
        for mid, err in mols.items():
            all_mols.append((group, mid, err))

    print(f"Loaded {len(all_mols)} molecules across {len(data)} datagroups")
    print(f"Z-score threshold (on log10 err): {Z_THRESHOLD}")

    # Per-property analysis
    by_prop = {}        # prop -> list of (group, mid, err, z)
    flagged = {}        # (group, mid) -> {prop: (err, z)}
    for prop in PROPS:
        triples = [(g, m, e[prop]) for g, m, e in all_mols
                   if prop in e and e[prop] is not None and np.isfinite(e[prop])]
        if not triples:
            continue
        errs = np.array([t[2] for t in triples], dtype=float)
        z, log_v, med, sigma = robust_z(errs)
        by_prop[prop] = {
            'triples': triples,
            'z': z,
            'log_v': log_v,
            'med': med,
            'sigma': sigma,
        }
        for (g, m, err), zi in zip(triples, z):
            if zi > Z_THRESHOLD:
                flagged.setdefault((g, m), {})[prop] = (float(err), float(zi))

    # Summary
    out_path = os.path.join(SCRIPT_DIR, 'outliers.json')
    summary_path = os.path.join(SCRIPT_DIR, 'outliers_summary.txt')

    with open(summary_path, 'w') as f:
        f.write(f"Total molecules: {len(all_mols)}\n")
        f.write(f"Z threshold (log10 err): {Z_THRESHOLD}\n\n")
        f.write("Per-property tail:\n")
        f.write(f"{'property':<16}{'N':>8}{'median(err²)':>16}{'sigma_log':>12}{'#outliers':>12}\n")
        f.write("-" * 64 + "\n")
        for prop in PROPS:
            if prop not in by_prop:
                continue
            d = by_prop[prop]
            n_out = int(np.sum(d['z'] > Z_THRESHOLD))
            f.write(f"{prop:<16}{len(d['triples']):>8}{10**d['med']:>16.4e}"
                    f"{d['sigma']:>12.3f}{n_out:>12}\n")
        f.write(f"\nUnion (any property flagged): {len(flagged)} molecules\n\n")

        # Top outliers per property
        for prop in PROPS:
            if prop not in by_prop:
                continue
            d = by_prop[prop]
            order = np.argsort(d['z'])[::-1]
            f.write(f"\n=== Top 15 outliers for {prop} ===\n")
            f.write(f"{'mol_id':<10}{'datagroup':<48}{'err²':>14}{'z':>8}\n")
            for k in order[:15]:
                if d['z'][k] <= Z_THRESHOLD:
                    break
                g, m, e = d['triples'][k]
                f.write(f"{m:<10}{g:<48}{e:>14.4e}{d['z'][k]:>8.2f}\n")

        # Union table (sorted by max z across props)
        f.write("\n\n=== UNION OUTLIERS (any property z > threshold) ===\n")
        f.write(f"{'mol_id':<10}{'datagroup':<48}{'flagged_props':<60}{'max_z':>8}\n")
        sorted_flagged = sorted(
            flagged.items(),
            key=lambda kv: -max(zz for _, zz in kv[1].values()),
        )
        for (g, m), props in sorted_flagged:
            max_z = max(zz for _, zz in props.values())
            tags = ",".join(f"{p}({z:.1f})" for p, (_, z) in props.items())
            f.write(f"{m:<10}{g:<48}{tags:<60}{max_z:>8.2f}\n")

    # JSON output
    out = {
        'threshold_z': Z_THRESHOLD,
        'n_total': len(all_mols),
        'per_property': {
            prop: {
                'n': len(by_prop[prop]['triples']),
                'median_err2': float(10 ** by_prop[prop]['med']),
                'sigma_log': float(by_prop[prop]['sigma']),
                'n_outliers': int(np.sum(by_prop[prop]['z'] > Z_THRESHOLD)),
                'top_outliers': [
                    {'mol': m, 'datagroup': g, 'err2': float(e),
                     'z': float(by_prop[prop]['z'][k])}
                    for k, (g, m, e) in enumerate(by_prop[prop]['triples'])
                    if by_prop[prop]['z'][k] > Z_THRESHOLD
                ],
            } for prop in by_prop
        },
        'union': [
            {
                'datagroup': g, 'mol': m,
                'flagged': {p: {'err2': e, 'z': z} for p, (e, z) in props.items()},
                'max_z': max(z for _, z in props.values()),
            }
            for (g, m), props in flagged.items()
        ],
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    # Plots: histogram per property of log10(error)
    n_plots = sum(1 for p in PROPS if p in by_prop)
    fig, axes = plt.subplots(
        nrows=(n_plots + 1) // 2, ncols=2,
        figsize=(12, 3 * ((n_plots + 1) // 2)),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    pi = 0
    for prop in PROPS:
        if prop not in by_prop:
            continue
        ax = axes_flat[pi]
        d = by_prop[prop]
        ax.hist(d['log_v'], bins=80, alpha=0.7, color='steelblue', edgecolor='none')
        cutoff = d['med'] + Z_THRESHOLD * d['sigma']
        ax.axvline(d['med'], color='black', linestyle='--', linewidth=1, label='median')
        ax.axvline(cutoff, color='red', linestyle='--', linewidth=1,
                   label=f'z>{Z_THRESHOLD:.0f}')
        n_out = int(np.sum(d['z'] > Z_THRESHOLD))
        ax.set_title(f"{prop}  (N={len(d['triples'])}, outliers={n_out})")
        ax.set_xlabel('log10(squared error)')
        ax.set_ylabel('count')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        pi += 1
    for k in range(pi, len(axes_flat)):
        axes_flat[k].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'error_distributions.png'), dpi=120)

    print(f"\nWrote:")
    print(f"  {out_path}")
    print(f"  {summary_path}")
    print(f"  {os.path.join(SCRIPT_DIR, 'error_distributions.png')}")
    print(f"\nUnion outliers: {len(flagged)} molecules (z > {Z_THRESHOLD})")


if __name__ == "__main__":
    main()
