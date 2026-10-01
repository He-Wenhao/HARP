"""Run HARP on ARBITRARY geometries to reproduce the
out-of-distribution size-extrapolation test (oligothiophenes T_n, paper Fig. 5)
with the released checkpoint.

run.py only supports the direct head and never produces the Fock matrix / AO
operators the hamiltonian head needs. This script regenerates the BP86/def2-SVP
baseline via **ORCA** — exactly how the training dataset was generated — parses
the Fock matrix h, overlap S, and BP86 polarizability alpha_bp86, applies the
dataset's energy shift, and runs the working ham inference path
(model -> state_solver(info) -> predict_properties).

Convention (validated element-wise against the dataset basic files, see
validate_orca_gen.py): h/S/alpha are in ORCA's native AO ordering (NO basis
permutation); alpha_bp86 is stored in ORCA's raw frame (NO axis permutation);
h is energy-shifted then orthonormalized (S^-1/2 h S^-1/2); operator matrices
are built from the [y,z,x]-reordered coordinates (as the loader does).

Modes:
  --validate <MOLID>  rerun a dataset molecule from its stored geometry and
                      compare predicted alpha/E/E_gap to models/h_v8 test.json.
  --series {oligothiophene,chloropolyene} --n-max N
                      sweep the homologous series -> size_extrapolation.csv
"""
import sys, os, json, csv, glob, re, argparse, configparser, tempfile, shutil
import numpy as np
import scipy.linalg
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from data.interface_pyscf import generate_basic
from model.xyz2graph import sampler
from deploy.deploy_cls import estimator_test
from train.solve_H import state_solver
from deploy.orca_b3lyp import (write_orca_input, run_orca, OrcaRunConfig,
                               parse_property_polarizability_au,
                               parse_property_dft_energy_hartree,
                               parse_log_nuclear_repulsion_hartree,
                               _get_property_file_path, _read_text)

ORCA_BIN = os.environ.get('ORCA_BIN') or shutil.which('orca') or 'orca'
Z = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17}
PERM = [1, 2, 0]  # coords x,y,z -> y,z,x for operator integrals (matches loader)
# Single-atom energies for the delta-learning baseline alignment: the dataset's
# obs_bp86 energy is the raw BP86 energy shifted by sum_atoms(CCSDt - bp86), so
# baseline_E sits near the CCSD(T) reference the model targets. (verified: mol
# 143 align = 1.124 Ha = the residual E gap when omitted.)
_E_SINGLE = json.load(open(os.path.join(ROOT_DIR, 'src', 'data', 'E_single.json')))

# Baseline functional for the on-the-fly ORCA run, set in main() from the model's
# delta_baseline. 'method' = ORCA keyword; 'align_key' = E_single key for the
# CCSDt-<baseline> single-atom energy alignment of baseline_E.
BASELINE = {'method': 'BP86', 'align_key': 'bp86'}
_DELTA2ORCA = {'bp86': ('BP86', 'bp86'), 'b3lyp': ('B3LYP', 'B3LYP')}


def energy_alignment(elements):
    return sum(_E_SINGLE['CCSDt'][e] - _E_SINGLE[BASELINE['align_key']][e] for e in elements)


# ----------------------------- config / model -----------------------------
def load_config():
    c = configparser.ConfigParser(); c.optionxform = str
    c.read(os.path.join(SCRIPT_DIR, 'config.ini'))
    return c


def parse_backbone(c):
    cfg = {}
    for k, v in c["backbone"].items():
        if k == "type":
            continue
        v = v.strip()
        if v.lower() in ("true", "false"):
            cfg[k] = c["backbone"].getboolean(k)
        else:
            try:
                cfg[k] = int(v)
            except ValueError:
                try:
                    cfg[k] = float(v)
                except ValueError:
                    cfg[k] = v
    return cfg


def build_estimator(c, device):
    scaling = {k: float(v) for k, v in c["scaling"].items()}
    element_list = [x.strip() for x in c["element"]["list"].split(",")]
    est = estimator_test(device, c["general"].getboolean("nodeRDM_flag"),
                         output_folder=os.path.join(SCRIPT_DIR, "output", "test"))
    est.build_model(scaling=scaling, element_list=element_list,
                    backbone_type=c["backbone"].get("type", "internal").strip(),
                    backbone_config=parse_backbone(c),
                    alpha_head=c["general"].get("alpha_head", "sum").strip().lower(),
                    head_type=c["general"].get("head_type", "direct").strip().lower())
    est.configure_egap_pooling(c["general"].get("egap_pooling", "sum").strip().lower())
    est.configure_gap_rescale(c["general"].get("gap_rescale", "affine").strip().lower())
    model_dir = os.path.join(SCRIPT_DIR, "output", "model")
    req = c["test"].get("model_file", "latest_checkpoint.pt").strip()
    path = os.path.join(model_dir, req)
    if not os.path.exists(path):
        cands = []
        for p in glob.glob(os.path.join(model_dir, "*_model.pt")):
            m = re.match(r"(\d+)_model\.pt$", os.path.basename(p))
            if m:
                cands.append((int(m.group(1)), p))
        path = sorted(cands)[-1][1]
    print(f"[ham_chain] checkpoint = {os.path.basename(path)}")
    est.load(path); est.model.eval()
    return est, element_list


# ------------------------- ORCA log matrix parsing -------------------------
def _is_header(line):
    toks = line.split()
    return bool(toks) and all(re.fullmatch(r'-?\d+', t) for t in toks)


def read_block_matrix(lines, title_idx, n):
    """Parse an ORCA block-printed n x n matrix after `title_idx`: reads exactly
    n rows per column-block, skipping dashes/blank lines before each header, and
    stops once n columns are collected (ignoring any trailing summary line)."""
    i = title_idx + 1
    blocks, cols = [], 0
    while cols < n:
        while not _is_header(lines[i]):
            i += 1
        i += 1
        rows = [[float(x) for x in lines[i + r].split()[1:]] for r in range(n)]
        i += n
        blk = np.array(rows)
        blocks.append(blk)
        cols += blk.shape[1]
    return np.hstack(blocks)


def parse_fock_overlap(log_text, n):
    out = log_text.splitlines()
    s_idx = next(i for i, l in enumerate(out) if 'OVERLAP MATRIX' in l)
    fock_idxs = [i for i, l in enumerate(out) if 'Fock matrix for operator 0' in l]
    if not fock_idxs:
        raise RuntimeError("no 'Fock matrix for operator 0' block in log")
    S = read_block_matrix(out, s_idx, n)
    h = read_block_matrix(out, fock_idxs[-1], n)
    return h, S


# ------------------------------ operators ------------------------------
def build_obs_mats(gb, elements, pos, Smhalf, device):
    """Mirror loader.read_obs_mat (center='nuclei_charge')."""
    pos_t = torch.tensor(pos, dtype=torch.float, device=device)
    data_obs = {}
    for op in ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz']:
        res = gb.integrator.calc_O(pos_t[None, :, :], elements, op, center="nuclei_charge")[0]
        data_obs[op] = Smhalf @ torch.as_tensor(res, dtype=torch.float, device=device) @ Smhalf
    Fmat = gb.integrator.calc_F(pos_t, elements)
    data_obs['F'] = Smhalf @ torch.as_tensor(Fmat, dtype=torch.float, device=device) @ Smhalf
    return data_obs


# ------------------------- ORCA baseline generation -------------------------
def generate_inputs(gb, elements, coords_ang, device, workdir, nprocs=1):
    """Run ORCA BP86/def2-SVP and build the ham-pipeline inputs, reproducing the
    dataset generation convention. Returns (data_in, data_obs, baseline_E_elec)."""
    os.makedirs(workdir, exist_ok=True)
    cfg = OrcaRunConfig(orca_bin=ORCA_BIN, nprocs=nprocs, method=BASELINE['method'],
                        basis='def2-SVP', align_energy_to_ccsdt=False)
    write_orca_input(os.path.join(workdir, 'run.inp'), elements, coords_ang, cfg)
    run_orca(workdir=workdir, cfg=cfg)

    log_text = _read_text(os.path.join(workdir, 'log'))
    prop_text = _read_text(_get_property_file_path(workdir))

    norb = int(round(sum(gb.nbasis[e] for e in elements)))
    h_raw, S = parse_fock_overlap(log_text, norb)
    alpha = np.array(parse_property_polarizability_au(prop_text))   # raw ORCA frame, NO perm
    E_scf = parse_property_dft_energy_hartree(prop_text)
    E_nn = parse_log_nuclear_repulsion_hartree(log_text)
    ne_elec = sum(Z[e] for e in elements)
    # baseline_E aligned to the CCSD(T) reference (dataset obs_bp86 convention)
    baseline_E_elec = float(E_scf - E_nn) + energy_alignment(elements)

    # energy shift (dataset convention) then orthonormalize
    shift = (-np.sum(scipy.linalg.eigvalsh(h_raw, S)[:ne_elec // 2]) * 2 + E_scf - E_nn) / ne_elec
    h_shifted = h_raw + shift * S
    Smhalf = torch.tensor(scipy.linalg.fractional_matrix_power(S, -1 / 2), dtype=torch.float, device=device)
    S_t = torch.tensor(S, dtype=torch.float, device=device)
    h_orth = Smhalf @ torch.tensor(h_shifted, dtype=torch.float, device=device) @ Smhalf

    pos = np.array(coords_ang)[:, PERM].tolist()   # y,z,x for operators (matches loader)
    data_in = {
        'elements': elements,
        'pos': torch.tensor(pos, dtype=torch.float, device=device),
        'h': h_orth, 'S': S_t, 'Smhalf': Smhalf,
        'E_nn': E_nn, 'ne': ne_elec // 2, 'norbs': norb,
        'nuclearCharge': torch.tensor([Z[e] for e in elements], dtype=torch.float, device=device),
        'alpha_bp86': torch.tensor(alpha, dtype=torch.float, device=device),  # raw ORCA frame
        'name': 'chain',
    }
    data_obs = build_obs_mats(gb, elements, pos, Smhalf, device)
    return data_in, data_obs, baseline_E_elec


# ------------------------------- inference -------------------------------
def predict_one(est, data_in, data_obs, baseline_E_elec, device):
    OPS = ['E', 'x', 'y', 'z', 'alpha', 'E_gap']
    natm = len(data_in['elements'])
    dummy = {'atomic_charge': torch.zeros(natm), 'E_gap': 0.0,
             'B': torch.zeros(natm, natm), 'alpha': torch.zeros(3, 3),
             'F': torch.zeros(natm, 3)}
    my_sampler = sampler([data_in], [dummy], device)
    minibatch, _, ind = my_sampler.sample(batch_ind=0, batch_size=1,
                                          irreps=est.irreps, op_names=[])
    with torch.no_grad():
        mats_dict = est.inference(minibatch)
        r_mats = [torch.stack([data_obs[k] for k in ['x', 'y', 'z']])]
        info = {'alpha_bp86': minibatch['alpha_bp86'], 'r_mats': r_mats}
        state_calc = state_solver(minibatch['h'], mats_dict,
                                  minibatch['ne'], minibatch['norbs'], device, info=info)
        prop = est.predict_properties(state_calc, ind, OPS,
                                      op_matrices=[data_obs], charge_matrices=[],
                                      labels={'baseline_E': [baseline_E_elec]})
    alpha = prop['alpha'][0].detach().cpu().numpy()
    return {'alpha': alpha, 'alpha_iso': float(np.trace(alpha) / 3),
            'E': float(prop['E'][0]), 'E_gap': float(prop['E_gap'][0])}


def predict_orbital_one(est, data_in, data_obs, device):
    """Frontier-orbital diagnostics from the predicted Hamiltonian, on the SAME
    eigenstates the polarizability uses (state_solver of H = h + V). Returns the
    longitudinal HOMO->LUMO transition dipole |<psi_H|x|psi_L>| (the numerator of
    the sum-over-states polarizability) and the spatial spread / delocalization
    length sigma = sqrt(<x^2> - <x>^2) of the HOMO and LUMO. 'x' is the
    longitudinal (a_xx) axis, largest for a linear chain. All matrix elements use
    the orthonormalized AO operators in data_obs, in the same basis as co/cv."""
    natm = len(data_in['elements'])
    dummy = {'atomic_charge': torch.zeros(natm), 'E_gap': 0.0,
             'B': torch.zeros(natm, natm), 'alpha': torch.zeros(3, 3),
             'F': torch.zeros(natm, 3)}
    my_sampler = sampler([data_in], [dummy], device)
    minibatch, _, ind = my_sampler.sample(batch_ind=0, batch_size=1,
                                          irreps=est.irreps, op_names=[])
    with torch.no_grad():
        mats_dict = est.inference(minibatch)
        r_mats = [torch.stack([data_obs[k] for k in ['x', 'y', 'z']])]
        info = {'alpha_bp86': minibatch['alpha_bp86'], 'r_mats': r_mats}
        sc = state_solver(minibatch['h'], mats_dict,
                          minibatch['ne'], minibatch['norbs'], device, info=info)
        co, cv, eo, ev = sc.co[0], sc.cv[0], sc.eo[0], sc.ev[0]
        homo, lumo = co[:, -1], cv[:, 0]   # highest occ / lowest virt (eigh-sorted)

        def me(bra, op, ket):
            return float(bra @ (data_obs[op] @ ket))

        sq = {'x': 'xx', 'y': 'yy', 'z': 'zz'}
        mu = {ax: abs(me(homo, ax, lumo)) for ax in ('x', 'y', 'z')}

        def spread(c, ax):
            x1 = me(c, ax, c); x2 = me(c, sq[ax], c)
            return float(np.sqrt(max(x2 - x1 * x1, 0.0)))

        gap_Ha = float(ev[0] - eo[-1])
    return dict(mu_x=mu['x'], mu_y=mu['y'], mu_z=mu['z'],
                mu_mag=float(np.sqrt(mu['x']**2 + mu['y']**2 + mu['z']**2)),
                sigma_H=spread(homo, 'x'), sigma_L=spread(lumo, 'x'),
                gap_Ha=gap_Ha, gap_eV=gap_Ha * 27.2114)


# ------------------------------- modes -------------------------------
def mode_validate(est, gb, device, molid, nprocs):
    base = os.path.join(ROOT_DIR, 'dataset', 'P_S_Cl_dataset_v3', 'P_S_Cl_data_1_1000', 'basic')
    f = glob.glob(os.path.join(base, f'{molid}_*.json'))[0]
    basic = json.load(open(f))
    elements, coords = basic['elements'], basic['coordinates']
    print(f"[validate] mol {molid} {os.path.basename(f)} natoms={len(elements)}")
    wd = tempfile.mkdtemp(prefix=f'orca_hv8_{molid}_')
    try:
        data_in, data_obs, bE = generate_inputs(gb, elements, coords, device, wd, nprocs)
        pred = predict_one(est, data_in, data_obs, bE, device)
    finally:
        shutil.rmtree(wd, ignore_errors=True)

    tj = json.load(open(os.path.join(SCRIPT_DIR, 'output', 'test', 'test.json')))
    ids = [fn[1].split('_')[0] for fn in tj['file_names']]
    if molid in ids:
        i = ids.index(molid)
        ta = np.array(tj['alpha']['pred'][i])
        print(f"  alpha_iso : run_ham_chain={pred['alpha_iso']:.3f}   test.json={np.trace(ta)/3:.3f}")
        print(f"  E (Ha)    : run_ham_chain={pred['E']:.5f}   test.json={tj['E']['pred'][i]:.5f}")
        print(f"  E_gap(Ha) : run_ham_chain={pred['E_gap']:.5f}   test.json={tj['E_gap']['pred'][i]:.5f}")
        print(f"  alpha tensor (run):\n{np.round(pred['alpha'],3)}\n  alpha tensor (test.json):\n{np.round(ta,3)}")
    else:
        print(f"  alpha_iso={pred['alpha_iso']:.3f}  E={pred['E']:.5f}  E_gap={pred['E_gap']:.5f}")


def mode_series(est, gb, device, series, n_max, out_csv, nprocs, n_min=1):
    sys.path.insert(0, os.path.join(ROOT_DIR, 'examples', 'oligothiophene'))
    if series == 'oligothiophene':
        from build_oligothiophenes import OLIGOTHIOPHENE as BUILD, formula
    else:
        from build_polymers import CHLORO_POLYENE as BUILD
        from collections import Counter
        def formula(els):
            c = Counter(els)
            return ''.join(f'{k}{c[k]}' for k in sorted(c))
    rows = []
    for n in range(n_min, n_max + 1):
        elements, coords = BUILD(n)
        f = formula(elements)
        print(f"\n=== {series} n={n}  Natoms={len(elements)}  {f} ===", flush=True)
        wd = tempfile.mkdtemp(prefix=f'orca_hv8_{series}_n{n}_')
        try:
            data_in, data_obs, bE = generate_inputs(gb, elements, coords, device, wd, nprocs)
            pred = predict_one(est, data_in, data_obs, bE, device)
            a = pred['alpha']
            print(f"  a_xx={a[0][0]:.2f} a_yy={a[1][1]:.2f} a_zz={a[2][2]:.2f} "
                  f"a_iso={pred['alpha_iso']:.2f}  E={pred['E']:.4f}  "
                  f"E_gap={pred['E_gap']*27.2114:.3f} eV", flush=True)
            rows.append(dict(n=n, natoms=len(elements), formula=f,
                             alpha_xx_au=f"{a[0][0]:.4f}", alpha_yy_au=f"{a[1][1]:.4f}",
                             alpha_zz_au=f"{a[2][2]:.4f}", alpha_iso_au=f"{pred['alpha_iso']:.4f}",
                             E_Ha=f"{pred['E']:.6f}", E_gap_Ha=f"{pred['E_gap']:.6f}", error=""))
        except Exception as e:
            import traceback
            print(f"  FAILED n={n}: {e!r}\n{traceback.format_exc()}", flush=True)
            rows.append(dict(n=n, natoms=len(elements), formula=f, error=repr(e)))
        finally:
            shutil.rmtree(wd, ignore_errors=True)
            try:
                import gc; gc.collect(); torch.cuda.empty_cache()
            except Exception:
                pass
    cols = ["n", "natoms", "formula", "alpha_xx_au", "alpha_yy_au", "alpha_zz_au",
            "alpha_iso_au", "E_Ha", "E_gap_Ha", "error"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nWrote {out_csv}")


def mode_orbital_series(est, gb, device, series, n_max, out_csv, nprocs, n_min=1, append=False):
    """Sweep the homologous series and extract frontier-orbital diagnostics
    (HOMO->LUMO transition dipole and HOMO/LUMO delocalization length) from the
    predicted Hamiltonian -> orbital_scaling.csv. Verifies the <psi_H|x|psi_L> ~ L
    claim and that the local NN still yields delocalized frontier orbitals."""
    sys.path.insert(0, os.path.join(ROOT_DIR, 'examples', 'oligothiophene'))
    if series == 'oligothiophene':
        from build_oligothiophenes import OLIGOTHIOPHENE as BUILD, formula
    else:
        from build_polymers import CHLORO_POLYENE as BUILD
        from collections import Counter
        def formula(els):
            cc = Counter(els); return ''.join(f'{k}{cc[k]}' for k in sorted(cc))
    rows = []
    for n in range(n_min, n_max + 1):
        elements, coords = BUILD(n)
        f = formula(elements)
        print(f"\n=== {series} n={n}  Natoms={len(elements)}  {f} ===", flush=True)
        wd = tempfile.mkdtemp(prefix=f'orca_orb_{series}_n{n}_')
        try:
            data_in, data_obs, bE = generate_inputs(gb, elements, coords, device, wd, nprocs)
            o = predict_orbital_one(est, data_in, data_obs, device)
            print(f"  mu_x={o['mu_x']:.3f} (mag {o['mu_mag']:.3f})  "
                  f"sigma_H={o['sigma_H']:.3f}  sigma_L={o['sigma_L']:.3f}  "
                  f"gap={o['gap_eV']:.3f} eV", flush=True)
            rows.append(dict(n=n, natoms=len(elements), formula=f,
                             mu_x_au=f"{o['mu_x']:.4f}", mu_mag_au=f"{o['mu_mag']:.4f}",
                             sigma_H_au=f"{o['sigma_H']:.4f}", sigma_L_au=f"{o['sigma_L']:.4f}",
                             gap_eV=f"{o['gap_eV']:.4f}", error=""))
        except Exception as e:
            import traceback
            print(f"  FAILED n={n}: {e!r}\n{traceback.format_exc()}", flush=True)
            rows.append(dict(n=n, natoms=len(elements), formula=f, error=repr(e)))
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    cols = ["n", "natoms", "formula", "mu_x_au", "mu_mag_au", "sigma_H_au",
            "sigma_L_au", "gap_eV", "error"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    mode = 'a' if (append and os.path.exists(out_csv)) else 'w'
    with open(out_csv, mode, newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if mode == 'w':
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\n{'Appended to' if mode == 'a' else 'Wrote'} {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', default=None)
    ap.add_argument('--series', choices=['oligothiophene'], default=None)
    ap.add_argument('--n-max', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--nprocs', type=int, default=1)
    ap.add_argument('--out-csv', default=None)
    ap.add_argument('--orbitals', action='store_true',
                    help='extract frontier transition dipole + delocalization length (not alpha)')
    ap.add_argument('--n-min', type=int, default=1)
    ap.add_argument('--append', action='store_true',
                    help='append rows (resume) instead of overwriting the CSV')
    args = ap.parse_args()

    c = load_config()
    # ORCA baseline functional follows the model's delta_baseline (bp86 / b3lyp).
    bl = c["general"].get("delta_baseline", "bp86").strip().lower()
    if bl not in _DELTA2ORCA:
        raise SystemExit(f"unsupported delta_baseline={bl!r} for run_ham_chain")
    BASELINE['method'], BASELINE['align_key'] = _DELTA2ORCA[bl]
    print(f"[ham_chain] ORCA baseline = {BASELINE['method']} (align key {BASELINE['align_key']})")
    est, element_list = build_estimator(c, args.device)
    gb = generate_basic(args.device, ROOT_DIR + '/', element_list)

    if args.validate:
        mode_validate(est, gb, args.device, args.validate, args.nprocs)
    elif args.series and args.orbitals:
        out = args.out_csv or os.path.join(SCRIPT_DIR, f'results_hv8_{args.series}', 'orbital_scaling.csv')
        mode_orbital_series(est, gb, args.device, args.series, args.n_max, out, args.nprocs,
                            n_min=args.n_min, append=args.append)
    elif args.series:
        out = args.out_csv or os.path.join(SCRIPT_DIR, f'results_hv8_{args.series}', 'size_extrapolation.csv')
        mode_series(est, gb, args.device, args.series, args.n_max, out, args.nprocs, n_min=args.n_min)
    else:
        raise SystemExit("specify --validate <MOLID> or --series <name>")


if __name__ == '__main__':
    main()
