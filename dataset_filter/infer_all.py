"""Run trained model on every datagroup (train+val+test); save per-molecule
per-property squared errors to dataset_filter/per_mol_errors.json.

Output format:
  { datagroup: { mol_id: { property: squared_error_scalar } } }

Properties: E, E_gap, dipole, quadrupole, atomic_charge, alpha, bond_order
"""
import os, sys, json, configparser
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
MODEL_DIR = os.environ.get('FILTER_MODEL_DIR', os.path.join(ROOT, 'models/HARP'))  # model used for the screen

sys.path.insert(0, os.path.join(ROOT, 'src'))
from data import dataloader
from util_basis import read_basis
from model.xyz2graph import sampler
from train.train import trainer

CM_TO_HARTREE = 4.55633528e-6


def parse_backbone_config(config):
    cfg = {}
    for k, v in config["backbone"].items():
        if k == "type":
            continue
        v = v.strip()
        if v.lower() in ("true", "false"):
            cfg[k] = config["backbone"].getboolean(k)
        else:
            try:
                cfg[k] = int(v)
            except ValueError:
                try:
                    cfg[k] = float(v)
                except ValueError:
                    cfg[k] = v
    return cfg


def t2v(t):
    if torch.is_tensor(t):
        if t.numel() == 1:
            return float(t)
        return t.detach().cpu().tolist()
    return t


def per_mol_error(pred, obs):
    """Squared error per property for one molecule."""
    err = {}

    # E (electronic energy, Hartree). Loader stores label as obs['energy'] - obs['E_nn'].
    if 'E' in pred and 'energy' in obs and 'E_nn' in obs:
        e_label = obs['energy'] - obs['E_nn']
        err['E'] = float((pred['E'] - e_label) ** 2)

    # E_gap: model in Hartree; obs Ee[0] in cm^-1
    if 'E_gap' in pred and isinstance(obs.get('Ee'), list) and obs['Ee']:
        gap_label = obs['Ee'][0] * CM_TO_HARTREE
        err['E_gap'] = float((pred['E_gap'] - gap_label) ** 2)

    # Dipole — mean over xyz
    if all(k in pred for k in 'xyz') and all(k in obs for k in 'xyz'):
        err['dipole'] = float(np.mean([(pred[k] - obs[k]) ** 2 for k in 'xyz']))

    # Quadrupole — mean over 6 components
    qkeys = ['xx', 'yy', 'zz', 'xy', 'yz', 'xz']
    if all(k in pred for k in qkeys) and all(k in obs for k in qkeys):
        err['quadrupole'] = float(np.mean([(pred[k] - obs[k]) ** 2 for k in qkeys]))

    # Atomic charge — mean over atoms
    if 'atomic_charge' in pred and 'atomic_charge' in obs:
        pc = np.asarray(pred['atomic_charge'], dtype=float)
        lc = np.asarray(obs['atomic_charge'], dtype=float)
        if pc.shape == lc.shape:
            err['atomic_charge'] = float(np.mean((pc - lc) ** 2))

    # Alpha — mean over 3x3 tensor
    if 'alpha' in pred and 'alpha' in obs:
        pa = np.asarray(pred['alpha'], dtype=float)
        la = np.asarray(obs['alpha'], dtype=float)
        if pa.shape == la.shape:
            err['alpha'] = float(np.mean((pa - la) ** 2))

    # Bond order — mean over labelled bonds (list of [i, j, value])
    if 'bond_order' in pred and 'bond_order' in obs:
        pb = np.asarray(pred['bond_order'], dtype=float)
        if pb.ndim == 2 and obs['bond_order']:
            diffs = []
            for b in obs['bond_order']:
                i, j, v = int(b[0]), int(b[1]), float(b[2])
                if i < pb.shape[0] and j < pb.shape[1]:
                    diffs.append((pb[i, j] - v) ** 2)
            if diffs:
                err['bond_order'] = float(np.mean(diffs))

    return err


def main():
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(os.path.join(MODEL_DIR, 'config.ini'))

    OPS = {k: float(v) for k, v in config["OPS"].items()}
    scaling = {k: float(v) for k, v in config["scaling"].items()}
    element_list = [x.strip() for x in config["element"]["list"].split(",")]
    backbone_type = config["backbone"].get("type", "internal").strip()
    backbone_config = parse_backbone_config(config)
    nodeRDM_flag = config["general"].getboolean("nodeRDM_flag")
    device = os.environ.get("DEVICE", "cuda:0")
    batch_size = int(os.environ.get("BATCH_SIZE", "50"))
    model_file = os.environ.get("MODEL_FILE", "2000_model.pt")

    train_groups = [x.strip() for x in config["data"]["datagroups"].split(",")]
    val_groups   = [x.strip() for x in config["data"]["val_datagroups"].split(",")]
    test_groups  = [x.strip() for x in config["test"]["datagroups"].split(",")]
    all_groups = test_groups + val_groups + train_groups

    print(f"Datagroups: {len(all_groups)} total")
    print(f"  test: {test_groups}")
    print(f"  val:  {val_groups}")
    print(f"  train: {train_groups}")
    print(f"Model file: {model_file}")

    basis_path = os.path.join(ROOT, 'src/util_basis')
    data_path = os.path.join(ROOT, 'dataset')
    read_basis(path=basis_path, savefile=True)

    est = trainer(device, nodeRDM_flag)
    est.build_model(scaling=scaling, element_list=element_list,
                    backbone_type=backbone_type, backbone_config=backbone_config)
    est.load(os.path.join(MODEL_DIR, 'output/model', model_file))
    est.model.eval()

    loader = dataloader(device=device, element_list=element_list,
                        path=data_path, starting_basis='def2-SVP')

    # Cap mols-per-chunk so the largest datagroups fit on GPU
    chunk_size = int(os.environ.get("CHUNK_SIZE", "4000"))
    per_group_dir = os.path.join(SCRIPT_DIR, '_per_group')
    os.makedirs(per_group_dir, exist_ok=True)

    def safe_name(g):
        return g.replace('/', '__')

    results = {}

    for group in all_groups:
        per_group_path = os.path.join(per_group_dir, f"{safe_name(group)}.json")
        if os.path.exists(per_group_path):
            with open(per_group_path) as f:
                results[group] = json.load(f)
            print(f"\n=== {group} ===  (cached: {len(results[group])} mols)")
            continue

        print(f"\n=== {group} ===")
        # Estimate total mols → decide # of chunks
        basic_dir = os.path.join(data_path, group, 'basic')
        n_total = sum(1 for f in os.listdir(basic_dir) if f.endswith('.json'))
        n_chunks = max(1, -(-n_total // chunk_size))
        print(f"  {n_total} molecules → {n_chunks} chunk(s) of ~{chunk_size}")

        # Load CCSD obs JSONs for the whole group once (small per-mol files)
        obs_dir = os.path.join(data_path, group, 'obs')
        obs_data = {}
        for fn in os.listdir(obs_dir):
            if fn.endswith('.json'):
                with open(os.path.join(obs_dir, fn)) as f:
                    obs_data[fn[:-5]] = json.load(f)

        group_errors = {}
        for chunk_idx in range(n_chunks):
            data, labels_all, obs_mats, charge_mats, fl = loader.load_data(
                [group], chunk_idx, n_chunks)
            my_sampler = sampler(data, labels_all, device)
            mol_ids = [f[1].split('_')[0] for f in fl]
            n = len(my_sampler)
            N_batches = -(-n // batch_size)
            print(f"  chunk {chunk_idx+1}/{n_chunks}: {n} mols, {N_batches} batches")

            preds = {}
            idx_offset = 0
            for batch_ind in range(N_batches):
                minibatch, labels, ind = my_sampler.sample(
                    batch_ind=batch_ind, batch_size=batch_size,
                    irreps=est.irreps, op_names=[])
                with torch.no_grad():
                    raw = est.model(minibatch)
                prop = est.direct_predict_properties(raw, labels, OPS)
                B = len(next(iter(prop.values())))
                for i in range(B):
                    gi = idx_offset + i
                    preds[gi] = {k: t2v(prop[k][i]) for k in prop}
                idx_offset += B

            for gi, mid in enumerate(mol_ids):
                if mid in obs_data and gi in preds:
                    group_errors[mid] = per_mol_error(preds[gi], obs_data[mid])

            del data, labels_all, obs_mats, charge_mats, my_sampler, preds
            torch.cuda.empty_cache()

        results[group] = group_errors
        with open(per_group_path, 'w') as f:
            json.dump(group_errors, f)
        print(f"  saved errors for {len(group_errors)} mols → {per_group_path}")

    out_path = os.path.join(SCRIPT_DIR, 'per_mol_errors.json')
    with open(out_path, 'w') as f:
        json.dump(results, f)
    total = sum(len(v) for v in results.values())
    print(f"\nSaved per-molecule errors for {total} molecules across "
          f"{len(results)} datagroups → {out_path}")


if __name__ == "__main__":
    main()
