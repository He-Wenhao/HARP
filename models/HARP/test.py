"""Evaluate HARP (Hamiltonian-head model) on the test datagroups in config.ini.

Unlike models/v8/test.py (which uses the legacy `direct_predict_properties`
path), this mirrors the training-time hamiltonian inference pipeline:

    model forward -> to_mat.raw_to_mat -> state_solver(H = h + V) -> predict_properties

That pipeline is exactly what src/deploy/deploy_cls.py::estimator_test.solve()
implements, so we reuse it here. All parameters are read from config.ini.

Output goes to models/HARP/output/test/test.json.
"""
import sys
import os
import re
import json
import glob
import configparser
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from data import dataloader
from util_basis import read_basis
from model.xyz2graph import sampler
from deploy.deploy_cls import estimator_test
from train.solve_H import state_solver


def load_config(config_file):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file)
    return config


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


def resolve_model_file(model_dir, requested):
    """Use the configured checkpoint if it exists, else fall back to the
    highest-numbered NNN_model.pt in the model dir."""
    cand = os.path.join(model_dir, requested)
    if os.path.exists(cand):
        return cand, requested
    numbered = []
    for p in glob.glob(os.path.join(model_dir, "*_model.pt")):
        m = re.match(r"(\d+)_model\.pt$", os.path.basename(p))
        if m:
            numbered.append((int(m.group(1)), p))
    if not numbered:
        raise FileNotFoundError(
            f"Neither {requested!r} nor any NNN_model.pt found in {model_dir}")
    numbered.sort()
    epoch, path = numbered[-1]
    print(f"[test] configured model_file '{requested}' not found; "
          f"falling back to latest checkpoint {os.path.basename(path)} (epoch {epoch})")
    return path, os.path.basename(path)


if __name__ == "__main__":
    config_path = os.path.join(SCRIPT_DIR, "config.ini")
    config = load_config(config_path)

    # ---- general / model config ----
    OPS = {k: float(v) for k, v in config["OPS"].items()}
    scaling = {k: float(v) for k, v in config["scaling"].items()}
    element_list = [x.strip() for x in config["element"]["list"].split(",")]
    backbone_type = config["backbone"].get("type", "internal").strip()
    backbone_config = parse_backbone_config(config)
    nodeRDM_flag = config["general"].getboolean("nodeRDM_flag")
    delta_learning = config["general"].getboolean("delta_learning")
    delta_baseline = config["general"].get("delta_baseline", "b3lyp").strip()
    egap_pooling = config["general"].get("egap_pooling", "sum").strip().lower()
    alpha_head = config["general"].get("alpha_head", "sum").strip().lower()
    head_type = config["general"].get("head_type", "direct").strip().lower()
    device = config["general"].get("device", "cuda:0").strip()

    # ---- test-specific ----
    test_datagroups = [x.strip() for x in config["test"]["datagroups"].split(",")]
    requested_model = config["test"].get("model_file", "latest_checkpoint.pt").strip()
    batch_size = config["test"].getint("batch_size", fallback=50)

    output_dir = os.path.join(SCRIPT_DIR, "output")
    model_dir = os.path.join(output_dir, "model")
    test_out_dir = os.path.join(output_dir, "test")
    os.makedirs(test_out_dir, exist_ok=True)
    path = ROOT_DIR + "/"

    model_path, model_name = resolve_model_file(model_dir, requested_model)
    print(f"[test] head_type={head_type} backbone={backbone_type} "
          f"lmax={backbone_config.get('lmax_list')} device={device}")
    print(f"[test] checkpoint={model_name}")

    # ---- build estimator (subclass of trainer with ham-aware solve()) ----
    est = estimator_test(device, nodeRDM_flag, output_folder=test_out_dir)
    est.build_model(scaling=scaling, element_list=element_list,
                    backbone_type=backbone_type, backbone_config=backbone_config,
                    alpha_head=alpha_head, head_type=head_type)
    est.configure_egap_pooling(egap_pooling)
    est.configure_gap_rescale(config["general"].get("gap_rescale", "affine").strip().lower())
    est.load(model_path)
    est.model.eval()

    # ---- load test data (load_obs=True => keep AO operator/charge matrices) ----
    basis_path = path + 'src/util_basis'
    data_path = path + 'dataset'
    read_basis(path=basis_path, savefile=True)

    loader = dataloader(device=device, element_list=element_list,
                        path=data_path, starting_basis='def2-SVP')
    # Must mirror the training config: a B3LYP-baseline model needs the
    # basic_B3LYP Fock/overlap/alpha, not the default (BP86) `basic`. Without
    # this the model is fed the wrong baseline and every Hamiltonian-derived
    # property (gap, dipole, quadrupole, alpha, charges) comes out garbage.
    # Build basic_B3LYP/ with scripts/build_basic_b3lyp.py (see README);
    # check_basic_matches_baseline() below stops the run if this is wrong.
    loader.basic_subdir = config["general"].get("basic_subdir", "basic").strip()
    loader.tmp_subdir = config["general"].get("tmp_subdir", "tmp").strip()
    data, labels_all, obs_mats, charge_mats, fl = loader.load_data(
        test_datagroups, 0, 1,
        delta_learning=delta_learning, delta_baseline=delta_baseline,
        load_obs=(head_type == 'hamiltonian'))
    if delta_learning:
        loader.check_basic_matches_baseline(data, fl, delta_baseline)
    my_sampler = sampler(data, labels_all, device)

    n_molecules = len(my_sampler)
    N_batches = -(-n_molecules // batch_size)
    print(f"[test] {n_molecules} molecules, batch_size={batch_size}, "
          f"N_batches={N_batches}")

    # ---- run hamiltonian inference pipeline ----
    # NOTE: we mirror deploy_cls.estimator_test.solve() but pass `labels` into
    # predict_properties so the energy head uses the delta-learning BP86 baseline
    # (labels['baseline_E']) exactly as during training. solve() omits labels,
    # which silently falls back to the orbital-energy-sum baseline (lc.e_h) and
    # makes the energy prediction wrong by ~0.5 Ha.
    operators_electric = [k for k in OPS
                          if k in ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz']]

    def t2v(tensor):
        return float(tensor) if tensor.dim() == 0 else tensor.tolist()

    properties_dic = {}
    with torch.no_grad():
        for batch_ind in range(N_batches):
            minibatch, labels, ind = my_sampler.sample(
                batch_ind=batch_ind, batch_size=batch_size,
                irreps=est.irreps, op_names=operators_electric)

            mats_dict = est.inference(minibatch)
            # Pass info={'alpha_bp86','r_mats'} so the Thole relay matrix uses the
            # BP86 polarizability baseline (T0). Without it state_solver sets T0=0
            # and alpha drops the ~50 a.u. baseline. Mirrors train.py:761-765.
            r_mats = [torch.stack([obs_mats[ii][k] for k in ['x', 'y', 'z']])
                      for ii in ind]
            info = {'alpha_bp86': minibatch['alpha_bp86'], 'r_mats': r_mats}
            state_calc = state_solver(minibatch['h'], mats_dict,
                                      minibatch['ne'], minibatch['norbs'], device,
                                      info=info)
            prop = est.predict_properties(
                state_calc, ind, OPS,
                op_matrices=obs_mats, charge_matrices=charge_mats, labels=labels)

            for key in prop:
                if key not in properties_dic:
                    properties_dic[key] = {'pred': [], 'label': []}
                properties_dic[key]['pred'] += [t2v(v) for v in prop[key]]
                if key == 'E':
                    properties_dic[key]['label'] += labels['Ee']
                elif key == 'bond_order':
                    properties_dic[key]['label'] += [u.tolist() for u in labels['B']]
                elif key in ('atomic_charge', 'alpha', 'F'):
                    properties_dic[key]['label'] += [u.tolist() for u in labels[key]]
                elif key == 'V':
                    continue
                else:
                    properties_dic[key]['label'] += labels[key]

            print(f'complete {batch_ind + 1} of {N_batches}')

    properties_dic['file_names'] = fl

    out_file = os.path.join(test_out_dir, 'test.json')
    with open(out_file, 'w') as f:
        json.dump(properties_dic, f)

    print(f'\nTest complete. Results saved to {out_file}')
    print(f'Properties predicted: '
          f'{[k for k in properties_dic if k != "file_names"]}')
