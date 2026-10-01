"""GPU timing of the released HARP (Hamiltonian-head) inference, per molecule.

Mirrors models/HARP/test.py at batch size 1 and times, bracketed by
cuda.synchronize: (i) the network forward alone (est.inference) and (ii) the
full neural inference = forward + state_solver (diagonalization, screened
response) + predict_properties, i.e. everything the model adds on top of the
B3LYP/def2-SVP baseline. Graph construction / data loading are excluded.
Writes models/HARP/output/test/inference_timing.json.

Run on a GPU node:  python scripts/measure_inference_time.py

"""
import sys
import os
import re
import json
import glob
import configparser
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SCRIPT_DIR = os.path.join(ROOT_DIR, 'models', 'HARP')
OUT_JSON = os.path.join(SCRIPT_DIR, 'output', 'test', 'inference_timing.json')
import time

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
    loader.basic_subdir = config["general"].get("basic_subdir", "basic").strip()
    loader.tmp_subdir = config["general"].get("tmp_subdir", "tmp").strip()
    data, labels_all, obs_mats, charge_mats, fl = loader.load_data(
        test_datagroups, 0, 1,
        delta_learning=delta_learning, delta_baseline=delta_baseline,
        load_obs=(head_type == 'hamiltonian'))
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

    N_WARM, N_REP = 5, 3

    def run_one(i):
        minibatch, labels, ind = my_sampler.sample(
            batch_ind=i, batch_size=1, irreps=est.irreps, op_names=operators_electric)
        r_mats = [torch.stack([obs_mats[ii][k] for k in ['x', 'y', 'z']]) for ii in ind]
        info = {'alpha_bp86': minibatch['alpha_bp86'], 'r_mats': r_mats}
        torch.cuda.synchronize(device); t0 = time.perf_counter()
        mats_dict = est.inference(minibatch)
        torch.cuda.synchronize(device); t1 = time.perf_counter()
        state_calc = state_solver(minibatch['h'], mats_dict, minibatch['ne'],
                                  minibatch['norbs'], device, info=info)
        est.predict_properties(state_calc, ind, OPS, op_matrices=obs_mats,
                               charge_matrices=charge_mats, labels=labels)
        torch.cuda.synchronize(device); t2 = time.perf_counter()
        return t1 - t0, t2 - t0

    per_mol = []
    with torch.no_grad():
        for _ in range(N_WARM):
            run_one(0)
        for i in range(n_molecules):
            ts = [run_one(i) for _ in range(N_REP)]
            per_mol.append({"fname": fl[i][1], "natom": int(len(data[i]["elements"])),
                            "t_forward_sec": min(t[0] for t in ts),
                            "t_total_sec": min(t[1] for t in ts)})
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{n_molecules}] fwd {per_mol[-1]['t_forward_sec']*1e3:.1f} ms"
                      f"  total {per_mol[-1]['t_total_sec']*1e3:.1f} ms", flush=True)
    out = {"device": device, "gpu_name": torch.cuda.get_device_name(device),
           "model_dir": SCRIPT_DIR, "model_file": model_name, "n_warmup": N_WARM,
           "n_repeat": N_REP, "per_molecule": per_mol}
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    import numpy as np
    f = np.array([p["t_forward_sec"] for p in per_mol]) * 1e3
    t = np.array([p["t_total_sec"] for p in per_mol]) * 1e3
    print(f"forward ms: min {f.min():.1f} median {np.median(f):.1f} max {f.max():.1f}")
    print(f"total   ms: min {t.min():.1f} median {np.median(t):.1f} max {t.max():.1f}")
    print("wrote", OUT_JSON)
