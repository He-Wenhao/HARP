#!/usr/bin/env python3
"""
Repo entry point for end-to-end deployment:
  geometry -> ORCA(B3LYP/def2-SVP) baseline -> NN delta -> final prediction

Supports two input styles:
  1) CLI JSON strings: --elements_json, --coords_json
  2) A molecule JSON file: --molecule_json <path>

Molecule JSON schema (minimal):
{
  "elements": ["Cl","N",...],
  "coords_ang": [[x,y,z], ...]
}

Optional keys:
{
  "orca_bin": "orca",
  "nprocs": 1,
  "maxcore": 3000,
  "device": "cpu",
  "raw_delta": false,
  "config": "models/HARP/config.ini",
  "model": "models/HARP/output/model/1265_model.pt"
}

If raw_delta is true (CLI: --raw-delta), ORCA is skipped and printed values are NN
corrections only (no B3LYP baseline added in direct_predict_properties).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import configparser
import tempfile
from typing import Any


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import torch  # noqa: E402

from deploy.orca_b3lyp import OrcaRunConfig, run_orca_b3lyp_and_parse  # noqa: E402
from util_basis import read_basis  # noqa: E402
from train.train import trainer  # noqa: E402
from model.xyz2graph import sampler  # noqa: E402
from util_basis.Irreps import Irreps_build  # noqa: E402


HARTREE_TO_EV = 27.2114


def _load_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(path)
    return cfg


def _parse_backbone_config(config: configparser.ConfigParser):
    out = {}
    for k, v in config["backbone"].items():
        if k == "type":
            continue
        v = v.strip()
        if v.lower() in ("true", "false"):
            out[k] = config["backbone"].getboolean(k)
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _parse_loss_dropout_config(config: configparser.ConfigParser):
    if "loss_dropout" not in config:
        return {"enabled": False, "default_rate": 0.0, "rates": {}}

    section = config["loss_dropout"]
    enabled = section.getboolean("enabled", fallback=False)
    default_rate = section.getfloat("default_rate", fallback=0.0)
    rates = {}
    for k, v in section.items():
        if k in {"enabled", "default_rate"}:
            continue
        rates[k] = float(v)
    return {
        "enabled": enabled,
        "default_rate": default_rate,
        "rates": rates,
    }


def _make_minimal_basic(elements, pos_ang):
    irreps = Irreps_build(elements)
    nbasis = irreps.get_nbasis()
    ne_dic = irreps.get_ne()
    nuclearCharge = torch.tensor([ne_dic[e] for e in elements], dtype=torch.float)
    ne = int(round(sum([ne_dic[e] for e in elements]) / 2))
    norbs = int(round(sum([nbasis[e] for e in elements])))
    # h/S are not used by the direct-property backbones; provide well-formed placeholders.
    h = torch.zeros((norbs, norbs), dtype=torch.float)
    S = torch.eye(norbs, dtype=torch.float)
    Smhalf = torch.eye(norbs, dtype=torch.float)
    alpha_bp86 = torch.zeros((3, 3), dtype=torch.float)
    return {
        "pos": torch.tensor(pos_ang, dtype=torch.float),
        "elements": elements,
        "S": S,
        "Smhalf": Smhalf,
        "h": h,
        "E_nn": 0.0,
        "ne": ne,
        "norbs": norbs,
        "nuclearCharge": nuclearCharge,
        "alpha_bp86": alpha_bp86,
    }


def _make_dummy_label(natom: int):
    # Satisfy sampler expectations; these labels are not used for prediction.
    return {
        "E": 0.0,
        "E_nn": 0.0,
        "E_gap": 0.0,
        "atomic_charge": torch.zeros(natom, dtype=torch.float),
        "B": torch.zeros((natom, natom), dtype=torch.float),
        "alpha": torch.zeros((3, 3), dtype=torch.float),
        "F": torch.zeros((natom, 3), dtype=torch.float),
        "x": 0.0, "y": 0.0, "z": 0.0,
        "xx": 0.0, "yy": 0.0, "zz": 0.0,
        "xy": 0.0, "xz": 0.0, "yz": 0.0,
        "header": None,
    }


def _orca_baseline_to_training_b3lyp(
    baseline: dict[str, Any], elements: list[str]
) -> dict[str, Any]:
    """Convert ORCA-parsed baseline into the training-time `b3lyp` schema.

    See `src/data/loader.py::read_b3lyp` and `src/model/xyz2graph.py` for the
    expected keys and conventions.
    """
    out: dict[str, Any] = {}

    # Dipole and quadrupole components: ORCA parser already applies dataset
    # convention (store -electronic contribution), in atomic units.
    for k in ("x", "y", "z", "xx", "yy", "zz", "xy", "xz", "yz"):
        if k in baseline:
            out[k] = float(baseline[k])

    # Polarizability tensor (a.u.)
    if "alpha" in baseline:
        out["alpha"] = torch.tensor(baseline["alpha"], dtype=torch.float)

    # Mulliken charges: ORCA provides q_A (in e). Training uses electron population N_A = -q_A + Z_A.
    if "atomic_charge" in baseline:
        irreps = Irreps_build(elements)
        ne_dic = irreps.get_ne()
        Z = torch.tensor([ne_dic[e] for e in elements], dtype=torch.float)
        q = torch.tensor(baseline["atomic_charge"], dtype=torch.float)
        out["atomic_charge"] = (-q + Z)

    # Bond orders: ORCA provides a list [[a,b,bo], ...]; training baseline key is `B` matrix.
    if "bond_order" in baseline:
        n = len(elements)
        B = torch.zeros((n, n), dtype=torch.float)
        for a, b, bo in baseline["bond_order"]:
            a = int(a)
            b = int(b)
            bo = float(bo)
            B[a, b] = bo
            B[b, a] = bo
        out["B"] = B

    # HOMO-LUMO gap: ORCA parser returns eV; training baseline uses Hartree.
    if "Ee" in baseline:
        out["E_gap"] = float(baseline["Ee"]) / HARTREE_TO_EV

    # Energy for delta learning: electronic energy (total - nuclear repulsion), Hartree.
    if "energy" in baseline and "E_nn" in baseline:
        out["E"] = float(baseline["energy"]) - float(baseline["E_nn"])

    return out


def predict_end2end_orca_delta(
    *,
    config_path: str,
    model_path: str,
    elements: list[str],
    coords_ang: list[list[float]],
    device: str | None = None,
    orca_bin: str = "orca",
    nprocs: int = 1,
    maxcore: int = 3000,
    raw_delta: bool = False,
) -> dict:
    """
    Run the same end-to-end path as the CLI and return a JSON-serializable dict
    of predictions for a single molecule.

    If raw_delta is True, skip ORCA and return model deltas without adding B3LYP baselines.
    """
    config = _load_config(config_path)
    OPS = {k: float(v) for k, v in config["OPS"].items()}
    scaling = {k: float(v) for k, v in config["scaling"].items()}
    element_list = [x.strip() for x in config["element"]["list"].split(",")]
    backbone_type = config["backbone"].get("type", "internal").strip()
    backbone_config = _parse_backbone_config(config)
    loss_dropout_config = _parse_loss_dropout_config(config)
    nodeRDM_flag = config["general"].getboolean("nodeRDM_flag")
    # E_gap aggregation mode. Defaults to 'sum' (legacy) so v7_bp86 and
    # earlier checkpoints keep their original behaviour. v8+ should set
    # `egap_pooling = mean` in config.ini's [general] section.
    egap_pooling = config["general"].get("egap_pooling", "sum").strip().lower()
    # alpha head: 'sum' (legacy) or 'thole' (relay matrix, full alpha, no baseline)
    alpha_head = config["general"].get("alpha_head", "sum").strip().lower()
    # delta_baseline tells ORCA which DFT method to run at inference, so the
    # baseline the network sees is the SAME method it was trained against.
    # Older models that didn't have this key default to 'b3lyp' for
    # backwards compatibility.
    delta_baseline = config["general"].get("delta_baseline", "b3lyp").strip().lower()
    dev = (device or config["general"].get("device", "cpu")).strip()

    read_basis(path=os.path.join(ROOT_DIR, "src", "util_basis"), savefile=True)

    est = trainer(dev, nodeRDM_flag)
    est.build_model(
        scaling=scaling,
        element_list=element_list,
        backbone_type=backbone_type,
        backbone_config=backbone_config,
        alpha_head=alpha_head,
    )
    est.configure_loss_dropout(**loss_dropout_config)
    est.configure_egap_pooling(egap_pooling)
    est.load(model_path)
    est.model.eval()

    baseline = None
    if not raw_delta:
        # Map delta_baseline → ORCA functional name. The training pipeline
        # writes labels under obs_{b3lyp,bp86,DSD_PBEP86}/, so the keys here
        # must match those choices.
        _ORCA_METHOD = {"b3lyp": "B3LYP", "bp86": "BP86",
                        "dsd_pbep86": "DSD-PBEP86"}
        if delta_baseline not in _ORCA_METHOD:
            raise ValueError(
                f"unknown delta_baseline={delta_baseline!r}; expected one of "
                f"{list(_ORCA_METHOD)}")
        orca_cfg = OrcaRunConfig(
            orca_bin=orca_bin,
            nprocs=nprocs,
            maxcore_mb=maxcore,
            method=_ORCA_METHOD[delta_baseline],
        )
        with tempfile.TemporaryDirectory(prefix=f"orca_{delta_baseline}_") as td:
            baseline = run_orca_b3lyp_and_parse(elements, coords_ang, td, cfg=orca_cfg)

    basic = _make_minimal_basic(elements, coords_ang)
    label = _make_dummy_label(natom=len(elements))
    if baseline is not None:
        # Sampler keys delta-learning baselines under `baseline` (renamed
        # from `b3lyp` in commit e672d9b — see src/model/xyz2graph.py:151).
        label["baseline"] = _orca_baseline_to_training_b3lyp(baseline, elements)
    data = [basic]
    labels = [label]
    my_sampler = sampler(data, labels, dev)

    operators_electric = [k for k in OPS if k in ["x", "y", "z", "xx", "yy", "zz", "xy", "xz", "yz"]]
    minibatch, batch_labels, _ = my_sampler.sample(
        batch_ind=0, batch_size=1, irreps=est.irreps, op_names=operators_electric
    )
    minibatch["pos"] = minibatch["pos"][:, [1, 2, 0]].contiguous()
    with torch.no_grad():
        raw = est.model(minibatch)
    prop = est.direct_predict_properties(raw, batch_labels, OPS)

    def t2v(t):
        if isinstance(t, torch.Tensor):
            if t.numel() == 1:
                return float(t.detach().cpu())
            return t.detach().cpu().tolist()
        return t

    return {k: [t2v(v) for v in prop[k]] for k in prop}


def _load_molecule_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("molecule_json must contain a JSON object")
    if "elements" not in d or "coords_ang" not in d:
        raise ValueError('molecule_json must contain keys: "elements", "coords_ang"')
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to the model config.ini (e.g. models/HARP/config.ini)")
    ap.add_argument("--model", default=None, help="Path to model .pt (weights)")
    ap.add_argument("--elements_json", default=None, help="JSON list of element symbols")
    ap.add_argument("--coords_json", default=None, help="JSON list of Nx3 coordinates (Angstrom)")
    ap.add_argument("--molecule_json", default=None, help="Path to molecule JSON file")
    ap.add_argument(
        "--output_json",
        default=None,
        help="Write prediction JSON to this path (in addition to printing to stdout)",
    )
    ap.add_argument("--device", default=None, help="Override device (e.g. cpu or cuda:0)")
    ap.add_argument("--orca_bin", default=None, help="ORCA executable (default: orca)")
    ap.add_argument("--nprocs", type=int, default=None)
    ap.add_argument("--maxcore", type=int, default=None)
    ap.add_argument(
        "--raw-delta",
        action="store_true",
        help="Skip ORCA; print NN corrections only (no B3LYP baseline added to outputs)",
    )
    args = ap.parse_args()

    mol: dict[str, Any] = {}
    if args.molecule_json is not None:
        mol = _load_molecule_json(args.molecule_json)

    config_path = args.config or mol.get("config")
    model_path = args.model or mol.get("model")
    if config_path is None or model_path is None:
        raise SystemExit("--config and --model are required (or provide them in --molecule_json)")

    if args.molecule_json is not None:
        elements = mol["elements"]
        coords_ang = mol["coords_ang"]
    else:
        if args.elements_json is None or args.coords_json is None:
            raise SystemExit("Provide --elements_json and --coords_json, or use --molecule_json")
        elements = json.loads(args.elements_json)
        coords_ang = json.loads(args.coords_json)

    device = args.device or mol.get("device")
    orca_bin = args.orca_bin or mol.get("orca_bin") or "orca"
    nprocs = int(args.nprocs if args.nprocs is not None else mol.get("nprocs", 1))
    maxcore = int(args.maxcore if args.maxcore is not None else mol.get("maxcore", 3000))
    raw_delta = bool(args.raw_delta or mol.get("raw_delta", False))
    output_json = args.output_json or mol.get("output_json")

    out = predict_end2end_orca_delta(
        config_path=str(config_path),
        model_path=str(model_path),
        elements=elements,
        coords_ang=coords_ang,
        device=device,
        orca_bin=str(orca_bin),
        nprocs=nprocs,
        maxcore=maxcore,
        raw_delta=raw_delta,
    )

    if output_json:
        out_path = str(output_json)
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
