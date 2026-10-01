import argparse
import random
import numpy as np
import os
import configparser
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# e3nn's constants.pt needs `slice` to be allowlisted for PyTorch 2.6+
torch.serialization.add_safe_globals([slice])

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

sys.path.append(os.path.join(ROOT_DIR, 'src'))
from train import main


def load_params(config_file: str):
    assert os.path.exists(config_file), f"Config not found: {config_file}"
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file)
    params = {}

    # OPS and Scaling
    for section in ["OPS", "scaling"]:
        params[section] = {k: float(v) for k, v in config[section].items()}

    # General
    type_map = {
        "batch_size": int,
        "steps_per_epoch": int,
        "N_epoch": int,
        "N_batches": int,
        "lr_init": float,
        "lr_final": float,
        "lr_decay_steps": int,
        "Nsave": int,
        "world_size": int,
        "nodeRDM_flag": bool,
        "load_model": bool,
        "delta_learning": bool,
    }

    for k, v in config["general"].items():
        if k in type_map:
            caster = type_map[k]
            if caster is bool:
                params[k] = config["general"].getboolean(k)
            else:
                params[k] = caster(v)
        else:
            params[k] = v

    # Elements
    params["element_list"] = [x.strip() for x in config["element"]["list"].split(",")]

    # Data groups
    params["datagroup"] = [x.strip() for x in config["data"]["datagroups"].split(",")]
    params["val_datagroup"] = [x.strip() for x in config["data"]["val_datagroups"].split(",")]

    # Backbone selection
    params["backbone_type"] = "internal"
    params["backbone_config"] = {}
    if "backbone" in config:
        params["backbone_type"] = config["backbone"].get("type", "internal").strip()
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
        params["backbone_config"] = cfg

    # Paths
    params["path"] = ROOT_DIR + "/"
    output_name = params.get("output_name", "output")
    params["output_path"] = os.path.join(ROOT_DIR, output_name)

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ESEN ablation model")
    parser.add_argument("config", nargs="?", default=os.path.join(SCRIPT_DIR, "config.ini"))
    args = parser.parse_args()
    config_path = args.config
    params = load_params(config_path)
    if params.get("ddp_mode") == "elastic":
        main(params)
