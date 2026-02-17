import os
import numpy as np
from datetime import datetime
import yaml
import glob
from omegaconf import OmegaConf
from sbto.data.filenames import (
    CONFIG_FILENAME,
    BEST_TRAJECTORY_FILENAME
)

EXP_DIR = "./datasets"
TRAJ_FILENAME = "time_x_u_traj"
ROLLOUT_FILENAME = "rollout_time_x_u_obs_traj"
SOLVER_STATES_DIR = "./solver_states"
ALL_SAMPLES_COSTS_FILENAME = "samples_costs"
OPT_STATS_FILENAME = "optimization_stats"

def get_filename_from_path(path: str):
    _, filename = os.path.split(path)
    filename, _ = os.path.splitext(filename)
    return filename

def load_yaml(yaml_path):
    d = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)
    return d

def get_config_path_from_rundir(run_dir: str):
    all_cfg_path = glob.glob(
        f"{run_dir}/**/{CONFIG_FILENAME}.yaml",
        include_hidden=True,
        recursive=True
        )
    if len(all_cfg_path) > 0:
        return all_cfg_path[0]
    else:
        return ""

def get_config_dict_from_rundir(run_dir: str):
    cfg_path = get_config_path_from_rundir(run_dir)
    if cfg_path:
        return load_yaml(cfg_path)
    else:
        return {}

def get_config_from_rundir(run_dir: str):
    cfg_dict = get_config_dict_from_rundir(run_dir)
    if cfg_dict:
        return OmegaConf.create(cfg_dict)
    else:
        return None
    
def get_opt_stats_path_from_rundir(run_dir: str):
    FILE_NAME = "optimization_stats"
    all_paths = glob.glob(
        f"{run_dir}/**__ws_incr/{FILE_NAME}.yaml",
        include_hidden=True,
        recursive=True
        )
    if len(all_paths) > 0:
        return all_paths[0]
    else:
        return ""
    
def get_xml_path_from_rundir(run_dir: str):
    all_xml_paths = glob.glob(
        f"{run_dir}/**/*.xml",
        include_hidden=True,
        recursive=True
        )
    if len(all_xml_paths) > 0:
        return all_xml_paths[0]
    else:
        return ""

def reconstruct_x_traj(data_dict):
    """
    Reconstruct original trajectory dictionary from split keys.
    """

    required_keys = [
        "base_xyz_quat",
        "actuator_pos",
        "obj_0_xyz_quat",
        "base_linvel_angvel",
        "actuator_vel",
        "obj_0_linvel_angvel",
    ]
    x_traj = []
    for k in required_keys:
        if k in data_dict:
            x_traj.append(data_dict[k])
    return np.hstack(x_traj)

def load_best_trajectory_from_rundir(rundir: str):
    data_path = os.path.join(rundir, f"{BEST_TRAJECTORY_FILENAME}.npz")
    data = dict(np.load(data_path))
    x_traj = reconstruct_x_traj(data)
    data["x"] = x_traj
    return data