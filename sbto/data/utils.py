import os
import numpy as np
from datetime import datetime
import yaml
import glob
from omegaconf import OmegaConf

from sbto.run.stats import PERF_FILENAME
from sbto.data.filenames import (
    CONFIG_FILENAME,
    BEST_TRAJECTORY_FILENAME,
    DATA_DIR,
    SOLVER_STATE_NAME
)

def get_date_time() -> str:
    now = datetime.now()
    return now.strftime('%Y_%m_%d__%H_%M_%S')

def create_dirs(exp_name: str, data_dir: str = "", description: str = "") -> str:
    date = get_date_time()
    run_name = date if description == "" else f"{date}__{description}"
    if data_dir == "":
        data_dir = DATA_DIR
    exp_result_dir = os.path.join(data_dir, exp_name, run_name)
    
    if os.path.exists(exp_result_dir):
        Warning(f"Directory {exp_result_dir} already exists.")
    else:
        os.makedirs(exp_result_dir)
    return exp_result_dir

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
    
def get_arg_from_cfg_dict(cfg_dict: dict, key: str):
    for k, v in cfg_dict.items():
        if k == key:
            return v
        elif isinstance(v, dict):
            result = get_arg_from_cfg_dict(v, key)
            if result is not None:
                return result
    return None
             
def get_opt_stats_path_from_rundir(run_dir: str):
    all_paths = glob.glob(
        f"{run_dir}/**/{PERF_FILENAME}.yaml",
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

def get_all_best_traj_data(task_dir: str):
    all_traj_data_paths = glob.glob(
        f"{task_dir}/**/{BEST_TRAJECTORY_FILENAME}.npz",
        recursive=True
    )
    return all_traj_data_paths

def solver_state_path_from_rundir(rundir: str, suffix: str = "") -> str:    
    if suffix:
        filename = f"{SOLVER_STATE_NAME}_{suffix}.npz"
    else:
        filename = f"{SOLVER_STATE_NAME}.npz"
    return os.path.join(rundir, filename)

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
            print(data_dict[k].shape)
            x_traj.append(data_dict[k])
    return np.concatenate(x_traj, axis=-1)

def load_best_trajectory_from_rundir(rundir: str, with_full_state: bool = True):
    data_path = os.path.join(rundir, f"{BEST_TRAJECTORY_FILENAME}.npz")
    data = dict(np.load(data_path))
    if with_full_state:
        x_traj = reconstruct_x_traj(data)
        data["x"] = x_traj
    return data