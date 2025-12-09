import os
import numpy as np
from datetime import datetime
import yaml
import glob

EXP_DIR = "./datasets"
TRAJ_FILENAME = "time_x_u_traj"
ROLLOUT_FILENAME = "rollout_time_x_u_obs_traj"
SOLVER_STATES_DIR = "./solver_states"
ALL_SAMPLES_COSTS_FILENAME = "samples_costs"


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

def get_config_dict_from_rundir(run_dir: str):
    CONFIG_NAME = "config"
    all_cfg_yaml = glob.glob(
        f"{run_dir}/*/{CONFIG_NAME}.yaml",
        include_hidden=True,
        recursive=True
        )
    all_cfg_yaml += glob.glob(
        f"{run_dir}/*/*/{CONFIG_NAME}.yaml",
        include_hidden=True,
        recursive=True
        )
    all_cfg_yaml += glob.glob(
        f"{run_dir}/*{CONFIG_NAME}.yaml",
        include_hidden=True,
        recursive=True
        )
    if len(all_cfg_yaml) > 0:
        return load_yaml(all_cfg_yaml[0])
    else:
        return {}
    
def get_config_path_from_rundir(run_dir: str):
    CONFIG_NAME = "config"
    all_cfg_yaml = glob.glob(
        f"{run_dir}/*/{CONFIG_NAME}.yaml",
        include_hidden=True,
        recursive=True
        )
    all_cfg_yaml += glob.glob(
        f"{run_dir}/*/*/{CONFIG_NAME}.yaml",
        include_hidden=True,
        recursive=True
        )
    all_cfg_yaml += glob.glob(
        f"{run_dir}/*{CONFIG_NAME}.yaml",
        include_hidden=True,
        recursive=True
        )
    if len(all_cfg_yaml) > 0:
        return all_cfg_yaml[0]
    else:
        return ""
    
def get_xml_path_from_rundir(run_dir: str):
    all_xml_paths = glob.glob(
        f"{run_dir}/*/*.xml",
        include_hidden=True,
        recursive=True
        )
    all_xml_paths += glob.glob(
        f"{run_dir}/*/*/.xml",
        include_hidden=True,
        recursive=True
        )
    all_xml_paths += glob.glob(
        f"{run_dir}/*.xml",
        include_hidden=True,
        recursive=True
        )
    if len(all_xml_paths) > 0:
        return all_xml_paths[0]
    else:
        return ""
    
def get_date_time() -> str:
    now = datetime.now()
    return now.strftime('%Y_%m_%d__%H_%M_%S')

def create_dirs(exp_name: str, description: str = "") -> str:
    date = get_date_time()
    run_name = date if description == "" else f"{date}__{description}"
    exp_result_dir = os.path.join(EXP_DIR, exp_name, run_name)
    
    if os.path.exists(exp_result_dir):
        Warning(f"Directory {exp_result_dir} already exists.")
    else:
        os.makedirs(exp_result_dir)
    return exp_result_dir

def save_trajectories(
    dir_path: str,
    time,
    x_traj,
    u_traj
    ) -> None:

    np.savez(
        os.path.join(dir_path, f"{TRAJ_FILENAME}.npz"),
        time=time,
        x=x_traj,
        u=u_traj
    )

def save_rollout(
    dir_path: str,
    time,
    x_traj,
    u_traj,
    obs_traj,
    costs = []
    ) -> None:
    np.savez(
        os.path.join(dir_path, f"{ROLLOUT_FILENAME}.npz"),
        time=time,
        x=x_traj,
        u=u_traj,
        o=obs_traj,
        c=costs
    )

def save_all_samples_and_cost(
    dir_path: str,
    samples,
    costs,
    ) -> None:
    np.savez(
        os.path.join(dir_path, f"{ALL_SAMPLES_COSTS_FILENAME}.npz"),
        samples=samples,
        costs=costs,
    )

def save_all_states(
    dir_path: str,
    states
    ) -> None:
    # Save all solver states
    solver_state_dir = os.path.join(dir_path, SOLVER_STATES_DIR)
    for i, state in enumerate(states):
        state.set_filename(f"solver_state_{i}.npz")
        state.save(solver_state_dir)

def load_trajectories(
    dir_path: str,
    ):
    file_path = os.path.join(dir_path, f"{TRAJ_FILENAME}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    data = np.load(file_path)
    return data["time"], data["x"], data["u"]