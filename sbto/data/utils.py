import os
import numpy as np
from datetime import datetime
import os
from typing import List, Optional, Tuple
import numpy as np
from copy import copy

from sbto.utils.config import ConfigBase

EXP_DIR = "./datasets"
TRAJ_FILENAME = "time_x_u_traj"
SOLVER_STATES_DIR = "./solver_states"
ALL_SAMPLES_COSTS_FILENAME = "samples_costs"

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

def sweep_param(
    config: ConfigBase,
    name: str,
    range: Tuple,
    num: int,
    axis: int | tuple = 0,
    logscale: bool = False
    ) -> List[ConfigBase]:
    """
    Returns a list of config with the parameter values
    swept over the desired range.
    """
    # Check param in in config
    if not name in config.args:
        raise ValueError(f"Parameter {name} not found.")
    # Check param is float
    param = config.args[name]
    arr_param = np.asarray(param)

    has_multiple_dim = np.sum(arr_param.shape) > 1
    is_tuple = isinstance(param, tuple)
    is_list = isinstance(param, list)
    is_float = isinstance(param, float)
    is_int = isinstance(param, int)
    if not any((is_float, is_int, is_tuple, is_list)):
        raise ValueError(f"Parameter {name} should be a float, an int or an iterable.")

    start, stop = range
    if not logscale:
        sweep_values = np.linspace(start, stop, num, endpoint=True)
    else:
        sweep_values = np.logspace(start, stop, num, endpoint=True)

    configs = []
    for v in sweep_values:
        cfg = copy(config)
        if is_int:
            v = int(v)
        elif is_float:
            v = float(v)
        elif is_tuple:
            if has_multiple_dim:
                arr = arr_param.copy()
                arr[axis] = v
                v = tuple(arr.tolist())
            else:
                v = (v,)
        elif is_list:
            if has_multiple_dim:
                arr = arr_param.copy()
                arr[axis] = v
                v = arr.tolist()
            else:
                v = [v]

        cfg.__setattr__(name, v)
        configs.append(cfg)
    
    return configs
