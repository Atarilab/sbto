import os
from typing import List, Optional, Tuple
import numpy as np
from datetime import datetime
from copy import copy, deepcopy

from sbto.utils.config import ConfigBase
from sbto.utils.plotting import plot_costs, plot_mean_cov, plot_state_control, plot_contact_plan
from sbto.utils.viewer import render_and_save_trajectory

EXP_DIR = "./runs"
TRAJ_FILENAME = "time_x_u_traj"
SOLVER_STATES_DIR = "./solver_states"

def get_date_time() -> str:
    now = datetime.now()
    return now.strftime('%Y_%m_%d__%H_%M_%S')

def create_dirs(exp_name: str, description: str = "") -> str:
    date = get_date_time()
    run_name = date if not description else f"{date}__{description}"
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

def load_trajectories(
    dir_path: str,
    ):
    file_path = os.path.join(dir_path, f"{TRAJ_FILENAME}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    data = np.load(file_path)
    return data["time"], data["x"], data["u"]

def save_results(
    dir_path: str,
    nlp,
    x_traj,
    u_traj,
    obs_traj,
    knots,
    all_solver_states,
    all_costs,
    ) -> None:

    # Save all solver states
    solver_state_dir = os.path.join(dir_path, SOLVER_STATES_DIR)
    for i, state in enumerate(all_solver_states):
        state.set_filename(f"solver_state_{i}.npz")
        state.save(solver_state_dir)

    time, state_traj = x_traj[:, 0], x_traj[:, 1:]

    save_trajectories(
        dir_path,
        time,
        x_traj,
        u_traj
    )

    mean_knots = all_solver_states[-1].mean
    cov_knots = all_solver_states[-1].cov
    plot_mean_cov(
        time,
        mean_knots,
        knots,
        cov_knots,
        u_traj,
        Nu=nlp.Nu,
        save_dir=dir_path,
    )

    plot_costs(
        all_costs,
        save_dir=dir_path
        )

    plot_state_control(
        time,
        state_traj,
        u_traj,
        knots,
        nlp.Nq,
        nlp.Nu,
        save_dir=dir_path
        )
    
    contact_realized = nlp.get_contact_status(obs_traj)

    if len(contact_realized) > 0:
        contact_realized[contact_realized > 1] = 1
        contact_plan = nlp.contact_plan if hasattr(nlp, "contact_plan") else None
        plot_contact_plan(
            contact_realized,
            contact_plan,
            dt=nlp.dt,
            save_dir=dir_path
        )

    render_and_save_trajectory(
        nlp.mj_model,
        nlp.mj_data,
        time,
        state_traj,
        save_path=dir_path
        )
    
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

def save_all_samples_and_cost(
    dir_path: str,
    samples,
    costs,
    ) -> None:
    np.savez(
        os.path.join(dir_path, f"all_samples_costs.npz"),
        samples=samples,
        costs=costs,
    )

def run_experiments(
    nlp,
    cfg_nlp,
    solver,
    cfg_solver,
    init_state_solver: Optional[str] = None,
    description: Optional[str] = None,
    ):

    if not isinstance(cfg_solver, list):
        cfg_solver = [cfg_solver]
    if not isinstance(cfg_nlp, list):
        cfg_nlp = [cfg_nlp]
    
    # run all configs
    for cfg_s in cfg_solver:
        for cfg_n in cfg_nlp:

            if not description is None: 
                # create run dir
                exp_name = nlp.__name__
                rundir = create_dirs(exp_name, description)
                # save configs
                for c in [cfg_n, cfg_s]:
                    c.save(rundir)

            # run optimization
            n = nlp(cfg_n)
            s = solver(nlp=n, cfg=cfg_s)
            if init_state_solver is None:
                state_solver = s.init_state()
            else:
                state_solver = init_state_solver
            solver_states, best_knots, cost, all_costs, all_samples = s.solve(deepcopy(state_solver))
            print("Best cost:", cost)

            # get final trajectories
            x_traj, qdes_traj, obs_traj = n.rollout(best_knots)
            x_traj = np.squeeze(x_traj)
            qdes_traj = np.squeeze(qdes_traj)
            obs_traj = np.squeeze(obs_traj)

            save_all_samples_and_cost(rundir, all_samples, all_costs)
            
            if not description is None:

                # save plots and video
                save_results(
                    rundir,
                    n,
                    x_traj,
                    qdes_traj,
                    obs_traj,
                    best_knots,
                    solver_states,
                    all_costs,
                )