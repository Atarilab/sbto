import os
from typing import List, Tuple
import numpy as np
from datetime import datetime
from copy import copy

from sbto.utils.config import ConfigBase
from sbto.utils.plotting import plot_costs, plot_mean_cov, plot_state_control, plot_contact_plan
from sbto.utils.viewer import render_and_save_trajectory

EXP_DIR = "./runs"
TRAJ_FILENAME = "time_x_u_traj"
SOLVER_STATE_DIR = "./solver_state"

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
    solver_state_dir = os.path.join(dir_path, SOLVER_STATE_DIR)
    for i, s in enumerate(all_solver_states):
        s.set_filename(f"solver_state_{i}.npz")
        s.save(solver_state_dir)

    time, state_traj = x_traj[:, 0], x_traj[:, 1:]

    save_trajectories(
        dir_path,
        time,
        x_traj,
        u_traj
    )

    plot_mean_cov(
        time,
        all_solver_states[-1].mean,
        knots,
        all_solver_states[-1].cov,
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
    if not isinstance(config.args[name], float):
        raise ValueError(f"Parameter {name} should be a float.")

    start, stop = range
    if not logscale:
        values = np.linspace(start, stop, num, endpoint=True)
    else:
        values = np.logspace(start, stop, num, endpoint=True)
    
    configs = []
    for v in values:
        cfg = copy(config)
        cfg.__setattr__(name, v)
        configs.append(cfg)
    
    return configs

def run_experiments(
    nlp,
    cfg_nlp,
    solver,
    cfg_solver,
    init_state_solver,
    description: str,
    ):

    if not isinstance(cfg_solver, list):
        cfg_solver = [cfg_solver]
    if not isinstance(cfg_nlp, list):
        cfg_nlp = [cfg_nlp]
    
    # run all configs
    for cfg_s in cfg_solver:
        for cfg_n in cfg_nlp:

            # create run dir
            exp_name = nlp.__name__
            rundir = create_dirs(exp_name, description)
            # save configs
            for c in [cfg_n, cfg_s]:
                c.save(rundir)

            # run optimization
            n = nlp(cfg_n)
            s = solver(nlp=n, cfg=cfg_s)
            solver_states, best_knots, cost, all_costs = s.solve(init_state_solver)

            # get final trajectories
            print("Best cost:", cost)
            x_traj, u_traj, obs_traj, cost = s.evaluate(best_knots)
            print("evaluate cost:", cost)

            # save plots and video
            save_results(
                rundir,
                n,
                x_traj,
                u_traj,
                obs_traj,
                best_knots,
                solver_states,
                all_costs,
            )
            



