import os
import numpy as np
from datetime import datetime
import os
import shutil
import numpy as np
import numpy.typing as npt
from dataclasses import asdict
from typing import List
import copy

from sbto.tasks.task_mj import TaskMj
from sbto.sim.sim_base import SimRolloutBase
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_base import OCPBase
from sbto.solvers.solver_base import SolverState, SamplingBasedSolver
from sbto.utils.plotting import plot_contact_plan, plot_costs, plot_mean_cov, plot_state_control
from sbto.utils.viewer import render_and_save_trajectory

Array = npt.NDArray[np.float64]

EXP_DIR = "./datasets"
TRAJ_FILENAME = "time_x_u_traj"
ROLLOUT_FILENAME = "rollout_time_x_u_obs_traj"
SOLVER_STATES_DIR = "./solver_states"
ALL_SAMPLES_COSTS_FILENAME = "samples_costs"
SOLVER_STATE_NAME = "solver_state"
INITIAL_SOLVER_STATE_SUFFIX = "0"
FINAL_SOLVER_STATE_SUFFIX = "final"
HYDRA_CFG = ".hydra"

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

def get_solver_state_path(dir_path: str, suffix: str) -> SolverState:
    ext = "npz"
    filename = f"{SOLVER_STATE_NAME}.{ext}"

    if suffix:
        filename = filename.replace(f".{ext}", f"_{suffix}.{ext}")
    return filename


def save_all_states(
    dir_path: str,
    states: List[SolverState]
    ) -> None:
    for i, state in enumerate(states):
        save_solver_state(dir_path, state, str(i))

def save_solver_state(
    dir_path: str,
    state: SolverState,
    suffix: str = ""
    ) -> None:
    filename = get_solver_state_path(dir_path, suffix)
    solver_state_file = os.path.join(dir_path, filename)
    np.savez(solver_state_file, **asdict(state))

def _get_state_from_rundir(dir_path: str, solver: SamplingBasedSolver, suffix: str) -> SolverState:
    filename = get_solver_state_path(dir_path, suffix)
    solver_state_file = os.path.join(dir_path, filename)
    solver_state_0 = copy.deepcopy(solver.state)
    data = np.load(solver_state_file)
    for k, v in data.items():
        setattr(solver_state_0, k, v)
    return solver_state_0

def get_initial_state_from_rundir(dir_path: str, solver: SamplingBasedSolver) -> SolverState:
    return _get_state_from_rundir(dir_path, solver, INITIAL_SOLVER_STATE_SUFFIX)

def get_final_state_from_rundir(dir_path: str, solver: SamplingBasedSolver) -> SolverState:
    return _get_state_from_rundir(dir_path, solver, FINAL_SOLVER_STATE_SUFFIX)

def save_plots(
    dir_path: str,
    task: TaskMj,
    time,
    x_traj,
    u_traj,
    obs_traj,
    knots,
    mean_knots,
    cov_knots,
    all_costs,
    ) -> None:

    Nu, Nq = task.mj_scene.Nu, task.mj_scene.Nq

    plot_mean_cov(
        time,
        mean_knots,
        knots,
        cov_knots,
        u_traj,
        Nu=Nu,
        save_dir=dir_path,
    )

    plot_costs(
        all_costs,
        save_dir=dir_path
        )

    plot_state_control(
        time,
        x_traj,
        u_traj,
        knots,
        Nq,
        Nu,
        save_dir=dir_path
        )
    
    contact_realized = task.get_contact_status(obs_traj)

    if len(contact_realized) > 0:
        contact_realized[contact_realized > 1] = 1
        contact_plan = task.contact_plan if hasattr(task, "contact_plan") else None
        plot_contact_plan(
            contact_realized,
            contact_plan,
            dt=task.mj_scene.dt,
            save_dir=dir_path
        )

def copy_hydra_config(hydra_rundir: str, dst_path: str):
    hydra_cfg_dir = f"{hydra_rundir}/{HYDRA_CFG}" if hydra_rundir else ""
    if os.path.exists(hydra_cfg_dir):
        shutil.copytree(hydra_cfg_dir, f"{dst_path}/{HYDRA_CFG}")

def save_results(
    sim: SimMjRollout,
    task: OCPBase,
    solver_state_0: SolverState,
    solver_state_final: SolverState,
    all_samples: Array,
    all_costs: Array,
    description: str = "",
    hydra_rundir: str = "",
    save_fig: bool = True,
    multiple_shooting: bool = False,
    ) -> str:
    task_name = task.__class__.__name__
    result_dir = create_dirs(task_name, description)

    print(f"[{description or 'Unnamed'}] Best cost: {solver_state_final.min_cost_all}")
    best_knots = solver_state_final.best_all
    # Get best traj
    if multiple_shooting:
        x_shooting = task.ref.x[sim.t_knots]
        t, x_traj, qdes_traj, obs_traj = map(np.squeeze, sim.rollout_multiple_shooting(best_knots, x_shooting, with_x0=True))
    else:
        t, x_traj, qdes_traj, obs_traj = map(np.squeeze, sim.rollout(best_knots, with_x0=True))

    save_trajectories(result_dir, t, x_traj, qdes_traj)
    save_all_samples_and_cost(result_dir, all_samples, all_costs)
    copy_hydra_config(hydra_rundir, result_dir)
    if solver_state_0:
        save_solver_state(result_dir, solver_state_0, INITIAL_SOLVER_STATE_SUFFIX)
    save_solver_state(result_dir, solver_state_final, FINAL_SOLVER_STATE_SUFFIX)

    if save_fig:
        save_plots(
            result_dir,
            task,
            t,
            x_traj,
            qdes_traj,
            obs_traj,
            best_knots,
            solver_state_final.mean,
            solver_state_final.cov,
            all_costs,
        )
        render_and_save_trajectory(
            sim.mj_scene.mj_model,
            sim.mj_scene.mj_data,
            t,
            x_traj,
            save_path=result_dir,
        )

    return result_dir