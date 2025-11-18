import os
import numpy as np
from datetime import datetime
import os
import shutil
import numpy as np
import numpy.typing as npt

from sbto.tasks.task_mj import TaskMj
from sbto.sim.sim_base import SimRolloutBase
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_base import OCPBase
from sbto.solvers.solver_base import SolverState
from sbto.utils.plotting import plot_contact_plan, plot_costs, plot_mean_cov, plot_state_control
from sbto.utils.viewer import render_and_save_trajectory

Array = npt.NDArray[np.float64]

EXP_DIR = "./datasets"
TRAJ_FILENAME = "time_x_u_traj"
ROLLOUT_FILENAME = "rollout_time_x_u_obs_traj"
SOLVER_STATES_DIR = "./solver_states"
ALL_SAMPLES_COSTS_FILENAME = "samples_costs"
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

def save_all_states(
    dir_path: str,
    states
    ) -> None:
    # Save all solver states
    solver_state_dir = os.path.join(dir_path, SOLVER_STATES_DIR)
    for i, state in enumerate(states):
        state.set_filename(f"solver_state_{i}.npz")
        state.save(solver_state_dir)

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
    solver_state: SolverState,
    all_samples: Array,
    all_costs: Array,
    description: str = "",
    hydra_rundir: str = "",
    save_fig: bool = True,
    ) -> None:
    task_name = task.__class__.__name__
    result_dir = create_dirs(task_name, description)

    print(f"[{description or 'Unnamed'}] Best cost: {solver_state.min_cost_all}")

    best_knots = solver_state.best
    # Get best traj
    t, x_traj, qdes_traj, obs_traj = map(np.squeeze, sim.rollout(best_knots, with_x0=True))

    save_trajectories(result_dir, t, x_traj, qdes_traj)
    save_all_samples_and_cost(result_dir, all_samples, all_costs)
    copy_hydra_config(hydra_rundir, result_dir)

    if save_fig:
        save_plots(
            result_dir,
            task,
            t,
            x_traj,
            qdes_traj,
            obs_traj,
            best_knots,
            solver_state.mean,
            solver_state.cov,
            all_costs,
        )
        render_and_save_trajectory(
            sim.mj_scene.mj_model,
            sim.mj_scene.mj_data,
            t,
            x_traj,
            save_path=result_dir,
        )