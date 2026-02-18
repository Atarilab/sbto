import os
import numpy as np
import shutil
import numpy.typing as npt
from dataclasses import asdict
from typing import List
import mujoco

from sbto.tasks.task_mj import TaskMj
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_base import OCPBase
from sbto.solvers.solver_base import SolverState
from sbto.utils.plotting import plot_contact_plan, plot_costs, plot_mean_cov, plot_state_control
from sbto.utils.viewer import render_and_save_trajectory
from sbto.data.utils import solver_state_path_from_rundir, create_dirs
from sbto.data.postprocess import split_x_traj
from sbto.data.aggregate import get_top_samples
from sbto.data.filenames import (
    ALL_SAMPLES_COSTS_FILENAME,
    BEST_TRAJECTORY_FILENAME,
    BEST_SAMPLES_IT_FILENAME,
    INITIAL_SOLVER_STATE_SUFFIX,
    FINAL_SOLVER_STATE_SUFFIX,
    HYDRA_CFG,
    MJ_MODEL_NAME,
    TOP_TRAJECTORIES_FILENAME,
)

Array = npt.NDArray[np.float64]

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

def save_solver_state(
    dir_path: str,
    state: SolverState,
    suffix: str = ""
    ) -> None:
    solver_state_path = solver_state_path_from_rundir(dir_path, suffix)
    np.savez(solver_state_path, **asdict(state))

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

def save_mj_model(dir_path: str, mj_spec: mujoco.MjSpec):
    file_name = os.path.join(dir_path, f"{MJ_MODEL_NAME}.xml")
    with open(file_name, "w") as f:
        f.write(mj_spec.to_xml())

def save_results(
    data_dir: str,
    sim: SimMjRollout,
    task: OCPBase,
    solver_state_0: SolverState,
    solver_state_final: SolverState,
    all_samples: Array,
    best_samples_it: List[Array],
    all_costs: Array,
    exp_name: str = "",
    description: str = "",
    hydra_rundir: str = "",
    save_fig: bool = True,
    save_video: bool = True,
    save_samples_costs: bool = True,
    save_best_samples_it: bool = True,
    multiple_shooting: bool = False,
    split_state: bool = False,
    save_top: float = 0.,
    n_last_it: int = 0,
    remove_keys: List[str] = [],
    ) -> str:
    exp_name = task.__class__.__name__ if not exp_name else exp_name
    result_dir = create_dirs(exp_name, data_dir, description)

    # Save config
    copy_hydra_config(hydra_rundir, result_dir)

    # Save mj model
    save_mj_model(result_dir, sim.mj_scene.edit.mj_spec)

    # Save inital and final solver state
    if solver_state_0:
        save_solver_state(result_dir, solver_state_0, INITIAL_SOLVER_STATE_SUFFIX)
    save_solver_state(result_dir, solver_state_final, FINAL_SOLVER_STATE_SUFFIX)
    
    if n_last_it > 0:
        N_it_samples = n_last_it
        all_samples = all_samples[-N_it_samples:]
    else:
        N_it_samples = all_samples.shape[0]
    last_costs = all_costs[-N_it_samples:]

    # Save all samples and costs from the optimization
    if save_samples_costs:
        if n_last_it > 0:
            print(f"Saving all samples and costs from last {n_last_it} iteration.")
        else:
            print("Saving all samples and costs.")
        save_all_samples_and_cost(result_dir, all_samples, last_costs)

    # Save best samples per iterations
    if save_best_samples_it:
        data = {str(i) : sample for i, sample in enumerate(best_samples_it)}
        data["c"] = np.min(all_costs, axis=0)
        np.savez_compressed(
            os.path.join(result_dir, f"{BEST_SAMPLES_IT_FILENAME}.npz"),
            **data
        )

    # Rollout best trajectories (with initial states)
    N_top_samples = 1 # Save the best one by default
    if save_top > 0.:
        # How many top traj to save
        if save_top >= 1:
            N_samples = last_costs.size
            N_top_samples = min(int(save_top), N_samples)
        else:
            percentile = save_top
            threshold = np.percentile(last_costs, percentile)
            top_mask = last_costs <= threshold
            N_top_samples = np.sum(top_mask)
 
    top_samples, top_costs = get_top_samples(last_costs, all_samples, N_top_samples)

    if multiple_shooting:
        x_shooting = task.ref.x[sim.t_knots]
        t, x_traj, qdes_traj, obs_traj = map(np.squeeze, sim.rollout_multiple_shooting(top_samples, x_shooting, with_x0=True))
    else:
        t, x_traj, qdes_traj, obs_traj = map(np.squeeze, sim.rollout(top_samples, with_x0=True))
    
    # Sanity check
    # cost = task.cost(x_traj[None, 1:, :], top_samples, obs_traj[None, :, :])
    # print(f"[{description or 'Unnamed'}] Best cost: {np.min(cost)}")

    print(f"[{description or 'Unnamed'}] Best cost: {solver_state_final.min_cost_all}")

    # By default all data keys are saved
    data_traj = {
        "time": t,
        "x": x_traj,
        "u": qdes_traj,
        "o": obs_traj,
        "c": top_costs,
        "t_knots": sim.t_knots,
    }

    # Split state
    if split_state:
        splitted_data = split_x_traj(x_traj, mj_model=sim.mj_scene.mj_model)
        data_traj.update({
            k: v
            for k, v in splitted_data.items()
            if not k in data_traj
        })

    # Save best
    file_path = os.path.join(result_dir, f"{BEST_TRAJECTORY_FILENAME}.npz")
    if N_top_samples == 1:
        best_data = data_traj.copy()
    else:
        arg_min_cost = np.argmin(top_costs)
        best_data = {k: np.squeeze(v[arg_min_cost]) for k, v in data_traj.items()}
    np.savez_compressed(
        file_path,
        **{
            k: v for k, v in best_data.items()
            if k not in remove_keys
        }
    )
    
    # Remove keys from data
    for k in remove_keys:
        if k in data_traj.keys():
            del data_traj[k]

    # Save top trajectories
    if N_top_samples > 1:
        print(f"Saving top {N_top_samples} trajectories.")
        file_path = os.path.join(result_dir, f"{TOP_TRAJECTORIES_FILENAME}.npz")
        np.savez_compressed(
            file_path,
            **data_traj
        )
    
    # Save all figures
    if save_fig:
        best_knots = solver_state_final.best_all
        save_plots(
            result_dir,
            task,
            best_data["time"],
            best_data["x"],
            best_data["u"],
            best_data["o"],
            best_knots,
            solver_state_final.mean,
            solver_state_final.cov,
            all_costs,
        )

    # Save video rendering
    if save_video:
        render_and_save_trajectory(
            sim.mj_scene.mj_model,
            sim.mj_scene.mj_data,
            best_data["time"],
            best_data["x"],
            save_path=result_dir,
        )

    return result_dir