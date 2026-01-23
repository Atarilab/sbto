import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from array import array
from tqdm import tqdm

from omegaconf import OmegaConf
from hydra.utils import instantiate

from sbto.data.utils import load_yaml, get_config_dict_from_rundir
from sbto.utils.extract_ref import ReferenceMotion
from sbto.evaluation.errors import *
from sbto.evaluation.opt_stats import *
from sbto.evaluation.diversity import *

###############################################################################
# CONFIG LOADING INTO A DATAFRAME
###############################################################################

def flatten_dict(d, parent_key="", sep=".", filter=None):
    items = {}
    for k, v in d.items():
        if filter is not None and filter(k):
            continue
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep, filter=filter))
        else:
            items[new_key] = v
    return items


def load_configs(dataset_root: str) -> dict:
    """
    RETURNS:
        { rundir_path : flattened_config_dict }
    """
    paths = glob.glob(
        f"{dataset_root}/**/config.yaml",
        recursive=True,
        include_hidden=True
    )
    # paths = filter(
    #     lambda path: "__ws_" in path,
    #     paths
    # )

    filter_out = lambda k_cfg: k_cfg.startswith("_") and k_cfg.endswith("_")

    configs = {}
    for p in paths:
        rundir = os.path.dirname(os.path.dirname(p))
        configs[rundir] = flatten_dict(load_yaml(p), filter=filter_out)

    return configs

def load_opt_stats_from_rundir(rundir: str, N_samples) -> dict:
    """
    RETURNS:
        { rundir_path : opt_stats }
    """
    opt_stats_path = os.path.join(rundir, "optimization_stats.yaml")
    if os.path.exists(opt_stats_path):
        opt_stats = load_yaml(opt_stats_path)
        data = {
            "opt_n_it": opt_stats["n_it"],
            "opt_duration": opt_stats["duration"],
            "total_sim_timesteps": total_sim_timesteps(opt_stats) * N_samples
        }
    else:
        data = {}
    return data

###############################################################################
# MEMORY-OPTIMIZED SBTO ERROR COMPUTATION
###############################################################################

def instantiate_ref_from_cfg(cfg, dt : float = 0.):
    mj_model = None
    if dt == 0.:
        mj_scene_ref = instantiate(cfg.task.mj_scene_ref)
        mj_model = mj_scene_ref.mj_model

    cfg_ref = instantiate(cfg.task.cfg_ref)

    try:
        ref = ReferenceMotion(
            cfg_ref.motion_path,
            mj_model,
            cfg_ref.t0,
            cfg_ref.t_end,
            cfg_ref.speedup,
            cfg_ref.z_offset,
            dt=dt,
        )
    except:
        ref = ReferenceMotion(
            cfg_ref.motion_path,
            mj_model,
            cfg_ref.t0,
            cfg_ref.speedup,
            cfg_ref.z_offset,
            dt=dt,
        )

    # free large mujoco model immediately
    if mj_model is not None:
        del mj_scene_ref

    return ref

def split_traj(qpos, qvel):
    id_splits_qpos = [
        7,
        29,
    ]
    id_splits_qvel = [
        6,
        29,
    ]
    data = {}
    (
        data["base_xyz_quat"],
        data["actuator_pos"],
        data["obj_0_xyz_quat"],
    ) = np.split(qpos, np.cumsum(id_splits_qpos), axis=-1)
    (
        data["base_linvel_angvel"],
        data["actuator_vel"],
        data["obj_0_linvel_angvel"],
    ) = np.split(qvel, np.cumsum(id_splits_qvel), axis=-1)
    return data

def compute_stats_rundir(rundir: str):

    cfg = OmegaConf.create(get_config_dict_from_rundir(rundir))
    
    traj_path = os.path.join(rundir, "best_trajectory.npz")
    data = dict(np.load(traj_path, mmap_mode="r"))

    
    # traj_path = os.path.join(rundir, "time_x_u_traj.npz")
    # data = dict(np.load(traj_path, mmap_mode="r"))
    # qpos, qvel = np.split(data["x"], [43], axis=-1)
    # data.update(split_traj(qpos, qvel))

    solver_state_path = os.path.join(rundir, "solver_state_final.npz")
    final_state = np.load(solver_state_path)
    min_cost = float(final_state["min_cost_all"])

    time = data["time"]
    dt = np.mean(np.diff(time))
    ref = instantiate_ref_from_cfg(cfg, dt)
    ref_filename = os.path.split(cfg.task.cfg_ref.motion_path)[-1]

    obj_all = data["obj_0_xyz_quat"]
    base_all = data["base_xyz_quat"]
    joints = data["actuator_pos"]

    obj_pos = obj_all[:, :3]
    obj_quat = obj_all[:, -4:]
    base_pos = base_all[:, :3]
    base_quat = base_all[:, -4:]

    act_acc, act_acc_ref = compute_total_act_acc(data["actuator_vel"], ref.dof_v, dt)
    act_acc_ratio = act_acc / act_acc_ref

    stats = {
        "err_pos_obj": float(compute_obj_pos_error(obj_pos, ref.object_root_pos)),
        "err_term_pos_obj": float(compute_term_obj_pos_error(obj_pos, ref.object_root_pos)),
        "err_quat_obj": float(compute_obj_quat_error(obj_quat, ref.object_rot)),
        "err_term_quat_obj": float(compute_term_obj_quat_error(obj_quat, ref.object_rot)),
        "err_pos_base": float(compute_base_pos_error(base_pos, ref.root_pos)),
        "err_term_pos_base": float(compute_term_base_pos_error(base_pos, ref.root_pos)),
        "err_quat_base": float(compute_base_quat_error(base_quat, ref.root_rot)),
        "err_term_quat_base": float(compute_term_base_quat_error(base_quat, ref.root_rot)),
        "err_joint": float(compute_joint_pos_error(joints, ref.dof_pos)),
        "act_acc": float(act_acc),
        "act_acc_ref": float(act_acc_ref),
        "act_acc_ratio": float(act_acc_ratio),
        "T": len(time),
        "min_cost": min_cost,
        "ref_filename": ref_filename,
    }

    opt_stats = load_opt_stats_from_rundir(rundir, cfg.solver.cfg.N_samples)
    stats.update(opt_stats)

    del cfg, ref, data
    return stats


def get_all_rundirs(dataset_root):

    paths = glob.glob(
        f"{dataset_root}/**/*__ws_incr*/",
        recursive=True,
        include_hidden=True
    )
    return paths


def _worker_compute_errors(rundir):
    try:
        return rundir, compute_stats_rundir(rundir)
    except Exception as ex:
        return rundir, ex


def compute_all_errors_parallel(rundirs, num_workers=None):

    # rundirs = list(get_all_rundirs(dataset_root))
    # print(f"Found {len(rundirs)} rundirs.")

    num_workers = num_workers or mp.cpu_count()

    errors_by_rundir = {}

    with mp.Pool(num_workers, maxtasksperchild=50) as pool:
        for rundir, result in tqdm(pool.imap_unordered(_worker_compute_errors, rundirs), total=len(rundirs)):
            if isinstance(result, Exception):
                print(f"[WARN] Failed {rundir}: {result}")
                continue
            errors_by_rundir[rundir] = result
    return errors_by_rundir


###############################################################################
# MERGE CONFIGS + ERRORS INTO ONE DATAFRAME
###############################################################################

def load_dataset_with_errors(dataset_root: str, num_workers=None) -> pd.DataFrame:
    """
    OUTPUT: DataFrame with both config.yaml fields and error metrics.
    Ensures perfect matching by rundir path.
    """
    print("Loading configs...")
    configs = load_configs(dataset_root)

    print("Computing errors (parallel)...")
    errors = compute_all_errors_parallel(configs.keys(), num_workers=num_workers)

    print(f"Merging configs ({len(configs)}) and errors ({len(errors)})...")

    # Align config + error dict using rundir key
    rows = []
    for rundir, cfg_row in configs.items():
        if rundir in errors:
            merged = dict(cfg_row)
            merged.update(errors[rundir])
            merged["rundir"] = rundir
            rows.append(merged)

    df = pd.DataFrame(rows)
    df.insert(0, "algo", "SBTO")
    print(f"Final dataset size: {df.shape}")
    return df

def load_data(path):
    return path, dict(np.load(path, mmap_mode="r+"))

def load_all_trajectories_dataset(dataset_root: str, only_best_traj: bool = False, num_workers = 20) -> pd.DataFrame:
    FILENAME = "best_trajectory.npz" if only_best_traj else "top_trajectories.npz"
    all_traj_paths = glob.glob(
        f"{dataset_root}/**/{FILENAME}",
        recursive=True,
    )
    # Compute stats for all traj
    all_traj = {}

    with mp.Pool(num_workers, maxtasksperchild=50) as pool:
        for path, data in tqdm(pool.imap_unordered(load_data, all_traj_paths), total=len(all_traj_paths)):
            if isinstance(data, Exception):
                print(f"[WARN] Failed {path}: {data}")
                continue
            all_traj[path] = data
    
    return all_traj

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    DATASET_ROOT = "datasets/SBTO_OmniRetarget_Dataset"
    DATASET_ROOT = "datasets/OmniRetarget/"
    DATASET_ROOT = "datasets/G1RobotObjRef20KnotsMPCCost"

    df = load_dataset_with_errors(DATASET_ROOT, num_workers=60)
    print(df.head())


