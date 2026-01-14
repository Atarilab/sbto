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


REF_DATASET_PATH = "./datasets/robot-object"

def instantiate_ref_from_path(ref_motion_path: str, dt):
    mj_model = None
    t0 = 0.
    speedup = 1.
    z_offset = 0.

    ref = ReferenceMotion(
        ref_motion_path,
        mj_model,
        t0,
        speedup,
        z_offset,
        dt=dt,
    )

    # free large mujoco model immediately
    if mj_model is not None:
        del mj_scene_ref

    return ref

def aggregate_data(obj):
    t = []
    qpos = []
    qvel = []
    for d in obj:
        t.append(d["time"]) 
        qpos.append(d["qpos"])
        qvel.append(d["qvel"])

    return np.asarray(t), np.asarray(qpos), np.asarray(qvel)

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
    ) = np.split(qvel, np.cumsum(id_splits_qvel, axis=-1))
    return data


def compute_stats_traj(traj_path: str):
    data = np.load(traj_path, mmap_mode="r")
    qvel = data["qvel"].reshape(-1,  data["qvel"].shape[-1])
    data_processed = split_traj(data["qpos"], qvel)
    
    ref_motion_filename = [s for s in traj_path.split("/") if "_original" in s][0]
    ref_motion_filename += ".npz"

    ref_motion_path = os.path.join(REF_DATASET_PATH, ref_motion_filename)

    fps = data["fps"]
    dt = 1. / fps
    

    ref = instantiate_ref_from_path(ref_motion_path, dt)   

    obj_all = data_processed["obj_0_xyz_quat"]
    base_all = data_processed["base_xyz_quat"]
    joints = data_processed["actuator_pos"]

    obj_pos = obj_all[:, :3]
    obj_quat = obj_all[:, -4:]
    base_pos = base_all[:, :3]
    base_quat = base_all[:, -4:]

    N = len(obj_pos)
    time = np.linspace(0, N*dt, N, endpoint=False)

    act_acc, act_acc_ref = compute_total_act_acc(data_processed["actuator_vel"], ref.dof_v, dt)
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
        "task.cfg_ref.motion_path": ref_motion_path,
    }

    del ref, data, data_processed
    return stats


def get_all_rundirs(dataset_root):
    for root, dirs, files in os.walk(dataset_root):
        if "best_trajectory.npz" in files:
            yield root


def _worker_compute_errors(rundir):
    try:
        return rundir, compute_stats_traj(rundir)
    except Exception as ex:
        return rundir, ex


def compute_all_errors_parallel(dataset_root, num_workers=None):

    all_traj_paths = glob.glob(
        f"{dataset_root}/**/trajectory_comparison.npz",
        recursive=True,
        include_hidden=True
    )
    print(f"Found {len(all_traj_paths)} rundirs.")

    num_workers = num_workers or mp.cpu_count()

    errors = []

    with mp.Pool(num_workers, maxtasksperchild=50) as pool:
        for rundir, result in tqdm(pool.imap_unordered(_worker_compute_errors, all_traj_paths), total=len(all_traj_paths)):
            if isinstance(result, Exception):
                print(f"[WARN] Failed {rundir}: {result}")
                continue
            errors.append(result)

    return errors


###############################################################################
# MERGE CONFIGS + ERRORS INTO ONE DATAFRAME
###############################################################################

def load_dataset_with_errors(dataset_root: str, num_workers=None) -> pd.DataFrame:
    """
    OUTPUT: DataFrame with both config.yaml fields and error metrics.
    Ensures perfect matching by rundir path.
    """
    print("Computing errors (parallel)...")
    errors = compute_all_errors_parallel(dataset_root, num_workers=num_workers)

    df = pd.DataFrame(errors)
    df.insert(0, "algo", "SPIDER")
    print(f"Final dataset size: {df.shape}")
    return df


###############################################################################
# MAIN
###############################################################################

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    DATASET_ROOT = "datasets/SPIDER_OmniRetarget_Dataset"

    df = load_dataset_with_errors(DATASET_ROOT, num_workers=20)
    print(df.head())