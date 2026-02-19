import os
import glob
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import time

from omegaconf import OmegaConf
from hydra.utils import instantiate

from sbto.data.utils import load_yaml, get_config_dict_from_rundir
from sbto.utils.extract_ref import ReferenceMotion
from sbto.evaluation.errors import *
from sbto.evaluation.opt_stats import *
from sbto.evaluation.diversity import *
from sbto.data.constants import *
from sbto.evaluation.success_rate import compute_success

CONFIG_FILENAME = "config.yaml"
OPT_STATS_FILENAME = "optimization_stats.yaml"
TRAJ_FILENAME = "best_trajectory.npz"

def _worker_load_config_dict(rundir):
    try:
        cfg_dict = get_config_dict_from_rundir(rundir)
        cfg = OmegaConf.create(cfg_dict)
        return rundir, cfg
    except Exception as ex:
        return rundir, ex

def load_configs_parallel(dataset_root: str, num_workers: int) -> dict:
    """
    RETURNS:
        { rundir_path : flattened_config_dict }
    """
    all_cfg_paths = glob.glob(
        f"{dataset_root}/**/{CONFIG_FILENAME}",
        recursive=True,
        include_hidden=True
    )
    rundirs = [
        os.path.dirname(os.path.dirname(p))
        for p in all_cfg_paths
    ]
    cfg_dicts = []

    with mp.Pool(num_workers, maxtasksperchild=50) as pool:
        for rundir, cfg_dict in tqdm(pool.imap(_worker_load_config_dict, rundirs), total=len(rundirs)):
            if isinstance(cfg_dict, Exception):
                print(f"[WARN] Failed {rundir}: {cfg_dict}")
                continue
            cfg_dicts.append(cfg_dict)

    return rundirs, cfg_dicts

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

def load_opt_stats_from_rundir(rundir: str, N_samples) -> dict:
    """
    RETURNS:
        { rundir_path : opt_stats }
    """
    opt_stats_path = os.path.join(rundir, OPT_STATS_FILENAME)
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

def get_ref_data(ref: ReferenceMotion):
    return {
        KEY_OBJECT_POS: ref.object_pos,
        KEY_ROOT_POS: ref.root_pos,
        KEY_OBJECT_ROT: ref.object_rot,
        KEY_ROOT_ROT: ref.root_rot,
        KEY_DOF_POS: ref.dof_pos,
        "dt": ref.dt,
    }

def get_ref_data_from_config(cfg, mj_scene_ref = None):
    _delete = False
    if mj_scene_ref is None:
        _delete = True
        mj_scene_ref = instantiate(cfg.task.mj_scene_ref)

    ref = ReferenceMotion(
        mj_scene_ref,
        cfg.task.cfg_ref.motion_path,
        cfg.task.cfg_ref.t0,
        cfg.task.cfg_ref.t_end,
        cfg.task.cfg_ref.speedup,
        cfg.task.cfg_ref.z_offset,
        cfg.task.cfg_ref.flip_quat_pos,
        cfg.task.cfg_ref.quat_wxyz,
    )

    if _delete:
        del mj_scene_ref

    return get_ref_data(ref)

def load_data_from_rundir(rundir: str):
    traj_path = os.path.join(rundir, TRAJ_FILENAME)
    data = np.load(traj_path, mmap_mode="r")
    return data

def compute_errors(data, ref_data):
    joints = data[KEY_DOF_POS]
    obj_pos = data[KEY_OBJECT_POS]
    obj_quat = data[KEY_OBJECT_ROT]
    base_pos = data[KEY_ROOT_POS]
    base_quat = data[KEY_ROOT_ROT]

    return {
        "err_pos_obj": float(compute_obj_pos_error(obj_pos, ref_data[KEY_OBJECT_POS])),
        "err_term_pos_obj": float(compute_term_obj_pos_error(obj_pos, ref_data[KEY_OBJECT_POS])),
        "err_quat_obj": float(compute_obj_quat_error(obj_quat, ref_data[KEY_OBJECT_ROT])),
        "err_term_quat_obj": float(compute_term_obj_quat_error(obj_quat, ref_data[KEY_OBJECT_ROT])),
        "err_pos_base": float(compute_base_pos_error(base_pos, ref_data[KEY_ROOT_POS])),
        "err_term_pos_base": float(compute_term_base_pos_error(base_pos, ref_data[KEY_ROOT_POS])),
        "err_quat_base": float(compute_base_quat_error(base_quat, ref_data[KEY_ROOT_ROT])),
        "err_term_quat_base": float(compute_term_base_quat_error(base_quat, ref_data[KEY_ROOT_ROT])),
        "err_joint": float(compute_joint_pos_error(joints, ref_data[KEY_DOF_POS])),
    }

def compute_smoothness(data, ref_data):
    act_acc, act_acc_ref = compute_total_act_acc(
        data[KEY_DOF_POS],
        ref_data[KEY_DOF_POS],
        ref_data["dt"]
        )
    act_acc_ratio = act_acc / act_acc_ref

    return {
        "act_acc": float(act_acc),
        "act_acc_ref": float(act_acc_ref),
        "act_acc_ratio": float(act_acc_ratio),
    }

def compute_opt_stats(rundir, cfg):
    opt_stats = load_opt_stats_from_rundir(rundir, cfg.solver.cfg.N_samples)
    return opt_stats

def compute_all_stats(rundir, cfg, ref_data):

    data = load_data_from_rundir(rundir)
    if ref_data is None:
        ref_data = get_ref_data_from_config(cfg)
    
    motion_path = cfg.task.cfg_ref.motion_path
    ref_filename = os.path.split(motion_path)[-1]
    stats = {
        "ref_filename": ref_filename,
        "motion_path": motion_path,
        "rundir": rundir,
        "T": len(data[KEY_TIME]),
        "min_cost": data[KEY_COST],
        "dt": ref_data["dt"]
    }

    stats.update(compute_errors(data, ref_data))
    stats.update(compute_smoothness(data, ref_data))
    stats.update(compute_opt_stats(rundir, cfg))

    filter_out = lambda k_cfg: k_cfg.startswith("_") and k_cfg.endswith("_")
    cfg_dict_flat = flatten_dict(OmegaConf.to_container(cfg), filter=filter_out)
    stats.update(cfg_dict_flat)

    return stats

def _worker_compute_stats(args):
    rundir = args[0]
    try:
        return rundir, compute_all_stats(*args)
    except Exception as ex:
        return rundir, ex

def compute_all_stats_parallel(
    rundirs,
    cfg_dicts,
    num_workers: int,
    ref_datas = None,
    ):
    all_stats = []
    N = len(rundirs)
    ref_datas = [ref_datas] * N if ref_datas is None else ref_datas
    args = zip(rundirs, cfg_dicts, ref_datas)
    with mp.Pool(num_workers, maxtasksperchild=50) as pool:
        for rundir, result in tqdm(pool.imap(_worker_compute_stats, args), total=len(rundirs)):
            if isinstance(result, Exception):
                print(f"[WARN] Failed {rundir}: {result}")
                continue
            all_stats.append(result)
        
    return all_stats

def is_mj_scene_ref_identical(cfg_dicts):
    _mj_scene_ref_dict = {}
    
    for cfg_dict in cfg_dicts:
        mj_scene_ref_dict = cfg_dict["task"]["mj_scene_ref"]
        if len(_mj_scene_ref_dict) > 0 and _mj_scene_ref_dict != mj_scene_ref_dict:
            return False
        _mj_scene_ref_dict = mj_scene_ref_dict
    
    return True

def get_mj_scene_ref_from_cfg(cfg):
    mj_scene_ref = instantiate(cfg.task.mj_scene_ref)
    return mj_scene_ref

def compute_all_ref_data(
    cfgs,
    mj_scene_ref,
    ):
    all_ref_data = []
    for cfg in tqdm(cfgs):
        all_ref_data.append(
            get_ref_data_from_config(cfg, mj_scene_ref)
        )
    return all_ref_data

def load_dataset_with_errors(dataset_root: str, num_workers=None) -> pd.DataFrame:
    """
    OUTPUT: DataFrame with both config.yaml fields and error metrics.
    Ensures perfect matching by rundir path.
    """
    num_workers = num_workers or mp.cpu_count()

    print("Loading configs (parallel)...")
    rundirs, cfgs = load_configs_parallel(dataset_root, num_workers)

    # Load mj_scene_ref once if identical across all configs (speedup)
    if is_mj_scene_ref_identical(cfgs):
        print("Preloading the reference data...")
        mj_scene_ref = get_mj_scene_ref_from_cfg(cfgs[0])
        ref_data = compute_all_ref_data(cfgs, mj_scene_ref)
    else:
        ref_data = None
    
    print("Computing stats (parallel)...")
    all_stats = compute_all_stats_parallel(rundirs, cfgs, num_workers, ref_data)

    df = pd.DataFrame(all_stats)
    print(df.head())

    df.insert(0, "algo", "SBTO")
    df["success"] = compute_success(df)

    print(f"Final dataset size: {df.shape}")
    return df


