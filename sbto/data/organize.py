import numpy as np
from collections import defaultdict
import os
import shutil

from sbto.data.load import (
    get_best_trajectory_from_rundir,
    )
from sbto.data.utils import (
    get_config_dict_from_rundir,
    get_filename_from_path,
    get_arg_from_cfg_dict,
    get_all_best_traj_data,
    )
from sbto.data.filenames import (
    BEST_TRAJECTORY_RAND_FILENAME,
    )

def group_run_dir_by_ref_file_name(task_dir: str):
    """
    Group run directories by reference file names.
    To be used when generating from a lot of reference files. 
    """
    for dir in os.listdir(task_dir):
        run_dir = os.path.join(task_dir, dir)
        if os.path.isdir(run_dir):
            cfg_dict = get_config_dict_from_rundir(run_dir)
            # Get all ref file paths
            try:
                ref_motion_path = get_arg_from_cfg_dict(cfg_dict, "motion_path")
            except Exception as _:
                continue
            
            # Create new dir with same name as motion ref
            # Move the rundir data there
            ref_motion_name = get_filename_from_path(ref_motion_path)
            run_dir_dst = os.path.join(task_dir, ref_motion_name)
            os.makedirs(run_dir_dst, exist_ok=True)
            shutil.move(run_dir, run_dir_dst)

def group_traj_data_by_ref_in_single_file(task_dir: str):
    run_dir_by_ref = defaultdict(list)
    all_traj_data_paths = get_all_best_traj_data(task_dir)

    for path in all_traj_data_paths:
        run_dir = os.path.split(path)[0]
        if os.path.isdir(run_dir):
            cfg_dict = get_config_dict_from_rundir(run_dir)
            # Get all ref file paths
            try:
                ref_motion_path = get_arg_from_cfg_dict(cfg_dict, "motion_path")
            except Exception as _:
                continue

            ref_motion_name = get_filename_from_path(ref_motion_path)
            run_dir_by_ref[ref_motion_name].append(run_dir)

    for ref_motion_name, rundir in run_dir_by_ref.items():
        # Continue if just one run
        if len(rundir) <= 1:
            continue

        all_data = defaultdict(list)
        for i, path in enumerate(rundir):
            data = get_best_trajectory_from_rundir(rundir)
            for k, v in data.items():
                all_data[k].append(np.squeeze(v))

        run_dir_dst = os.path.join(task_dir, ref_motion_name)
        os.makedirs(run_dir_dst, exist_ok=True)
        rand_data_path = os.path.join(run_dir_dst, f"{BEST_TRAJECTORY_RAND_FILENAME}.npz")
        
        print(ref_motion_name, i+1)
        del all_data["t_knots"]
        np.savez_compressed(rand_data_path, **all_data)

