import argparse
from sbto.data.organize import (
    group_run_dir_by_ref_file_name,
    group_traj_data_by_ref_in_single_file,
)

def main(task_dir: str):
    # Group rundir by reference file names
    group_run_dir_by_ref_file_name(task_dir)
    # Aggregate all best trajectories in a single .npz file
    # per reference file name
    group_traj_data_by_ref_in_single_file(task_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group run directories and trajectory data by reference file name."
    )

    parser.add_argument(
        "task_dir",
        type=str,
        help="Path to the task dataset directory",
    )

    args = parser.parse_args()

    main(args.task_dir)
