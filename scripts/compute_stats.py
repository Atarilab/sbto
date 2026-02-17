import argparse
import shutil
from sbto.evaluation.load import load_dataset_with_errors as load

def main(
    task_dir: str,
    delete_failures: bool = False,
    ):
    data = load(task_dir)

    if delete_failures:
        failure_rundirs = data[~data["success"]]["rundir"]
        for rundir in failure_rundirs:
            shutil.rmtree(rundir)
        print(len(failure_rundirs), "failure runs deleted.")

    print("=== Success Counts ===")
    n_success = data["success"].sum()
    print(f"{n_success / len(data) * 100.}%")

    print("=== Smoothness ===")
    smoothness = data[data["success"]]["act_acc_ratio"].values.mean()
    print(f" {smoothness}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SBTO dataset statistics."
    )

    parser.add_argument(
        "task_dir",
        type=str,
        help="Path to the dataset root directory",
    )

    parser.add_argument(
        "--delete-failures",
        action="store_true",
        help="Delete failed run directories",
    )

    args = parser.parse_args()

    main(
        task_dir=args.task_dir,
        delete_failures=args.delete_failures,
    )