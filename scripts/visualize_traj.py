import argparse
import mujoco

from sbto.main import instantiate_from_cfg
from sbto.data.utils import get_config_from_rundir, load_best_trajectory_from_rundir
from sbto.utils.viewer import visualize_trajectory_with_reference, visualize_trajectory


def main(rundir: str, with_ref: bool = True):

    cfg = get_config_from_rundir(rundir)
    data = load_best_trajectory_from_rundir(rundir)

    sim, task, _, _ = instantiate_from_cfg(cfg)
    mj_model = sim.mj_scene.mj_model
    mj_data = mujoco.MjData(mj_model)

    if with_ref:
        visualize_trajectory_with_reference(
            mj_model, mj_data, task.ref.time, data["x"], task.ref.x
        )
    else:
        visualize_trajectory(
            mj_model, mj_data, data["time"], data["x"]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize best trajectory from a run directory."
    )

    parser.add_argument(
        "rundir",
        type=str,
        help="Path to run directory containing config and trajectory data.",
    )

    parser.add_argument(
        "--no-ref",
        action="store_true",
        help="Disable reference trajectory visualization.",
    )

    args = parser.parse_args()

    main(
        rundir=args.rundir,
        with_ref=not args.no_ref,
    )
