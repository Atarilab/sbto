import argparse
import re
import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME #nominal pose from mjlab


def build_u_nominal_from_keyframe(mj_model: mujoco.MjModel) -> np.ndarray:
    """
    Build a nominal PD target vector (one value per actuator) from
    KNEES_BENT_KEYFRAME.
    """
    # Collect all joint names in the model
    joint_names = []
    for j in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        joint_names.append(name)

    # Initialize all joints with 0
    nominal_by_name: dict[str, float] = {}
    for name in joint_names:
        if name is not None:
            nominal_by_name[name] = 0.0

    # fill nominal angles with keyframe patterns
    for pattern, val in KNEES_BENT_KEYFRAME.joint_pos.items():
        regex = re.compile(pattern)
        for name in joint_names:
            if name is None:
                continue
            if regex.fullmatch(name):
                nominal_by_name[name] = float(val)

    # nominal angle for the joint controlled by actuator a
    nu = mj_model.nu
    u_nominal = np.zeros(nu, dtype=np.float32)

    for a in range(nu):
        #  joint id the actuator is attached to
        joint_id = int(mj_model.actuator_trnid[a, 0])
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            u_nominal[a] = 0.0
        else:
            u_nominal[a] = nominal_by_name.get(joint_name, 0.0)

    return u_nominal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz",
        type=str,
        default="sbto/data/pretraining_actor_input.npz",
        help="Path to RL-format NPZ with keys x,u,o,c,...",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default="sbto/models/unitree_g1/scene_29dof.xml",
        help="Path to the MuJoCo scene XML used for 29-DoF G1.",
    )
    args = parser.parse_args()

    print(f"Loading MuJoCo model from: {args.xml}")
    mj_model = mujoco.MjModel.from_xml_path(args.xml)

    print("Building nominal PD targets from KNEES_BENT_KEYFRAME...")
    u_nominal = build_u_nominal_from_keyframe(mj_model)
    

    print(f"\nLoading NPZ: {args.npz}")
    data = np.load(args.npz)
    keys = list(data.files)


    if "u" not in keys:
        raise KeyError("NPZ does not contain key 'u' (control trajectories).")

    u = data["u"].astype(np.float32)  # (N, T, nu)
    N, T, U = u.shape
    print(f"u shape = {u.shape}  (N={N}, T={T}, U={U})")

    if U != mj_model.nu:
        raise ValueError(
            f"Dimension mismatch: u has size U={U}, but mj_model.nu={mj_model.nu}."
        )

    # Broadcast u_nominal to (N, T, U) and compute policy output
    u_nominal_broadcast = u_nominal.reshape(1, 1, U)
    u_policy = u - u_nominal_broadcast

    print("\nSanity check on u_policy:")
    print("  u       mean/std:", float(u.mean()), float(u.std()))
    print("  u_nom   min/max :", float(u_nominal.min()), float(u_nominal.max()))
    print("  u_policy mean/std:", float(u_policy.mean()), float(u_policy.std()))

    # Save back into NPZ with u_policy'
    out_dict = dict(data)  # copy all original arrays
    out_dict["u_policy"] = u_policy.astype(np.float32)

    np.savez(args.npz, **out_dict)
    print(f"\nSaved updated NPZ with 'u_policy' to: {args.npz}")
    print("Done.")


if __name__ == "__main__":
    main()