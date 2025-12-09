import os
import numpy as np
import mujoco


def build_state_traj_from_npz(npz_path, mj_model):
    """
    Reconstruct MuJoCo state [qpos | qvel] for ALL trajectories from the RL-format NPZ

    Returns
        state_traj_all: (N, T, nq + nv) = (N, T, 84) = actor input
    """
    data = np.load(npz_path)

    # (N, T, *)
    act_pos            = data["actuator_pos"].astype(np.float32)        # (N, T, 29)
    act_vel            = data["actuator_vel"].astype(np.float32)        # (N, T, 29)
    base_xyz_quat      = data["base_xyz_quat"].astype(np.float32)       # (N, T, 7)
    base_linvel_angvel = data["base_linvel_angvel"].astype(np.float32)  # (N, T, 6)
    obj_xyz_quat       = data["obj_0_xyz_quat"].astype(np.float32)      # (N, T, 7)
    obj_linvel_angvel  = data["obj_0_linvel_angvel"].astype(np.float32) # (N, T, 6)

    N, T, A = act_pos.shape  # N = 1024, T = 201, A = 29

    # DOFs from MuJoCo
    nq = mj_model.nq   # 43
    nv = mj_model.nv   # 41

 
    expected_nq = 7 + A + 7
    expected_nv = 6 + A + 6

    if nq != expected_nq:
        raise ValueError(f"Expected nq = 7 + {A} + 7 = {expected_nq}, got {nq}")
    if nv != expected_nv:
        raise ValueError(f"Expected nv = 6 + {A} + 6 = {expected_nv}, got {nv}")

    # Build qpos and qvel for ALL trajectories
    qpos = np.concatenate(
        [base_xyz_quat, act_pos, obj_xyz_quat],
        axis=-1
    )  # (N, T, nq = 43)

    qvel = np.concatenate(
        [base_linvel_angvel, act_vel, obj_linvel_angvel],
        axis=-1
    )  # (N, T, nv = 41)

    # State trajectory: [qpos, qvel]
    state_traj_all = np.concatenate(
        [qpos, qvel],
        axis=-1
    )  # (N, T, nq + nv = 84)

    return state_traj_all


def extract_anchor_from_traj(xml_path, state_traj_single):
    """
    Compute anchor_pos_b and anchor_ori_b from ONE trajectory using MuJoCo sensors.

    Uses the sensors defined in g1_mjx.xml / scene_29dof.xml:

        <framepos  name="torso_position"    ... />
        <framequat name="torso_orientation" ... />

    Args:
        xml_path          (str): path to scene_29dof.xml
        state_traj_single (np.ndarray): (T, nq + nv) [qpos | qvel] for one trajectory

    Returns:
        anchor_pos_b: (T, 3)  world position of the torso site
        anchor_ori_b: (T, 4)  world orientation of the torso site as quaternion
    """
    # load model and data
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(mj_model)

    # relevant sensors only
    sensor_names = ["torso_position", "torso_orientation"]

    #sensor addresses and dims
    sensor_info = []  
    for name in sensor_names:
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = mj_model.sensor_adr[sid]
        dim = mj_model.sensor_dim[sid]
        sensor_info.append((name, adr, dim))

    T  = state_traj_single.shape[0]
    nq = mj_model.nq

    # Split state_traj into qpos and qvel
    qpos_traj = np.copy(state_traj_single[:, :nq])
    qvel_traj = np.copy(state_traj_single[:, nq:])

    anchor_pos_b = np.zeros((T, 3), dtype=np.float32)
    anchor_ori_b = np.zeros((T, 4), dtype=np.float32)

    for t in range(T):
        data.qpos[:] = qpos_traj[t]
        data.qvel[:] = qvel_traj[t]

        mujoco.mj_forward(mj_model, data)

        # Read  relevant sensors only 
        for name, adr, dim in sensor_info:
            values = np.copy(data.sensordata[adr:adr + dim])
            if name == "torso_position":
                anchor_pos_b[t] = values  # (3,)
            elif name == "torso_orientation":
                anchor_ori_b[t] = values  # (4,)

    return anchor_pos_b, anchor_ori_b


def main():
    xml_path = "sbto/models/unitree_g1/scene_29dof.xml"
    npz_path = "sbto/data/rollout_time_x_u_obs_traj_scene_29dof_rl_format.npz"

    # Load model once 
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

   
    state_traj_all = build_state_traj_from_npz(npz_path, mj_model)
    N, T, D = state_traj_all.shape
    print(f"state_traj_all shape = {state_traj_all.shape}  (N={N}, T={T}, D={D})")

    
    anchor_pos_all = np.zeros((N, T, 3), dtype=np.float32)
    anchor_ori_all = np.zeros((N, T, 4), dtype=np.float32)

    # Loop over all trajectories
    for i in range(N):
        print(f"Processing trajectory {i+1}/{N}...")
        state_traj_single = state_traj_all[i]  # (T, nq+nv)
        anchor_pos_b, anchor_ori_b = extract_anchor_from_traj(xml_path, state_traj_single)

        anchor_pos_all[i] = anchor_pos_b
        anchor_ori_all[i] = anchor_ori_b

    
    old = dict(np.load(npz_path))

    old["anchor_pos_b"] = anchor_pos_all.astype(np.float32)  # (N, T, 3)
    old["anchor_ori_b"] = anchor_ori_all.astype(np.float32)  # (N, T, 4)

    np.savez(npz_path, **old)
    print(f"Updated NPZ with anchor_pos_b and anchor_ori_b: {npz_path}")
    print(f"anchor_pos_b shape: {anchor_pos_all.shape}")
    print(f"anchor_ori_b shape: {anchor_ori_all.shape}")


if __name__ == "__main__":
    main()