import os
import numpy as np
import mujoco


def add_rl_obs_from_x(npz_path: str, xml_path: str) -> None:
    
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)

    
    data = np.load(npz_path)
    x = data["x"].astype(np.float32)          # (N, T, 84)

    N, T, NX = x.shape
    nq, nv = mj_model.nq, mj_model.nv

    if NX != nq + nv:
        raise ValueError(f"x last dim = {NX}, but model has nq+nv = {nq+nv}")

    # Flatten over (N, T)
    x_flat = x.reshape(-1, NX)               # (N*T, nq+nv)
    qpos_flat = x_flat[:, :nq]               # (N*T, nq)
    qvel_flat = x_flat[:, nq:]               # (N*T, nv)

    # nq = 7 (base) + A (joints) + 7 (obj)
    A = nq - 14 #A=29
    if A <= 0:
        raise ValueError(f"Cannot infer A from nq={nq} (expected nq = 7 + A + 7).")

    total = N * T

    # preallocation
    anchor_pos_b_flat = np.zeros((total, 3), dtype=np.float32)
    anchor_ori_b_flat = np.zeros((total, 4), dtype=np.float32)

    # Base lin/ang vel
    base_lin_vel_flat = qvel_flat[:, 0:3].copy()        # (N*T, 3)
    base_ang_vel_flat = qvel_flat[:, 3:6].copy()        # (N*T, 3)

    # Joint pos/vel 
    joint_pos_flat = qpos_flat[:, 7:7 + A].copy()       # (N*T, A)
    joint_vel_flat = qvel_flat[:, 6:6 + A].copy()       # (N*T, A)

    #  sensor indices for torso_position / torso_orientatio
    sid_pos = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position")
    sid_ori = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_orientation")

    adr_pos, dim_pos = mj_model.sensor_adr[sid_pos], mj_model.sensor_dim[sid_pos]
    adr_ori, dim_ori = mj_model.sensor_adr[sid_ori], mj_model.sensor_dim[sid_ori]

    if dim_pos != 3:
        raise ValueError(f"torso_position dim = {dim_pos}, expected 3")
    if dim_ori != 4:
        raise ValueError(f"torso_orientation dim = {dim_ori}, expected 4")

    
    for i in range(total):
        mj_data.qpos[:] = qpos_flat[i]
        mj_data.qvel[:] = qvel_flat[i]

        mujoco.mj_forward(mj_model, mj_data)

        anchor_pos_b_flat[i] = mj_data.sensordata[adr_pos:adr_pos + dim_pos]
        anchor_ori_b_flat[i] = mj_data.sensordata[adr_ori:adr_ori + dim_ori]

    # Reshape back to (N, T, ·)
    anchor_pos_b = anchor_pos_b_flat.reshape(N, T, 3)
    anchor_ori_b = anchor_ori_b_flat.reshape(N, T, 4)
    base_lin_vel = base_lin_vel_flat.reshape(N, T, 3)
    base_ang_vel = base_ang_vel_flat.reshape(N, T, 3)
    joint_pos    = joint_pos_flat.reshape(N, T, A)
    joint_vel    = joint_vel_flat.reshape(N, T, A)

    # Saving
    out = dict(data)
    out["anchor_pos_b"] = anchor_pos_b
    out["anchor_ori_b"] = anchor_ori_b
    out["base_linvel_angvel"] = np.concatenate([base_lin_vel, base_ang_vel], axis=-1)
    out["actuator_pos"]    = joint_pos
    out["actuator_vel"]    = joint_vel

    np.savez(npz_path, **out)
    print(f"Updated npz with RL obs: {npz_path}")
    print(f"  anchor_pos_b: {anchor_pos_b.shape}")
    print(f"  anchor_ori_b: {anchor_ori_b.shape}")
    print(f"  base_linvel_angvel: {(N,T,6)}")
    print(f"  actuator_pos:    {joint_pos.shape}")
    print(f"  actuator_vel:    {joint_vel.shape}")


def main():
    npz_path = "sbto/data/rollout_time_x_u_obs_traj.npz"
    xml_path = "sbto/models/unitree_g1/scene_29dof.xml"
    add_rl_obs_from_x(npz_path, xml_path)


if __name__ == "__main__":
    main()