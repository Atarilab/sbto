
"""
reconstructs all RL-style observation pieces that MJLab expects
from x state trajectories,by running the MuJoCo model forward once 
per state and writing the results back into the same NPZ file
"""
import os
import numpy as np
import mujoco


def add_rl_obs_from_x(npz_path: str, xml_path: str) -> None:
    
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data  = mujoco.MjData(mj_model)  
    body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
   )
    num_bodies = len(body_names)
    body_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in body_names
    ]
    
    # sanity check
    if any(bid < 0 for bid in body_ids):
        raise ValueError(f"Some body names not found in model: {body_ids}")
    
    data = np.load(npz_path)
    x = data["x"].astype(np.float32)          

    N, T, NX = x.shape
    nq, nv = mj_model.nq, mj_model.nv

    if NX != nq + nv:
        raise ValueError(f"x last dim = {NX}, but model has nq+nv = {nq+nv}")

    # Flatten over (N, T) to loop easily
    x_flat = x.reshape(-1, NX)               # (N*T, nq+nv)
    qpos_flat = x_flat[:, :nq]               # (N*T, nq)
    qvel_flat = x_flat[:, nq:]               # (N*T, nv)

    # nq = 7 (base) + A (joints) + 7 (obj)
    A = nq - 14 #A=29
    if A <= 0:
        raise ValueError(f"Cannot infer A from nq={nq} (expected nq = 7 + A + 7).")

    total = N * T
    robot_body_pos_w_flat  = np.zeros((total, num_bodies, 3), dtype=np.float32)
    robot_body_quat_w_flat = np.zeros((total, num_bodies, 4), dtype=np.float32)

    # preallocation
    anchor_pos_b_flat = np.zeros((total, 3), dtype=np.float32)
    anchor_ori_b_flat = np.zeros((total, 4), dtype=np.float32)

    # Base lin/ang vel
    base_lin_vel_flat = qvel_flat[:, 0:3].copy()        # (N*T, 3)
    base_ang_vel_flat = qvel_flat[:, 3:6].copy()        # (N*T, 3)

    # Joint pos/vel 
    joint_pos_flat = qpos_flat[:, 7:7 + A].copy()       # (N*T, A)
    joint_vel_flat = qvel_flat[:, 6:6 + A].copy()       # (N*T, A)

    #  object state from x

    obj_qpos_flat = qpos_flat[:, 7 + A : 7 + A + 7].copy()   # (N*T, 7)
    obj_qvel_flat = qvel_flat[:, 6 + A : 6 + A + 6].copy()   # (N*T, 6)

    obj_pos_flat  = obj_qpos_flat[:, 0:3]   # xyz
    obj_quat_flat = obj_qpos_flat[:, 3:7]   # wxyz

    obj_lin_vel_flat = obj_qvel_flat[:, 0:3]  # vx,vy,vz
    obj_ang_vel_flat = obj_qvel_flat[:, 3:6]  # wx,wy,wz

    #  sensor indices for torso_position / torso_orientatio
    sid_pos = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position")
    sid_ori = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_orientation")

    adr_pos, dim_pos = mj_model.sensor_adr[sid_pos], mj_model.sensor_dim[sid_pos]
    adr_ori, dim_ori = mj_model.sensor_adr[sid_ori], mj_model.sensor_dim[sid_ori]

    # robot body pos/quat
    for i in range(total):
        mj_data.qpos[:] = qpos_flat[i]
        mj_data.qvel[:] = qvel_flat[i]

        mujoco.mj_forward(mj_model, mj_data)
        for j, body_id in enumerate(body_ids):
            robot_body_pos_w_flat[i, j, :]  = mj_data.xpos[body_id] # world (x,y,z) of that link
            robot_body_quat_w_flat[i, j, :] = mj_data.xquat[body_id] # world orientation of that link

        anchor_pos_b_flat[i] = mj_data.sensordata[adr_pos:adr_pos + dim_pos]
        anchor_ori_b_flat[i] = mj_data.sensordata[adr_ori:adr_ori + dim_ori]

    # Reshape back to (N, T, ·)
    anchor_pos_b = anchor_pos_b_flat.reshape(N, T, 3)
    anchor_ori_b = anchor_ori_b_flat.reshape(N, T, 4)
    base_lin_vel = base_lin_vel_flat.reshape(N, T, 3)
    base_ang_vel = base_ang_vel_flat.reshape(N, T, 3)
    joint_pos    = joint_pos_flat.reshape(N, T, A)
    joint_vel    = joint_vel_flat.reshape(N, T, A)
    object_pos_w      = obj_pos_flat.reshape(N, T, 3)
    object_quat_w     = obj_quat_flat.reshape(N, T, 4)
    object_lin_vel_w  = obj_lin_vel_flat.reshape(N, T, 3)
    object_ang_vel_w  = obj_ang_vel_flat.reshape(N, T, 3)
    robot_body_pos_w  = robot_body_pos_w_flat.reshape(N, T, num_bodies, 3)
    robot_body_quat_w = robot_body_quat_w_flat.reshape(N, T, num_bodies, 4)



    # Saving
    out = dict(data)
    out["anchor_pos_b"] = anchor_pos_b
    out["anchor_ori_b"] = anchor_ori_b
    out["base_linvel_angvel"] = np.concatenate([base_lin_vel, base_ang_vel], axis=-1)
    out["base_lin_vel"] = base_lin_vel
    out["base_ang_vel"] = base_ang_vel
    out["joint_pos"]    = joint_vel
    out["joint_vel"]    = joint_vel
    out["object_pos_w"]     = object_pos_w
    out["object_quat_w"]    = object_quat_w
    out["object_lin_vel_w"] = object_lin_vel_w
    out["object_ang_vel_w"] = object_ang_vel_w
    out["robot_body_pos_w"]  = robot_body_pos_w
    out["robot_body_quat_w"] = robot_body_quat_w

    np.savez(npz_path, **out)
    print(f"Updated npz with RL obs: {npz_path}")
    print(f"  anchor_pos_b:        {anchor_pos_b.shape}")
    print(f"  anchor_ori_b:        {anchor_ori_b.shape}")
    print(f"  base_linvel_angvel:  {(N, T, 6)}")
    print(f"  base_lin_vel:        {base_lin_vel.shape}")
    print(f"  base_ang_vel:        {base_ang_vel.shape}")
    print(f"  joint_pos:        {joint_pos.shape}")
    print(f"  joint_vel:        {joint_vel.shape}")
    print(f"  object_pos_w:        {object_pos_w.shape}")
    print(f"  object_quat_w:       {object_quat_w.shape}")
    print(f"  object_lin_vel_w:    {object_lin_vel_w.shape}")
    print(f"  object_ang_vel_w:    {object_ang_vel_w.shape}")
    print(f"  robot_body_pos_w:    {robot_body_pos_w.shape}")
    print(f"  robot_body_quat_w:   {robot_body_quat_w.shape}")


def main():
    npz_path = "sbto/pretraining/rollout_time_x_u_obs_traj.npz"
    xml_path = "sbto/models/unitree_g1/scene_29dof.xml"
    
    add_rl_obs_from_x(npz_path, xml_path)


if __name__ == "__main__":
    main()