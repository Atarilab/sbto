import os
import numpy as np
import mujoco


def build_state_traj_from_npz(npz_path, mj_model): #reconstruct exactly what MuJoCo expects in data.qpos and data.qvel 
    data = np.load(npz_path)

    # time                 (T,)
    # actuator_pos         (T, 29)
    # actuator_vel         (T, 29)
    # base_xyz_quat        (T, 7)
    # base_linvel_angvel   (T, 6)
    # obj_0_xyz_quat       (T, 7)
    # obj_0_linvel_angvel  (T, 6)
    
    act_pos              = data["actuator_pos"].astype(np.float32)          # (T, 29)
    act_vel              = data["actuator_vel"].astype(np.float32)          # (T, 29)
    base_xyz_quat        = data["base_xyz_quat"].astype(np.float32)         # (T, 7)
    base_linvel_angvel   = data["base_linvel_angvel"].astype(np.float32)    # (T, 6)
    obj_xyz_quat         = data["obj_0_xyz_quat"].astype(np.float32)        # (T, 7)
    obj_linvel_angvel    = data["obj_0_linvel_angvel"].astype(np.float32)   # (T, 6)

    T, A = act_pos.shape

    # DOFs from MuJoCo
    nq = mj_model.nq   # 43
    nv = mj_model.nv   # 41


    expected_nq = 7 + A + 7 # base pose + all joint angles + object pose
    expected_nv = 6 + A + 6

    if nq != expected_nq:
        raise ValueError(f"Expected nq = 7 + {A} + 7 = {expected_nq}, got {nq}")
    if nv != expected_nv:
        raise ValueError(f"Expected nv = 6 + {A} + 6 = {expected_nv}, got {nv}")

    # Build qpos and qvel to match the model ordering
    qpos = np.concatenate([base_xyz_quat, act_pos, obj_xyz_quat], axis=-1)        # (T, nq=43)
    qvel = np.concatenate([base_linvel_angvel, act_vel, obj_linvel_angvel], axis=-1)  # (T, nv=41)

    # State trajectory: [qpos, qvel]
    state_traj = np.concatenate([qpos, qvel], axis=-1)   # (T, nq+nv = 84)

    return state_traj


def extract_anchor_from_traj(xml_path, state_traj):
    """
    Compute anchor_pos_b and anchor_ori_b from a state trajectory using MuJoCo sensors.

    Uses the sensors defined in g1_mjx.xml:

        <framepos name="torso_position"   ... />
        <framequat name="torso_orientation" ... />

    Args:
        xml_path   (str): path to g1_mjx.xml
        state_traj (np.ndarray): (T, nq + nv) [qpos | qvel] for each timestep

    Returns:
        anchor_pos_b: (T, 3)  world position of the torso site
        anchor_ori_b: (T, 4)  world orientation of the torso site as quaternion
    """
    # Load model and data
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(mj_model)

    # Sensors 
    sensor_names = ["torso_position", "torso_orientation"]

    sensor_info = []  
    #extract relevant sensors (torso pos + ori)
    for name in sensor_names:
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name) # returns the sensor index
        adr = mj_model.sensor_adr[sid]
        dim = mj_model.sensor_dim[sid]
        sensor_info.append((name, adr, dim))

    T  = state_traj.shape[0]
    nq = mj_model.nq

    # Split state_traj into qpos and qvel
    qpos_traj = np.copy(state_traj[:, :nq])
    qvel_traj = np.copy(state_traj[:, nq:])

    anchor_pos_b = np.zeros((T, 3), dtype=np.float32)
    anchor_ori_b = np.zeros((T, 4), dtype=np.float32)

    for t in range(T):
        data.qpos[:] = qpos_traj[t]
        data.qvel[:] = qvel_traj[t]

        # Mujoco function to compute all sensors
        mujoco.mj_forward(mj_model, data)

        for name, adr, dim in sensor_info:
            values = np.copy(data.sensordata[adr:adr + dim])
            if name == "torso_position":
                # 3D position of the torso site in world frame
                anchor_pos_b[t] = values  # (3,)
            elif name == "torso_orientation":
                # Quaternion of the torso site in world frame
                anchor_ori_b[t] = values  # (4,)

    return anchor_pos_b, anchor_ori_b


def main():
   
    xml_path = "sbto/models/unitree_g1/scene_29dof.xml"
    npz_path = "sbto/data/time_x_u_traj_rl_format.npz"

    # Load model
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    #  state trajectory [qpos | qvel] from RL-format npz
    state_traj = build_state_traj_from_npz(npz_path, mj_model)

    anchor_pos_b, anchor_ori_b = extract_anchor_from_traj(xml_path, state_traj)

    old = dict(np.load(npz_path))

    old["anchor_pos_b"] = anchor_pos_b.astype(np.float32)  # (T, 3)
    old["anchor_ori_b"] = anchor_ori_b.astype(np.float32)  # (T, 4, quaternion)

    np.savez(npz_path, **old)
    print(f"Updated npz file with anchor_pos_b and anchor_ori_b: {npz_path}")


if __name__ == "__main__":
    main()