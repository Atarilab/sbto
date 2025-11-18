import pickle
import mujoco
import numpy as np
from scipy.interpolate import interp1d
from typing import List

from sbto.utils.finite_diff import finite_diff_qpos_traj, finite_diff_quat_traj

def compute_time_from_fps(fps, N):
    return np.arange(N) / fps

def quatxyzw2quatwxyz(quat):
    new_quat = np.empty_like(quat)
    new_quat[:, 0] = quat[:, -1]
    new_quat[:, 1:] = quat[:, :3]
    return new_quat

def normalize_quat(quat):
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat

def concatenate_full_state(data):
    if "object_rot" in data and "object_root_pos" in data:
        qpos = np.concatenate(
            (
                data["root_pos"],
                data["root_rot"],
                data["dof_pos"],
                data["object_root_pos"],
                data["object_rot"],
            ), axis=-1
        )
        qvel = np.concatenate(
            (
                data["root_v"],
                data["root_w"],
                data["dof_v"],
                data["object_v"],
                data["object_w"],
            ), axis=-1
        )

    else:
        qpos = np.concatenate(
            (
                data["root_pos"],
                data["root_rot"],
                data["dof_pos"],
            ), axis=-1
        )
        qvel = np.concatenate(
            (
                data["root_v"],
                data["root_w"],
                data["dof_v"],
            ), axis=-1
        )

    x = np.concatenate((qpos, qvel), axis=-1)
    return qpos, qvel, x

def interpolate_data(data, dt: float):
    t_interp = np.arange(0, data["time"][-1], dt)
    NO_INTERP = ["time", "fps"]

    for k, v in data.items():
        if k not in NO_INTERP and v is not None:
            interpolate = interp1d(
                data["time"],
                y=v,
                axis=0,
            )
            data[k] = interpolate(t_interp)
            if "rot" in k:
                data[k] = normalize_quat(data[k])

    data["time"] = t_interp
    return data

def compute_vel_from_pos(data, dt: float):
    if "object_rot" in data and "object_root_pos" in data:
        data["object_v"] = finite_diff_qpos_traj(data["object_root_pos"], dt)
        data["object_w"] = finite_diff_quat_traj(data["object_rot"], dt)
    
    data["root_v"] = finite_diff_qpos_traj(data["root_pos"], dt)
    data["root_w"] = finite_diff_quat_traj(data["root_rot"], dt)
    data["dof_v"] = finite_diff_qpos_traj(data["dof_pos"], dt)

    return data

def extract_sensor_data(mj_model, state_traj, sensor_name: str):
    """Extracts sensors over a trajectory [T, nq+nv]."""
    data = mujoco.MjData(mj_model)

    sensor_info = []
    sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    adr = mj_model.sensor_adr[sid]
    dim = mj_model.sensor_dim[sid]
    sensor_info = (adr, dim)

    T = len(state_traj)
    sensor_data = []

    nq = mj_model.nq
    qpos_traj = state_traj[:, :nq]
    qvel_traj = state_traj[:, nq:]

    for t in range(T):
        data.qpos[:] = qpos_traj[t]
        data.qvel[:] = qvel_traj[t]
        mujoco.mj_forward(mj_model, data)

        step = []
        adr, dim = sensor_info
        step.append(np.copy(data.sensordata[adr:adr+dim]))
        sensor_data.append(step)

    try:
        sensor_data = np.squeeze(np.array(sensor_data))
    except Exception:
        pass

    return sensor_data

def load_reference(
    ref_motion_path: str,
    xml_path: str,
    speedup: float = 1.0,
    z_offset: float = 0.0,
    ):
    objs = []
    with open(ref_motion_path, "rb") as f:
        while True:
            try:
                objs.append(pickle.load(f))
            except EOFError:
                break

    data = {}
    for sub in objs:
        data.update(sub)

    N = len(data["root_pos"])
    data["time"] = compute_time_from_fps(data["fps"] * speedup, N)

    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    dt_interp = mj_model.opt.timestep
    if dt_interp > 0:
        data = interpolate_data(data, dt_interp)

    data = compute_vel_from_pos(data, dt_interp)

    if z_offset != 0:
        data["root_pos"][:, 2] -= z_offset
        if "object_root_pos" in data:
            data["object_root_pos"][:, 2] -= z_offset

    data["root_rot"] = quatxyzw2quatwxyz(data["root_rot"])
    if "object_rot" in data:
        data["object_rot"] = quatxyzw2quatwxyz(data["object_rot"])

    data["qpos"], data["qvel"], data["x"] = concatenate_full_state(data)

    return data

class ReferenceMotion:
    """
    Loads reference trajectories from pickled logs.
    Automatically loads and processes trajectory in __init__.
    """
    def __init__(
        self,
        ref_motion_path: str,
        xml_path: str,
        t0: float = 0.,
        speedup: float = 1.0,
        z_offset: float = 0.0,
    ):
        self.ref_motion_path = ref_motion_path
        self.xml_path = xml_path
        self.speedup = speedup
        self.z_offset = z_offset

        mj_model = mujoco.MjModel.from_xml_path(xml_path)
        act_joint_ids = mj_model.actuator_trnid[:, 0]
        self.act_qpos_adr = mj_model.jnt_qposadr[act_joint_ids] 

        self.data = load_reference(
            ref_motion_path,
            xml_path,
            speedup,
            z_offset,
        )
        self.shift_start_time(t0)

        for key, value in self.data.items():
            setattr(self, key, value)

    def add_sensor_data(self, mj_model, sensor_names: List[str]):
        for sensor_name in sensor_names:
            sensor_data = extract_sensor_data(mj_model, self.data["x"], sensor_name)
            self.data[sensor_name] = sensor_data

    def shift_start_time(self, t0: float):
        """
        Shift / trim the trajectory so that new time starts at 0
        and corresponds to the old time t0.

        Parameters
        ----------
        t0 : float
            The time (in seconds) at which the new trajectory should start.
        """
        time = self.data["time"]
        if t0 <= 0:
            return  # nothing to do

        # Find the first index with time >= t0
        i0 = np.searchsorted(time, t0)

        # Slice all time-dependent arrays
        for key, value in list(self.data.items()):
            # Only slice arrays with matching length on axis 0
            if isinstance(value, np.ndarray) and value.shape[0] == len(time):
                self.data[key] = value[i0:]

        # Reset time to start at zero
        self.data["time"] = self.data["time"] - self.data["time"][0]

    def extend_to_length(self, T_needed: int):
        """
        Extend all time-dependent arrays to have at least T_needed timesteps
        by repeating the last timestep's data.

        Parameters
        ----------
        T_needed : int
            Desired minimum number of timesteps.
        """
        time = self.data["time"]
        T = len(time)

        if T_needed <= T:
            return  # already long enough

        # How many steps need to be added
        extra = T_needed - T

        # Determine dt (use last interval, or 0 if length < 2)
        if T >= 2:
            dt = time[-1] - time[-2]
        else:
            dt = 0.0

        # --- Extend time ---
        new_times = time[-1] + dt * np.arange(1, extra + 1)
        self.data["time"] = np.concatenate([time, new_times], axis=0)

        # --- Extend each time-dependent array ---
        for key, arr in list(self.data.items()):
            if not isinstance(arr, np.ndarray):
                continue
            # Only extend arrays whose first axis is time-dependent
            if arr.shape[0] != T:
                continue

            last_val = arr[-1:]
            pad = np.repeat(last_val, repeats=extra, axis=0)
            self.data[key] = np.concatenate([arr, pad], axis=0)

        # Re-bind attributes (keep class in sync with .data)
        for key, value in self.data.items():
            setattr(self, key, value)
    
    def get_act_qpos_range(self):
        q_min = np.min(self.qpos[:, self.act_qpos_adr], axis=-1)
        q_max = np.max(self.qpos[:, self.act_qpos_adr], axis=-1)
        return q_min, q_max
    
    def get_act_qpos_mean(self):
        return np.mean(self.qpos[:, self.act_qpos_adr], axis=0)
    
    @property
    def qpos0(self):
        return self.qpos[0]
    
    @property
    def x0(self):
        return self.x[0]
    
    @property
    def act_qpos0(self):
        return self.qpos[0, self.act_qpos_adr]
    
    @property
    def act_qpos(self):
        return self.qpos[:, self.act_qpos_adr]

if __name__ == "__main__":
    import mujoco
    import os
    from sbto.utils.viewer import render_and_save_trajectory, visualize_trajectory
    from sbto.utils.plotting import plot_contact_plan
    import sbto.tasks.g1.constants as G1 

    path = "test/sub3_largebox_003.pkl"
    xml = "sbto/models/unitree_g1/scene_mjx.xml"

    data = load_reference(path, xml, sensor_names=G1.Sensors.FEET_CONTACTS, speedup=1.1, z_offset=0.025)
    print(data.keys())
    print(data['x'].shape)

    mj_model = mujoco.MjModel.from_xml_path(xml)
    mj_data = mujoco.MjData(mj_model)

    cnt_plan = np.stack([data[sns_cnt][:, 0] for sns_cnt in G1.Sensors.FEET_CONTACTS]).T
    cnt_plan[cnt_plan > 1] = 1
    print(cnt_plan.shape)
    plot_contact_plan(
        np.zeros_like(cnt_plan),
        cnt_plan
    )
    
    visualize_trajectory(mj_model, mj_data, data["time"], data["x"])
    file_name = os.path.split(path)[-1][:-4]
    # render_and_save_trajectory(mj_model, mj_data, data["time"], data["x"], save_path=f"test/test_{file_name}.mp4", fps=30)


