import mujoco
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List

from sbto.sim.scene_mj import MjScene
from sbto.utils.finite_diff import (
    finite_diff_qpos_traj_high_order,
    finite_diff_quat_traj,
)

def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion array [T,4]."""
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] -> [w,x,y,z]."""
    return np.column_stack([q[:, 3], q[:, :3]])

def flip_quat_pos_in_traj(free_joint_traj: np.ndarray) -> np.ndarray:
    """Convert free joint [quat, pos] -> [pos, quat]."""
    free_joint_traj_flipped = np.empty_like(free_joint_traj)
    free_joint_traj_flipped[:, :3] = free_joint_traj[:, -3:]
    free_joint_traj_flipped[:, 3:] = free_joint_traj[:, :-3]
    return free_joint_traj_flipped

def compute_time_array(fps: float, N: int) -> np.ndarray:
    return np.arange(N) / fps

def load_npz_reference(path: str) -> Dict[str, np.ndarray]:
    """
    Loads reference from NPZ and extracts the required qpos fields.
    Only returns minimal dict: qpos, fps.
    """
    file = np.load(path, mmap_mode="r")
    qpos = file["qpos"]
    fps = int(file["fps"])

    return {
        "qpos": qpos,
        "fps": float(fps),
    }

def make_quaternions_continuous(quat_traj: np.ndarray):
    """
    Ensures that quaternions are continuous with sign flip.
    """
    quat_dot_prod = np.sum(quat_traj[:-1, :] * quat_traj[1:, :], axis=-1)
    sign_flip = np.argwhere(quat_dot_prod < 0) + 1
    for id in sign_flip:
        quat_traj[np.squeeze(id):, :] *= -1
    return quat_traj

def interpolate_trajectory(
    values: np.ndarray, time: np.ndarray, t_new: np.ndarray, is_quat=False
):
    """Generic interpolation for batched arrays."""
    if is_quat:
        values = make_quaternions_continuous(values)
        interp = interp1d(time, values, axis=0, copy=False, assume_sorted=True)
        return normalize_quat(interp(t_new))
    else:
        interp = interp1d(time, values, kind="cubic", axis=0, copy=False, assume_sorted=True)
        return interp(t_new)

class ReferenceMotion:
    """
    Clean, minimal reference motion loader from NPZ.
    - Keeps only qpos + time + fps in data dict.
    - Everything else is provided via properties:
        root_pos, root_rot, dof_pos, etc.
    """
    def __init__(
        self,
        mj_scene: MjScene,
        ref_motion_path: str,
        t0: float = 0.0,
        t_end: float = 0.0,
        speedup: float = 1.0,
        z_offset: float = 0.0,
        flip_quat_pos: bool = True,
        quat_wxyz: bool = True,
    ):
        self.mj_scene = mj_scene
        self.dt = self.mj_scene.dt
        self.sensor_data = {}

        # Load base data
        base = load_npz_reference(ref_motion_path)
        self.fps = base["fps"] * speedup
        self._qpos = base["qpos"]

        # Fix quaterion format
        if self.mj_scene.is_floating_base:
            base_qpos = np.concatenate((
                self.mj_scene.base_pos_adr,
                self.mj_scene.base_quat_adr
                ))
            if flip_quat_pos:
                self._qpos[:, base_qpos] = flip_quat_pos_in_traj(self._qpos[:, base_qpos])
            if not quat_wxyz:
                self._qpos[:, base_qpos] = quat_xyzw_to_wxyz(self._qpos[:, base_qpos])

        if self.mj_scene.is_obj:
            obj_qpos = self.mj_scene.obj_qpos_adr
            if flip_quat_pos:
                self._qpos[:, obj_qpos] = flip_quat_pos_in_traj(self._qpos[:, obj_qpos])
            if not quat_wxyz:
                self._qpos[:, obj_qpos] = quat_xyzw_to_wxyz(self._qpos[:, obj_qpos])

        self.time = compute_time_array(self.fps, len(self._qpos))
        self.trim_traj(t0, t_end)
        self.apply_z_offset(z_offset)
        self._qpos_dict = self.slice_reference(self._qpos)
        self._qpos_dict = self.interpolate_to_mj_dt(self._qpos_dict)
        self._vel_dict = self.compute_velocities(self._qpos_dict)
        self.x = self.concatenate_full_state(self._qpos_dict, self._vel_dict)

    def trim_traj(self, t0: float, t_end: float):
        """Trim trajectory so that new time starts at t0."""
        if t0 <= 0 and t_end <=0:
            return
        
        if t_end <= t0:
            return
        
        if t0 > 0:
            idx = np.searchsorted(self.time, t0)
            self._qpos = self._qpos[idx:]
            self.time = self.time[idx:] - self.time[idx]

        if t_end > 0:
            idx = np.searchsorted(self.time, t_end)
            self._qpos = self._qpos[:idx]
            self.time = self.time[:idx]

    def apply_z_offset(self, z_offset):
        if z_offset != 0:
            self._qpos[:, 2] -= z_offset  # root_pos[2]
            if self.mj_scene.is_obj:
                obj_qpos_z = self.mj_scene.obj_pos_adr[-1]
                self._qpos[:, obj_qpos_z] -= z_offset  # object_pos[2]

    def slice_reference(self, qpos: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Splits qpos into semantic blocks.
        Assumes layout:
            [quat(4), root_pos(3), dof(...), object_rot(4), object_pos(3)]
        """
        qpos_dict = {
            "dof_pos": qpos[:, self.mj_scene.act_qposadr],
        }
        if self.mj_scene.is_floating_base:
            qpos_dict.update({
                "root_pos": qpos[:, self.mj_scene.base_pos_adr],
                "root_rot": qpos[:, self.mj_scene.base_quat_adr],
            })
        if self.mj_scene.is_obj:
            qpos_dict.update({
                "object_rot": qpos[:, self.mj_scene.obj_quat_adr],
                "object_pos": qpos[:, self.mj_scene.obj_pos_adr],
            })
        return qpos_dict

    def interpolate_to_mj_dt(self, qpos_dict):
        dt_in = 1.0 / self.fps

        if abs(self.dt - dt_in) < 1e-4:
            return

        t_new = np.arange(0, self.time[-1], self.dt)
        qpos_dict_interp = {}
        for k, v in qpos_dict.items():
            is_quat = True if "rot" in k else False
            qpos_dict_interp[k] = interpolate_trajectory(v, self.time, t_new, is_quat=is_quat)
        
        self.time = t_new
        return qpos_dict_interp

    def compute_velocities(self, qpos_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute velocities for root/dof/object segments."""
        out = {}

        # Root
        if "root_pos" in qpos_dict:
            out["root_v"] = finite_diff_qpos_traj_high_order(qpos_dict["root_pos"], self.dt)
            out["root_w"] = finite_diff_quat_traj(qpos_dict["root_rot"], self.dt)

        # DOF
        out["dof_v"] = finite_diff_qpos_traj_high_order(qpos_dict["dof_pos"], self.dt)

        # Object (optional)
        if "object_pos" in qpos_dict:
            out["object_v"] = finite_diff_qpos_traj_high_order(qpos_dict["object_pos"], self.dt)
            out["object_w"] = finite_diff_quat_traj(qpos_dict["object_rot"], self.dt)

        return out

    def concatenate_full_state(self, qpos_dict, vel_dict) -> np.ndarray:
        all = []
        keys_qpos = [
            "root_pos",
            "root_rot",
            "dof_pos",
            "obj_pos",
            "obj_rot",
        ]
        keys_qvel = [
            "root_v",
            "root_w",
            "dof_v",
            "obj_v",
            "obj_w",
        ]
        for k in keys_qpos :
            v = qpos_dict.get(k, None)
            if v is not None:
                all.append(v)
        for k in keys_qvel:
            v = vel_dict.get(k, None)
            if v is not None:
                all.append(v)
        return np.hstack(all)

    def compute_sensor_data(self, sensor_names: List[str]):
        """
        Extracts sensor values for each timestep along the trajectory.
        The results are stored as attributes:
            self.<sensor_name>
        
        Sensor trajectory shape:
            [T, sensor_dim]
        """
        model = self.mj_scene.mj_model
        data = mujoco.MjData(model)
        T = len(self.time)

        def get_sid(sensor_name: str):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)

        sensor_name2adr = {
            sensor_name : model.sensor_adr[get_sid(sensor_name)]
            for sensor_name in sensor_names
        }
        sensor_name2dim = {
            sensor_name : model.sensor_dim[get_sid(sensor_name)]
            for sensor_name in sensor_names
        }
        self.sensor_data = {
            sensor_name : np.empty((T, sensor_name2dim[sensor_name]))
            for sensor_name in sensor_names
        }

        for t in range(T):
            data.qpos[:] =  self.x[t, :self.mj_scene.Nq]
            data.qvel[:] =  self.x[t, self.mj_scene.Nq:]
            mujoco.mj_forward(model, data)

            for sensor_name in sensor_names:
                adr = sensor_name2adr[sensor_name]
                dim = sensor_name2dim[sensor_name]
                self.sensor_data[sensor_name][t] = data.sensordata[adr:adr + dim]

    @property
    def T(self): return len(self.time)
    
    @property
    def x0(self): return self.x[0]

    @property
    def root_rot(self): return self._qpos_dict.get("root_rot")

    @property
    def root_pos(self): return self._qpos_dict.get("root_pos")

    @property
    def dof_pos(self): return self._qpos_dict["dof_pos"]

    @property
    def object_pos(self): return self._qpos_dict.get("object_pos")

    @property
    def object_rot(self): return self._qpos_dict.get("object_rot")

    @property
    def root_v(self): return self._vel_dict.get("root_v")

    @property
    def root_w(self): return self._vel_dict.get("root_w")

    @property
    def dof_v(self): return self._vel_dict["dof_v"]

    @property
    def object_v(self): return self._vel_dict.get("object_v")

    @property
    def object_w(self): return self._vel_dict.get("object_w")

    @property
    def act_qpos(self):
        return self.dof_pos

    @property
    def act_qpos0(self):
        return self.dof_pos[0]
    
    @property
    def act_qpos_range(self):
        return self.act_qpos.min(axis=0), self.act_qpos.max(axis=0)

    @property
    def act_qpos_mean(self):
        return np.mean(self.act_qpos, axis=0)
