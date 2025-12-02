import pickle
import mujoco
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple

from sbto.utils.finite_diff import (
    finite_diff_qpos_traj_high_order,
    finite_diff_quat_traj,
)


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion array [T,4]."""
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] -> [w,x,y,z]."""
    return np.column_stack([q[:, 3], q[:, :3]])


def compute_time_array(fps: float, N: int) -> np.ndarray:
    return np.arange(N) / fps


# ------------------------------------------------------------
# Loading NPZ reference
# ------------------------------------------------------------

def load_npz_reference(path: str) -> Dict[str, np.ndarray]:
    """
    Loads reference from NPZ and extracts the required qpos fields.
    Only returns minimal dict: qpos, fps.
    """
    file = np.load(path)
    qpos = file["qpos"]
    fps = file["fps"].item() if isinstance(file["fps"], np.ndarray) else file["fps"]

    return {
        "qpos": qpos,
        "fps": float(fps),
    }


# ------------------------------------------------------------
# Trajectory processing utilities
# ------------------------------------------------------------

def interpolate_trajectory(
    values: np.ndarray, time: np.ndarray, t_new: np.ndarray, is_quat=False
):
    """Generic interpolation for batched arrays."""
    interp = interp1d(time, values, axis=0)
    out = interp(t_new)
    return normalize_quat(out) if is_quat else out


def compute_velocities(qpos_dict: Dict[str, np.ndarray], dt: float) -> Dict[str, np.ndarray]:
    """Compute velocities for root/dof/object segments."""
    out = {}

    # Root
    out["root_v"] = finite_diff_qpos_traj_high_order(qpos_dict["root_pos"], dt)
    out["root_w"] = finite_diff_quat_traj(qpos_dict["root_rot"], dt)

    # DOF
    out["dof_v"] = finite_diff_qpos_traj_high_order(qpos_dict["dof_pos"], dt)

    # Object (optional)
    if "object_root_pos" in qpos_dict:
        out["object_v"] = finite_diff_qpos_traj_high_order(qpos_dict["object_root_pos"], dt)
        out["object_w"] = finite_diff_quat_traj(qpos_dict["object_rot"], dt)

    return out


def slice_reference(qpos: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Splits qpos into semantic blocks.
    Assumes layout:
        [quat(4), root_pos(3), dof(...), object_rot(4), object_pos(3)]
    """
    root_rot = qpos[:, :4]
    root_pos = qpos[:, 4:7]

    # Middle = robot DOF, last 7 = object (quat + pos)
    dof_pos = qpos[:, 7:-7]
    object_rot = qpos[:, -7:-3]
    object_pos = qpos[:, -3:]

    return {
        "root_rot": root_rot,
        "root_pos": root_pos,
        "dof_pos": dof_pos,
        "object_rot": object_rot,
        "object_root_pos": object_pos,
    }


def concatenate_full_state(qpos_dict, vel_dict) -> np.ndarray:
    qpos = np.hstack([
        qpos_dict["root_pos"],
        qpos_dict["root_rot"],
        qpos_dict["dof_pos"],
        qpos_dict.get("object_root_pos", []),
        qpos_dict.get("object_rot", []),
    ])

    qvel = np.hstack([
        vel_dict["root_v"],
        vel_dict["root_w"],
        vel_dict["dof_v"],
        vel_dict.get("object_v", []),
        vel_dict.get("object_w", []),
    ])

    return np.hstack([qpos, qvel])


# ------------------------------------------------------------
# Main Class
# ------------------------------------------------------------

class ReferenceMotion:
    """
    Clean, minimal reference motion loader from NPZ.
    - Keeps only qpos + time + fps in data dict.
    - Everything else is provided via properties:
        root_pos, root_rot, dof_pos, etc.
    """

    def __init__(
        self,
        ref_motion_path: str,
        mj_model: mujoco.MjModel,
        t0: float = 0.,
        speedup: float = 1.0,
        z_offset: float = 0.0,
    ):
        # Load base data
        base = load_npz_reference(ref_motion_path)
        self.fps = base["fps"] * speedup
        self.qpos = base["qpos"]

        # Initial time array
        self.time = compute_time_array(self.fps, len(self.qpos))

        # Mujoco model properties
        self.mj_model = mj_model

        # Apply shift
        self.shift_start_time(t0)

        # Apply z-offset
        if z_offset != 0:
            self.qpos[:, 6] -= z_offset  # root_pos[2]
            self.qpos[:, -1] -= z_offset  # object_root_pos[2]

        # Interpolate to MJ timestep
        self.interpolate_to_mj_dt()

        # Pre-slice qpos into dict
        self._qpos_dict = slice_reference(self.qpos)

        # Compute velocities
        dt = self.mj_model.opt.timestep
        self._vel_dict = compute_velocities(self._qpos_dict, dt)

        # Full state vector
        self.x = concatenate_full_state(self._qpos_dict, self._vel_dict)

        # Precompute actuator addresses
        act_ids = self.mj_model.actuator_trnid[:, 0]
        self.act_qpos_adr = self.mj_model.jnt_qposadr[act_ids]

        self.sensor_data = {}
        self.extra = 0

    # ------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------

    def interpolate_to_mj_dt(self):
        dt_mj = self.mj_model.opt.timestep
        dt_in = 1.0 / self.fps

        if abs(dt_mj - dt_in) < 1e-9:
            return

        t_new = np.arange(0, self.time[-1], dt_mj)
        self.qpos = interpolate_trajectory(self.qpos, self.time, t_new)
        self.time = t_new

    # ------------------------------------------------------------
    # Time shifting
    # ------------------------------------------------------------

    def shift_start_time(self, t0: float):
        """Trim trajectory so that new time starts at t0."""
        if t0 <= 0:
            return

        idx = np.searchsorted(self.time, t0)
        self.qpos = self.qpos[idx:]
        self.time = self.time[idx:] - self.time[idx]

    # ------------------------------------------------------------
    # Extend trajectory
    # ------------------------------------------------------------

    def extend_to_length(self, T_needed: int):
        """
        Extend the trajectory to have at least T_needed timesteps
        by repeating the final timestep data.
        Automatically updates:
            - time
            - qpos
            - sliced qpos dict
            - velocities
            - full x state
        """
        T = len(self.time)
        if T_needed <= T:
            return

        self.extra = T_needed - T

        dt = self.mj_model.opt.timestep

        # --- Extend time ---
        new_times = self.time[-1] + dt * np.arange(1, self.extra + 1)
        self.time = np.concatenate([self.time, new_times], axis=0)

        # --- Extend qpos ---
        last_val = self.qpos[-1:]
        padding = np.repeat(last_val, repeats=self.extra, axis=0)
        self.qpos = np.concatenate([self.qpos, padding], axis=0)

        # Recompute sliced fields after qpos changed
        self._qpos_dict = slice_reference(self.qpos)

        # Recompute velocities
        dt_mj = self.mj_model.opt.timestep
        self._vel_dict = compute_velocities(self._qpos_dict, dt_mj)

        # Recompute x
        self.x = concatenate_full_state(self._qpos_dict, self._vel_dict)

    # ------------------------------------------------------------
    # Sensor extraction
    # ------------------------------------------------------------
    def add_sensor_data(self, sensor_names: List[str]):
        """
        Extracts sensor values for each timestep along the trajectory.
        The results are stored as attributes:
            self.<sensor_name>
        
        Sensor trajectory shape:
            [T, sensor_dim]
        """
        data = mujoco.MjData(self.mj_model)

        T = len(self.time)
        nq = self.mj_model.nq
        nv = self.mj_model.nv

        qpos_traj = self.x[:, :nq]
        qvel_traj = self.x[:, nq:]

        for sensor_name in sensor_names:
            sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            adr = self.mj_model.sensor_adr[sid]
            dim = self.mj_model.sensor_dim[sid]

            out = np.zeros((T, dim))

            for t in range(T):
                data.qpos[:] = qpos_traj[t]
                data.qvel[:] = qvel_traj[t]
                mujoco.mj_forward(self.mj_model, data)
                out[t] = data.sensordata[adr:adr + dim]

            self.sensor_data[sensor_name] = out

    # ------------------------------------------------------------
    # Lazy property getters (clean!)
    # ------------------------------------------------------------
    @property
    def T(self): return len(self.time)
    
    @property
    def x0(self): return self.x[0]

    @property
    def root_rot(self): return self._qpos_dict["root_rot"]

    @property
    def root_pos(self): return self._qpos_dict["root_pos"]

    @property
    def dof_pos(self): return self._qpos_dict["dof_pos"]

    @property
    def object_root_pos(self): return self._qpos_dict["object_root_pos"]

    @property
    def object_rot(self): return self._qpos_dict["object_rot"]

    @property
    def root_v(self): return self._vel_dict["root_v"]

    @property
    def root_w(self): return self._vel_dict["root_w"]

    @property
    def dof_v(self): return self._vel_dict["dof_v"]

    @property
    def object_v(self): return self._vel_dict.get("object_v")

    @property
    def object_w(self): return self._vel_dict.get("object_w")

    # ------------------------------------------------------------
    # Actuator utilities
    # ------------------------------------------------------------

    @property
    def act_qpos(self):
        return self.qpos[:, self.act_qpos_adr]

    @property
    def act_qpos0(self):
        return self.qpos[0, self.act_qpos_adr]

    # Range and mean
    def get_act_qpos_range(self):
        q = self.act_qpos
        return q.min(axis=0), q.max(axis=0)

    def get_act_qpos_mean(self):
        return np.mean(self.act_qpos, axis=0)
