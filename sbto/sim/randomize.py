import numpy as np
from dataclasses import dataclass
from typing import Tuple, Union, Optional

from sbto.sim.sim_mj_rollout import Array, SimMjRollout
from sbto.tasks.cost import quaternion_dist_nb

def randomize_joint_pos(
    mj_model,
    N: int,
    x_0: np.ndarray,
    scale_q: Union[np.ndarray, float] = 0.1,
    scale_v: Union[np.ndarray, float] = 0.1,
    is_floating_base: bool = False,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomizes joint positions and velocities around a nominal state x_0.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu
    assert x_0.shape == (nq + nv,), f"x_0 must have shape ({nq + nv},), got {x_0.shape}"
    
    act_joint_ids = mj_model.actuator_trnid[:, 0]
    act_qposadr = mj_model.jnt_qposadr[act_joint_ids]
    act_dofsadr = mj_model.jnt_dofadr[act_joint_ids]
    
    scale_q = np.asarray(scale_q, dtype=float)
    scale_v = np.asarray(scale_v, dtype=float)

    q_0, v_0 = np.split(x_0, [nq])

    q = np.tile(q_0, (N, 1))
    v = np.tile(v_0, (N, 1))

    q[:, :act_qposadr[-1]] += rng.standard_normal((N, act_qposadr[-1])) * scale_q
    v[:, :act_dofsadr[-1]] += rng.standard_normal((N, act_dofsadr[-1])) * scale_v

    if is_floating_base:
        normalize_quat(q, slice(3, 7))

    return np.concatenate([q, v], axis=-1)

def normalize_quat(x_rand: np.ndarray, slice):
    x_rand = np.atleast_2d(x_rand)
    x_rand[:, slice] /= np.linalg.norm(x_rand[:, slice], axis=-1,  keepdims=True)
    return x_rand

def randomize_obj_pos(
    N: int,
    obj_pos_quat: np.ndarray,
    x_range: Tuple[float, float] = (-0.01, 0.01),
    y_range: Tuple[float, float] = (-0.01, 0.01),
    z_range: Tuple[float, float] = (0.0, 0.0),
    w_range: Tuple[float, float] = (-0.5, 0.5),
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Randomize the position (x, y, z) and yaw rotation of an object N times.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    obj_pos_quat = np.asarray(obj_pos_quat).copy()
    assert obj_pos_quat.shape == (7,)

    xyz = obj_pos_quat[:3]
    quat = obj_pos_quat[3:7]

    x_offsets = rng.uniform(*x_range, N)
    y_offsets = rng.uniform(*y_range, N)
    z_offsets = rng.uniform(*z_range, N)
    w_offsets = rng.uniform(*w_range, N)

    xyz_rand = np.tile(xyz, (N, 1))
    xyz_rand[:, 0] += x_offsets
    xyz_rand[:, 1] += y_offsets
    xyz_rand[:, 2] += z_offsets

    qw = np.cos(w_offsets / 2.0)
    qz = np.sin(w_offsets / 2.0)
    yaw_quats = np.stack(
        [qw, np.zeros_like(qw), np.zeros_like(qw), qz],
        axis=1,
    )

    quat_rand = np.empty_like(yaw_quats)
    for i in range(N):
        quat_rand[i] = quat_multiply(yaw_quats[i], quat)

    return np.hstack([xyz_rand, quat_rand])

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions.
    Both q1, q2 are [qw, qx, qy, qz].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def has_nonzero_range(rng):
    return any(v != 0.0 for v in rng)

@dataclass
class ConfigRandomizeRollout():
    rand_t_knots: int = 0
    scale_q: float = 0.05
    scale_v: float = 0.1
    obj_x_range: tuple = (-0.02, 0.03)
    obj_y_range: tuple = (-0.015, 0.015)
    obj_z_range: tuple = (0.0, 0.0)
    obj_w_range: tuple = (-0.3, 0.3)
    max_base_pos_dist: float = 0.25
    max_base_quat_dist: float = 0.25
    max_grav_quat_dist: float = 0.25
    _N_rollout_steps: int = 50
    _N_max_it: int = 25
    _N_samples: int = 128

class RandomizeRollout:
    def __init__(
        self,
        cfg: ConfigRandomizeRollout,
        sim: SimMjRollout,
        seed: int = 0,
    ):
        self.sim = sim
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        if cfg.rand_t_knots > 0:
            sim.randomize_t_knots(cfg.rand_t_knots, rng=self.rng)

        if self.is_randomized():
            self._set_random_initial_state()

    def is_randomized(self) -> bool:
        return (
            self.is_randomize_joints()
            or self.is_randomize_object()
        )

    def is_randomize_joints(self) -> bool:
        return (
            self.cfg.scale_q != 0.0
            or self.cfg.scale_v != 0.0
        )

    def is_randomize_object(self) -> bool:
        return any([
            has_nonzero_range(self.cfg.obj_x_range),
            has_nonzero_range(self.cfg.obj_y_range),
            has_nonzero_range(self.cfg.obj_z_range),
            has_nonzero_range(self.cfg.obj_w_range),
        ])


    def is_randomize(self) -> bool:
        return any([
            self.cfg.scale_q != 0.0,
            self.cfg.scale_v != 0.0,
            any(v != 0.0 for v in self.cfg.obj_x_range),
            any(v != 0.0 for v in self.cfg.obj_y_range),
            any(v != 0.0 for v in self.cfg.obj_z_range),
            any(v != 0.0 for v in self.cfg.obj_w_range),
        ])

    def is_randomize_obj(self) -> bool:
        return any([
            any(v != 0.0 for v in self.cfg.obj_x_range),
            any(v != 0.0 for v in self.cfg.obj_y_range),
            any(v != 0.0 for v in self.cfg.obj_z_range),
            any(v != 0.0 for v in self.cfg.obj_w_range),
        ])

    def _validate_states(self, states: np.ndarray) -> np.ndarray:
        sim = self.sim
        cfg = self.cfg

        # --- Base position ---
        base_pos_dist = np.linalg.norm(
            states[:, :3] - sim.x_0[:3], axis=-1
        )
        base_ok = base_pos_dist < cfg.max_base_pos_dist

        # --- Base orientation (relative to initial) ---
        quat = states[:, 3:7].reshape(-1, 1, 4)
        ref = sim.x_0[3:7].reshape(1, 4)
        w = np.ones_like(ref)
        quat_ok = quaternion_dist_nb(quat, ref, w) < cfg.max_base_quat_dist

        # --- Gravity alignment ---
        ref_grav = np.array([1., 0., 0., 0.]).reshape(1, 4)
        grav_ok = quaternion_dist_nb(quat, ref_grav, w) < cfg.max_grav_quat_dist

        valid = base_ok & quat_ok & grav_ok

        # --- Object fixed if not randomized ---
        if sim.mj_scene.is_obj and not self.is_randomize_object():
            obj_states = states[:, sim.mj_scene.obj_qpos_adr]          # (N, D)
            ref = sim.x_0[sim.mj_scene.obj_qpos_adr]                   # (D,)

            same_obj_pos = np.all(
                np.isclose(obj_states[:, :3], ref[None, :3], atol=5e-2),
                axis=-1,
            )
            same_obj_quat = np.all(
                np.isclose(obj_states[:, 3:7], ref[None, 3:7], atol=2e-2),
                axis=-1,
            )
            valid &= same_obj_pos & same_obj_quat

        return valid

    def _sample_initial_states(
        self,
        N: int,
        rollout_steps: int,
    ) -> np.ndarray:
        sim = self.sim
        cfg = self.cfg
        x0 = sim.x_0

        # --- Joint randomization ---
        states = randomize_joint_pos(
            sim.mj_scene.mj_model,
            N,
            x0,
            scale_q=cfg.scale_q,
            scale_v=cfg.scale_v,
            rng=self.rng,
        )

        # --- Object randomization ---
        if sim.mj_scene.is_obj and self.is_randomize_object():
            states[:, sim.mj_scene.obj_qpos_adr] = randomize_obj_pos(
                N,
                x0[sim.mj_scene.obj_qpos_adr],
                cfg.obj_x_range,
                cfg.obj_y_range,
                cfg.obj_z_range,
                cfg.obj_w_range,
                rng=self.rng,
            )

        # --- Optional rollout check ---
        if not self.is_randomize_joints():
            return states

        return self._rollout_and_extract_last(states, rollout_steps)

    def _rollout_and_extract_last(
        self,
        states: np.ndarray,
        T: int,
    ) -> np.ndarray:
        sim = self.sim
        N = states.shape[0]

        if sim.N_allocated != N:
            sim._init_batches(N, T)

        sim.initial_states[:, 1:] = states

        pd_target = states[:, sim.mj_scene.act_qposadr]
        pd_traj = np.tile(pd_target[:, None, :], (1, T, 1))

        _, rollout_states, _, _ = sim._rollout_dynamics(
            pd_traj, with_x0=False
        )

        return rollout_states[:, -self.cfg._N_rollout_steps // 3:, :].reshape(-1, states.shape[-1])

    def _set_random_initial_state(self):
        cfg = self.cfg
        sim = self.sim

        for _ in range(cfg._N_max_it):
            states = self._sample_initial_states(
                N=cfg._N_samples,
                rollout_steps=cfg._N_rollout_steps,
            )

            valid = self._validate_states(states)
            if np.any(valid):
                sim.set_initial_state(states[np.argmax(valid)])
                return

        print(
            f"[RandomizeRollout] Failed after {cfg._N_max_it} attempts"
        )