from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
)


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    """Safely normalize quaternions."""
    return q / torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


class TrajectoryNpzSimLoader:
    """Loads a single trajectory from your NPZ and resamples it to a fixed FPS.

    Expected NPZ format (your sbto file):

      time            : (T,) or (N, T)          float32
      base_xyz_quat   : (N, T, 7) or (T, 7)     [x, y, z, qw, qx, qy, qz]
      actuator_pos    : (N, T, A) or (T, A)

    We select one trajectory with index `traj_index` (default 0), then
    resample in time and compute base + joint velocities.
    """

    def __init__(
        self,
        input_file: str,
        output_fps: int,
        device: torch.device | str,
        traj_index: int = 0,
        repeat_last_frame: int = 0,
    ):
        self.input_file = input_file
        self.output_fps = int(output_fps)
        self.output_dt = 1.0 / float(self.output_fps)
        self.device = device
        self.traj_index = int(traj_index)
        self.current_idx = 0
        self.repeat_last_frame = max(0, int(repeat_last_frame))

        self._load()
        self._resample_to_output_fps_using_times()
        self._repeat_last_frame()
        self._compute_velocities_from_resampled()

    # -------------------------------------------------------------------------
    # NPZ format and pick one trajectory
    # -------------------------------------------------------------------------
    def _load(self) -> None:
        data = np.load(self.input_file, allow_pickle=True)

        if "base_xyz_quat" not in data or "actuator_pos" not in data or "time" not in data:
            raise RuntimeError(
                f"Expected keys 'base_xyz_quat', 'actuator_pos', 'time' in {self.input_file}"
            )

        base_xyz_quat = data["base_xyz_quat"].astype(np.float32)  # (N, T, 7) or (T, 7)
        actuator_pos = data["actuator_pos"].astype(np.float32)    # (N, T, A) or (T, A)
        time_np = data["time"].astype(np.float32)                 # (T,) or (N, T)

        # Handle batched (N, T, ·) and unbatched (T, ·) formats.
        if base_xyz_quat.ndim == 3:
            N, T, D = base_xyz_quat.shape
            assert D == 7, f"base_xyz_quat last dim must be 7, got {D}"
            assert actuator_pos.shape[0] == N and actuator_pos.shape[1] == T, \
                "actuator_pos must match base_xyz_quat in first two dims"

            idx = self.traj_index
            if not (0 <= idx < N):
                raise IndexError(f"traj_index={idx} but N={N}")

            base_xyz_quat_i = base_xyz_quat[idx]     # (T, 7)
            actuator_pos_i = actuator_pos[idx]       # (T, A)

            if time_np.ndim == 2:
                times_np = time_np[idx]              # (T,)
            else:
                times_np = time_np                   # (T,)

        elif base_xyz_quat.ndim == 2:
            # Already a single trajectory (T, 7)
            base_xyz_quat_i = base_xyz_quat
            actuator_pos_i = actuator_pos
            if time_np.ndim == 1:
                times_np = time_np
            elif time_np.ndim == 2:
                # Pick row 0 as a fallback
                times_np = time_np[0]
            else:
                raise RuntimeError(f"Unsupported time shape: {time_np.shape}")
        else:
            raise RuntimeError(f"Unsupported base_xyz_quat shape: {base_xyz_quat.shape}")

        # Sanity checks
        assert times_np.ndim == 1, f"time must be 1D after selecting traj, got {times_np.shape}"
        T = times_np.shape[0]
        assert base_xyz_quat_i.shape[0] == T, "base_xyz_quat and time length must match"
        assert actuator_pos_i.shape[0] == T, "actuator_pos and time length must match"

        self.input_times_np = times_np
        self.input_times = torch.from_numpy(times_np).to(self.device)

        # Extract base pos + quaternion
        base_pos = base_xyz_quat_i[:, 0:3]   # (T, 3)
        base_quat = base_xyz_quat_i[:, 3:7] # (T, 4) wxyz

        self.motion_base_poss_input = torch.from_numpy(base_pos).to(self.device)
        self.motion_base_rots_input = _normalize_quat(
            torch.from_numpy(base_quat).to(self.device)
        )
        self.motion_dof_poss_input = torch.from_numpy(actuator_pos_i).to(self.device)

        # No object in your format
        self.has_object = False

        T_len = len(self.input_times_np)
        self.input_frames = int(T_len)
        if T_len > 1:
            self.duration = float(self.input_times_np[-1] - self.input_times_np[0])
            self.input_dt = float(self.duration / max(1, T_len - 1))
            self.input_fps = int(round(1.0 / max(1e-8, self.input_dt)))
        else:
            self.duration = 0.0
            self.input_dt = self.output_dt
            self.input_fps = int(round(1.0 / self.input_dt))

        print(f"Loaded trajectory with T={T_len}, input_dt={self.input_dt:.6f}, "
              f"input_fps≈{self.input_fps}")

    # -------------------------------------------------------------------------
    # 2) Resample to desired FPS using time stamps
    # -------------------------------------------------------------------------
    def _resample_to_output_fps_using_times(self) -> None:
        if self.input_frames <= 1:
            self.motion_base_poss = self.motion_base_poss_input
            self.motion_base_rots = self.motion_base_rots_input
            self.motion_dof_poss = self.motion_dof_poss_input
            self.output_frames = self.input_frames
            return

        t0 = float(self.input_times_np[0])
        duration = float(self.input_times_np[-1] - self.input_times_np[0])
        if duration <= 0.0:
            self.motion_base_poss = self.motion_base_poss_input
            self.motion_base_rots = self.motion_base_rots_input
            self.motion_dof_poss = self.motion_dof_poss_input
            self.output_frames = self.input_frames
            return

        times_out = np.arange(0.0, duration + 1e-8, self.output_dt, dtype=np.float32) + t0
        self.output_frames = int(times_out.shape[0])

        idx1 = np.searchsorted(self.input_times_np, times_out, side="right")
        idx1 = np.clip(idx1, 1, self.input_frames - 1)
        idx0 = idx1 - 1
        t0s = self.input_times_np[idx0]
        t1s = self.input_times_np[idx1]
        denom = np.maximum(1e-8, (t1s - t0s))
        blend_np = (times_out - t0s) / denom

        index_0 = torch.from_numpy(idx0.astype(np.int64)).to(self.device)
        index_1 = torch.from_numpy(idx1.astype(np.int64)).to(self.device)
        blend = torch.from_numpy(blend_np.astype(np.float32)).to(self.device)

        # Base positions
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )

        # Base quaternions: blend with sign correction
        q0 = self.motion_base_rots_input[index_0]
        q1 = self.motion_base_rots_input[index_1]
        dot = (q0 * q1).sum(-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        q = q0 * (1 - blend.unsqueeze(1)) + q1 * blend.unsqueeze(1)
        self.motion_base_rots = _normalize_quat(q)

        # Joint positions
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

    # -------------------------------------------------------------------------
    # 3) Optionally repeat last frame
    # -------------------------------------------------------------------------
    def _repeat_last_frame(self) -> None:
        if self.repeat_last_frame <= 0:
            return

        last_pos = self.motion_base_poss[-1:].repeat(self.repeat_last_frame, 1)
        last_rot = self.motion_base_rots[-1:].repeat(self.repeat_last_frame, 1)
        last_dof = self.motion_dof_poss[-1:].repeat(self.repeat_last_frame, 1)

        self.motion_base_poss = torch.cat([self.motion_base_poss, last_pos], dim=0)
        self.motion_base_rots = torch.cat([self.motion_base_rots, last_rot], dim=0)
        self.motion_dof_poss = torch.cat([self.motion_dof_poss, last_dof], dim=0)

        self.output_frames += self.repeat_last_frame

    # -------------------------------------------------------------------------
    # 4) Compute base and joint velocities from resampled positions
    # -------------------------------------------------------------------------
    def _compute_velocities_from_resampled(self) -> None:
        # Linear vel from pos derivative and angular vel from quaternion finite diff
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]

        q = self.motion_base_rots
        if q.shape[0] >= 3:
            q_prev, q_next = q[:-2], q[2:]
            q_rel = quat_mul(q_next, quat_conjugate(q_prev))
            omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
            # Pad ends
            omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        else:
            omega = torch.zeros_like(self.motion_base_poss)
        self.motion_base_ang_vels = omega

    # -------------------------------------------------------------------------
    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def get_next_state(
        self,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        bool,
    ]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= getattr(self, "output_frames", self.input_frames):
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


# -----------------------------------------------------------------------------
# Conversion: run sim, log full body states, save mjlab motion.npz
# -----------------------------------------------------------------------------
def convert(
    input_file: str,
    output_file: str,
    output_fps: float = 50.0,
    device: str = "cpu",
    traj_index: int = 0,
    repeat_last_frame: int = 0,
):
    """Convert your sbto NPZ to a mjlab motion.npz using the G1FlatEnv scene.

    Output keys:
      fps               : [int]
      joint_pos         : (T_out, n_joints)
      joint_vel         : (T_out, n_joints)
      body_pos_w        : (T_out, n_bodies, 3)
      body_quat_w       : (T_out, n_bodies, 4)
      body_lin_vel_w    : (T_out, n_bodies, 3)
      body_ang_vel_w    : (T_out, n_bodies, 3)
    """
    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / float(output_fps)

    scene = Scene(G1FlatEnvCfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    motion = TrajectoryNpzSimLoader(
        input_file=input_file,
        output_fps=int(round(output_fps)),
        device=sim.device,
        traj_index=traj_index,
        repeat_last_frame=repeat_last_frame,
    )

    robot: Entity = scene["robot"]

    print(f"Robot has {len(robot.body_names)} bodies: {robot.body_names}")

    log: dict[str, Any] = {
        "fps": [int(round(output_fps))],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }

    frames_total = getattr(motion, "output_frames", motion.input_frames)
    pbar = tqdm(total=frames_total, desc="Converting", unit="frame", ncols=100)

    scene.reset()
    file_saved = False

    while not file_saved:
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # Root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]  # place in env
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        # angular vel in base frame
        root_states[:, 10:] = quat_apply_inverse(motion_base_rot, motion_base_ang_vel)
        robot.write_root_state_to_sim(root_states)

        # Joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, :] = motion_dof_pos
        joint_vel[:, :] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        # Log everything in world frame
        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())

        pbar.update(1)
        if reset_flag:
            file_saved = True

    pbar.close()

    # Stack arrays
    for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[k] = np.stack(log[k], axis=0)

    np.savez(output_file, **log)  # type: ignore[arg-type]
    print(f"\nSaved mjlab motion to: {output_file}")
    print(f"Output contains {log['body_pos_w'].shape[0]} frames with "
          f"{log['body_pos_w'].shape[1]} body links")
    print(f"  All robot bodies: {robot.body_names}")


def main(
    input_file: str,
    output_file: str,
    output_fps: float = 50.0,
    device: str = "cpu",
    traj_index: int = 0,
    repeat_last_frame: int = 0,
):
    convert(
        input_file=input_file,
        output_file=output_file,
        output_fps=output_fps,
        device=device,
        traj_index=traj_index,
        repeat_last_frame=repeat_last_frame,
    )


if __name__ == "__main__":
    tyro.cli(main)