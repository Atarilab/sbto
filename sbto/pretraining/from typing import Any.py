from typing import Any
import mujoco
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
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
  return q / torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)

def extract_contacts_from_distance(
  eef_positions: np.ndarray,
  object_position: np.ndarray,
  object_rotation: np.ndarray | None = None,
  object_size: np.ndarray | None = None,
  threshold: float = 0.05,
) -> dict[str, Any]:
  """Extract contact information using distance threshold with object size consideration.
  
  This is a simpler fallback method that checks if end effector positions are within
  the object's bounding box (accounting for size) plus a threshold margin.
  
  Args:
    eef_positions: (num_eefs, 3) array of end effector positions in world frame
    object_position: (3,) array of object position (center) in world frame
    object_rotation: (4,) array of object quaternion (w, x, y, z) in world frame, or None for axis-aligned
    object_size: (3,) array of object half-extents (size in MuJoCo convention), or None to use threshold only
    threshold: Distance threshold in meters for contact detection (default: 0.05m = 5cm)
               If object_size is None, this is used as distance from center.
               If object_size is provided, this is added as margin around the bounding box.
    
  Returns:
    Dictionary with contact information:
      - contact_positions: (num_eefs, 3) array of contact positions (closest point on object if in contact, NaN otherwise)
      - contact_indicators: (num_eefs,) boolean array indicating if contact exists
  """
  num_eefs = eef_positions.shape[0]
  contact_indicators = np.zeros(num_eefs, dtype=bool)
  contact_positions = np.full((num_eefs, 3), np.nan, dtype=np.float32)
  
  # If object size is not provided, use simple distance from center
  if object_size is None:
    for i in range(num_eefs):
      dist = np.linalg.norm(eef_positions[i] - object_position)
      if dist <= threshold:
        contact_indicators[i] = True
        contact_positions[i] = object_position.copy()
  else:
    # Transform end effector positions to object's local frame
    if object_rotation is not None:
      # Convert quaternion to rotation matrix and transform
      # quat_apply_inverse: transform from world to object frame
      # Ensure float32 dtype for torch operations
      eef_pos_local = quat_apply_inverse(
        torch.from_numpy(object_rotation.astype(np.float32)).unsqueeze(0),
        torch.from_numpy((eef_positions - object_position.reshape(1, 3)).astype(np.float32))
      ).numpy()
    else:
      # No rotation, just translate
      eef_pos_local = eef_positions - object_position.reshape(1, 3)
    
    # Check if end effector is within bounding box + threshold
    # object_size is half-extents, so bounds are [-size-threshold, +size+threshold]
    for i in range(num_eefs):
      # Skip if position is NaN (end effector not found)
      if np.isnan(eef_pos_local[i]).any():
        continue
      
      # Check if within bounding box (with threshold margin)
      # object_size + threshold broadcasts correctly: (3,) + scalar -> (3,)
      bounds = object_size + threshold
      within_bounds = np.all(
        np.abs(eef_pos_local[i]) <= bounds
      )
      
      if within_bounds:
        contact_indicators[i] = True
        # Find closest point on the cube surface
        # Clamp to bounding box bounds
        closest_local = np.clip(eef_pos_local[i], -object_size, object_size)
        # Transform back to world frame
        if object_rotation is not None:
          # Ensure float32 dtype for torch operations
          closest_world = quat_apply(
            torch.from_numpy(object_rotation.astype(np.float32)).unsqueeze(0),
            torch.from_numpy(closest_local.astype(np.float32)).unsqueeze(0)
          ).numpy()[0] + object_position
        else:
          closest_world = closest_local + object_position
        contact_positions[i] = closest_world
  
  return {
    "contact_positions": contact_positions,
    "contact_indicators": contact_indicators,
  }


class TrajectoryNpzSimLoader:
  def __init__(
    self,
    input_file: str,
    output_fps: int,
    device: torch.device | str,
    repeat_last_frame: int = 0,
  ):
    self.input_file = input_file
    self.output_fps = int(output_fps)
    self.output_dt = 1.0 / float(self.output_fps)
    self.device = device
    self.current_idx = 0
    self.repeat_last_frame = max(0, int(repeat_last_frame))

    self._load()
    self._resample_to_output_fps_using_times()
    self._repeat_last_frame()
    self._compute_velocities_from_resampled()

  def _load(self) -> None:
    data = np.load(self.input_file, allow_pickle=True)

    # Check if this is Victor format or standard format
    is_victor_format = "base_xyz_quat" in data and "actuator_pos" in data
    
    if is_victor_format:
      # Victor format: separate arrays for base and actuator data
      assert "time" in data, f"Key 'time' not found in {self.input_file}"
      times_np = data["time"].astype(np.float32)
      
      # Load base pose (xyz + quat)
      base_xyz_quat = data["base_xyz_quat"].astype(np.float32)  # (T, 7): [x, y, z, qw, qx, qy, qz]
      assert base_xyz_quat.shape[1] == 7, f"base_xyz_quat must have 7 columns, got {base_xyz_quat.shape}"
      
      # Load actuator positions
      actuator_pos = data["actuator_pos"].astype(np.float32)  # (T, 29)
      assert actuator_pos.shape[1] == 29, f"actuator_pos must have 29 columns, got {actuator_pos.shape}"
      
      T = times_np.shape[0]
      assert base_xyz_quat.shape[0] == T, "base_xyz_quat length must match time length"
      assert actuator_pos.shape[0] == T, "actuator_pos length must match time length"
      
      self.input_times_np = times_np
      self.input_times = torch.from_numpy(times_np).to(self.device)
      
      # Extract base position and rotation
      self.motion_base_poss_input = torch.from_numpy(base_xyz_quat[:, 0:3]).to(self.device)  # xyz
      self.motion_base_rots_input = _normalize_quat(
        torch.from_numpy(base_xyz_quat[:, 3:7]).to(self.device)  # quat (w, x, y, z)
      )
      self.motion_dof_poss_input = torch.from_numpy(actuator_pos).to(self.device)
      
      # Check for object data
      self.has_object = False
      if "obj_0_xyz_quat" in data:
        obj_xyz_quat = data["obj_0_xyz_quat"].astype(np.float32)  # (T, 7)
        assert obj_xyz_quat.shape == (T, 7), f"obj_0_xyz_quat must be shape ({T}, 7), got {obj_xyz_quat.shape}"
        self.has_object = True
        self.object_pos_input = torch.from_numpy(obj_xyz_quat[:, 0:3]).to(self.device)
        self.object_rots_input = _normalize_quat(
          torch.from_numpy(obj_xyz_quat[:, 3:7]).to(self.device)
        )
    else:
      # Standard format: single 'x' array
      assert "x" in data, f"Key 'x' not found in {self.input_file}"
      x_np = data["x"]
      assert x_np.ndim == 2 and x_np.shape[1] in (71, 84), (
        f"x must be (T,71) or (T,84); got {x_np.shape}"
      )

      assert "time" in data, f"Key 'time' not found in {self.input_file}"
      times_np = data["time"].astype(np.float32)
      assert times_np.ndim == 1 and times_np.shape[0] == x_np.shape[0], (
        "times must be 1D and match x length"
      )

      self.input_times_np = times_np
      self.input_times = torch.from_numpy(times_np).to(self.device)

      T = x_np.shape[0]
      rs = torch.from_numpy(x_np.astype(np.float32)).to(self.device)

      # Use only qpos portions: [pos(3), quat wxyz(4), joint pos(29)]
      # New format (T,84) has x = (qpos(43), qvel(41)). Legacy (T,71) keeps only qpos fields.
      self.motion_base_poss_input = rs[:, 0:3]
      self.motion_base_rots_input = _normalize_quat(rs[:, 3:7])
      self.motion_dof_poss_input = rs[:, 7:36]

      # Optional object
      # Priority: if x has embedded object in qpos (shape 84), use that; otherwise fallback to separate 'object_states'
      self.has_object = False
      if x_np.shape[1] == 84:
        # x = (qpos(43), qvel(41)) where:
        # qpos[0:7] - first 7 entries (base pos(3) + base quat(4))
        # qpos[7:36] - next 29 entries (joint positions)
        # qpos[36:40] = obj_quat_wxyz (4 values: w, x, y, z)
        # qpos[40:43] = obj_pos (3 values: x, y, z)
        self.has_object = True
        self.object_rots_input = _normalize_quat(rs[:, 36:40])
        self.object_pos_input = rs[:, 40:43]
        assert self.object_rots_input.shape == (T, 4), (
          f"object_rots_input shape mismatch: expected ({T}, 4), got {self.object_rots_input.shape}"
        )
        assert self.object_pos_input.shape == (T, 3), (
          f"object_pos_input shape mismatch: expected ({T}, 3), got {self.object_pos_input.shape}"
        )
      elif "object_states" in data:
        obj_np = data["object_states"].astype(np.float32)
        assert obj_np.ndim == 2 and obj_np.shape[0] == T and obj_np.shape[1] == 7, (
          "object_states must be shape (T,7)"
        )
        self.has_object = True
        self.object_pos_input = torch.from_numpy(obj_np[:, 0:3]).to(self.device)
        self.object_rots_input = _normalize_quat(torch.from_numpy(obj_np[:, 3:7]).to(self.device))

    T = len(self.input_times_np)
    self.input_frames = int(T)
    self.duration = float(self.input_times_np[-1] - self.input_times_np[0]) if T > 1 else 0.0
    self.input_dt = float(self.duration / max(1, T - 1)) if T > 1 else self.output_dt
    self.input_fps = int(round(1.0 / max(1e-8, self.input_dt)))

  def _resample_to_output_fps_using_times(self) -> None:
    if self.input_frames <= 1:
      self.motion_base_poss = self.motion_base_poss_input
      self.motion_base_rots = self.motion_base_rots_input
      self.motion_dof_poss = self.motion_dof_poss_input
      if self.has_object:
        self.object_poss = self.object_pos_input
        self.object_rots = self.object_rots_input
      self.output_frames = self.input_frames
      return

    t0 = float(self.input_times_np[0])
    duration = float(self.input_times_np[-1] - self.input_times_np[0])
    if duration <= 0.0:
      self.motion_base_poss = self.motion_base_poss_input
      self.motion_base_rots = self.motion_base_rots_input
      self.motion_dof_poss = self.motion_dof_poss_input
      if self.has_object:
        self.object_poss = self.object_pos_input
        self.object_rots = self.object_rots_input
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

    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0], self.motion_base_poss_input[index_1], blend.unsqueeze(1)
    )
    q0 = self.motion_base_rots_input[index_0]
    q1 = self.motion_base_rots_input[index_1]
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    q = q0 * (1 - blend.unsqueeze(1)) + q1 * blend.unsqueeze(1)
    self.motion_base_rots = _normalize_quat(q)
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0], self.motion_dof_poss_input[index_1], blend.unsqueeze(1)
    )

    if self.has_object:
      self.object_poss = self._lerp(self.object_pos_input[index_0], self.object_pos_input[index_1], blend.unsqueeze(1))
      oq0 = self.object_rots_input[index_0]
      oq1 = self.object_rots_input[index_1]
      odot = (oq0 * oq1).sum(-1, keepdim=True)
      oq1 = torch.where(odot < 0, -oq1, oq1)
      oq = oq0 * (1 - blend.unsqueeze(1)) + oq1 * blend.unsqueeze(1)
      self.object_rots = _normalize_quat(oq)

  def _repeat_last_frame(self) -> None:
    """Append the last frame N times to extend the trajectory."""
    if self.repeat_last_frame <= 0:
      return

    last_pos = self.motion_base_poss[-1:].repeat(self.repeat_last_frame, 1)
    last_rot = self.motion_base_rots[-1:].repeat(self.repeat_last_frame, 1)
    last_dof = self.motion_dof_poss[-1:].repeat(self.repeat_last_frame, 1)

    self.motion_base_poss = torch.cat([self.motion_base_poss, last_pos], dim=0)
    self.motion_base_rots = torch.cat([self.motion_base_rots, last_rot], dim=0)
    self.motion_dof_poss = torch.cat([self.motion_dof_poss, last_dof], dim=0)

    if self.has_object:
      last_obj_pos = self.object_poss[-1:].repeat(self.repeat_last_frame, 1)
      last_obj_rot = self.object_rots[-1:].repeat(self.repeat_last_frame, 1)
      self.object_poss = torch.cat([self.object_poss, last_obj_pos], dim=0)
      self.object_rots = torch.cat([self.object_rots, last_obj_rot], dim=0)

    self.output_frames += self.repeat_last_frame

  def _compute_velocities_from_resampled(self) -> None:
    # lin vel from pos derivative and ang vel from quaternion finite difference
    self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
    self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]

    q = self.motion_base_rots
    if q.shape[0] >= 3:
      q_prev, q_next = q[:-2], q[2:]
      q_rel = quat_mul(q_next, quat_conjugate(q_prev))
      omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
      omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    else:
      omega = torch.zeros_like(self.motion_base_poss)
    self.motion_base_ang_vels = omega

  def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return a * (1 - blend) + b * blend

  def get_next_state(self) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], bool]:
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


def convert(
  input_file: str,
  output_file: str,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  repeat_last_frame: int = 0,
  contact_method: str = "distance",
  contact_threshold: float = 0.05,
):
  """Convert dataset NPZ (qpos in x) to mjlab motion.npz using simulation states.
  
  Extracts data for all robot bodies to match the expected mjlab motion format.
  Supports both Victor format (separate arrays) and standard format (x array).
  
  When object data is present, also extracts contact information for end effectors.
  
  Args:
    input_file: Path to input NPZ file
    output_file: Path to output NPZ file
    output_fps: Output frame rate
    device: Device to use for simulation
    repeat_last_frame: Number of times to repeat the last frame
    contact_method: Method to extract contacts - "mujoco" (preferred, uses physics) or "distance" (uses threshold)
    contact_threshold: Distance threshold in meters for "distance" method (default: 0.05m = 5cm)
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
    repeat_last_frame=repeat_last_frame,
  )

  robot: Entity = scene["robot"]
  box: Entity | None = scene.entities.get("box") if hasattr(scene, "entities") else None
  log_object = getattr(motion, "has_object", False) and (box is not None and not box.data.is_fixed_base)

  # Extract ALL bodies - mjlab expects complete body data for all robot bodies
  # The body_names in the config are used for tracking/observation, but motion files need all bodies
  print(f"Robot has {len(robot.body_names)} bodies: {robot.body_names}")

  # End effector names for contact detection
  eef_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
  
  # Get end effector body indices in robot.body_names
  eef_indices = []
  for eef_name in eef_names:
    if eef_name in robot.body_names:
      eef_indices.append(robot.body_names.index(eef_name))
    else:
      print(f"Warning: End effector '{eef_name}' not found in robot body names")
      eef_indices.append(-1)
  
  # Store object size for distance method (will be set during first frame if using distance method)
  object_size_global = None

  log: dict[str, Any] = {
    "fps": [int(round(output_fps))],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  if log_object:
    log["object_pos_w"] = []
    log["object_quat_w"] = []
    log["object_lin_vel_w"] = []
    log["object_ang_vel_w"] = []
    # Contact information
    log["contact_positions"] = []  # (T, num_eefs, 3)
    log["contact_indicators"] = []  # (T, num_eefs)

  frames_total = getattr(motion, "output_frames", motion.input_frames)
  pbar = tqdm(total=frames_total, desc="Converting", unit="frame", ncols=100)

  scene.reset()
  file_saved = False
  
  # Track previous object position from motion data for velocity computation
  prev_obj_pos_motion = None
  prev_obj_rot_motion = None
  
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

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = quat_apply_inverse(motion_base_rot, motion_base_ang_vel)
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, :] = motion_dof_pos
    joint_vel[:, :] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    if log_object:
      # Ensure object data exists
      if not hasattr(motion, "object_poss") or not hasattr(motion, "object_rots"):
        raise RuntimeError(
          "Object data not found: object_poss or object_rots missing. "
          "This should not happen if has_object is True."
        )
      if reset_flag:
        frame_idx = (frames_total - 1) if frames_total > 0 else 0
      else:
        frame_idx = motion.current_idx - 1
      # Ensure frame_idx is within bounds
      frame_idx = max(0, min(frame_idx, motion.object_poss.shape[0] - 1))
      
      # Get current object position from motion data (before adding env_origin)
      curr_obj_pos_motion = motion.object_poss[frame_idx : frame_idx + 1].clone()
      curr_obj_rot_motion = motion.object_rots[frame_idx : frame_idx + 1].clone()
      
      # Compute velocity from motion data positions
      if prev_obj_pos_motion is not None:
        # Linear velocity: (current_pos - prev_pos) / dt
        obj_lin_vel_motion = (curr_obj_pos_motion[0] - prev_obj_pos_motion) / sim_cfg.mujoco.timestep
        
        # Angular velocity: compute from quaternion difference
        q_prev = prev_obj_rot_motion
        q_curr = curr_obj_rot_motion[0]
        q_rel = quat_mul(q_curr, quat_conjugate(q_prev))
        obj_ang_vel_motion = axis_angle_from_quat(q_rel) / sim_cfg.mujoco.timestep
      else:
        # First frame: use zero velocity
        obj_lin_vel_motion = torch.zeros(3, device=device, dtype=torch.float32)
        obj_ang_vel_motion = torch.zeros(3, device=device, dtype=torch.float32)
      
      # Object position should be in world coordinates, add env_origin offset
      obj_pos_slice = curr_obj_pos_motion.clone()
      obj_pos_slice[:, :2] += scene.env_origins[:, :2]
      
      obj_pose = torch.cat([obj_pos_slice, curr_obj_rot_motion], dim=-1)
      box.write_root_link_pose_to_sim(obj_pose)
      
      # Store current position for next iteration
      prev_obj_pos_motion = curr_obj_pos_motion[0].clone()
      prev_obj_rot_motion = curr_obj_rot_motion[0].clone()

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    # Extract data for all joints
    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    
    # Extract data for ALL bodies (mjlab expects complete body data)
    body_pos_w = robot.data.body_link_pos_w[0, :].cpu().numpy().copy()
    log["body_pos_w"].append(body_pos_w)
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())

    if log_object:
      # Get object position from simulation (after setting pose and forward step)
      curr_obj_pos = box.data.body_link_pos_w[0, 0].cpu().numpy().copy()
      curr_obj_rot = box.data.body_link_quat_w[0, 0].cpu().numpy().copy()
      
      log["object_pos_w"].append(curr_obj_pos)
      log["object_quat_w"].append(curr_obj_rot)
      
      # Use velocities computed from motion data (computed earlier in the loop)
      obj_lin_vel = obj_lin_vel_motion.cpu().numpy()
      obj_ang_vel = obj_ang_vel_motion.cpu().numpy()
      
      log["object_lin_vel_w"].append(obj_lin_vel)
      log["object_ang_vel_w"].append(obj_ang_vel)
      

      # Use distance threshold method (simpler fallback)
      # Get end effector positions from body_pos_w
      eef_positions = np.array([
        body_pos_w[idx] if idx >= 0 else np.array([np.nan, np.nan, np.nan])
        for idx in eef_indices
      ])
      
      # Get object size from MuJoCo model (half-extents for box geometry)
      object_size = None
      if box is not None:
        # Try to find the object body - check common names
        object_body_names = ["largebox_link", "box", "object"]
        object_body_id = -1
        for body_name in object_body_names:
          object_body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
          if object_body_id >= 0:
            break
        
        if object_body_id >= 0:
          # Find geoms belonging to this body
          for geom_id in range(sim.mj_model.ngeom):
            if sim.mj_model.geom_bodyid[geom_id] == object_body_id:
              # Check if it's a box geometry
              if sim.mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                object_size = sim.mj_model.geom_size[geom_id].copy()  # Half-extents
                break
        
        # If still not found, try to get from box entity's first geom
        if object_size is None and hasattr(box, 'spec'):
          # Try to get geom size from the entity's spec
          try:
            geom_ids = box.indexing.geom_ids
            if len(geom_ids) > 0:
              geom_id = geom_ids[0].item() if hasattr(geom_ids[0], 'item') else int(geom_ids[0])
              if sim.mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                object_size = sim.mj_model.geom_size[geom_id].copy()
          except Exception:
            pass
      
      # Store object size globally (only print once)
      if object_size_global is None:
        object_size_global = object_size
        if object_size is not None:
          print(f"[Contact Detection] Object size (half-extents): {object_size}")
          print(f"[Contact Detection] Effective bounding box: ±{object_size + contact_threshold} (size + threshold)")
        else:
          print(f"[Contact Detection] Warning: Object size not found, using simple distance threshold ({contact_threshold}m) from center")
      
      contact_data = extract_contacts_from_distance(
        eef_positions,
        curr_obj_pos,
        object_rotation=curr_obj_rot,
        object_size=object_size,
        threshold=contact_threshold,
      )
      
      log["contact_positions"].append(contact_data["contact_positions"])
      log["contact_indicators"].append(contact_data["contact_indicators"])
      
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
  
  if log_object and len(log["object_pos_w"]) > 0:
    log["object_pos_w"] = np.stack(log["object_pos_w"], axis=0)
    log["object_quat_w"] = np.stack(log["object_quat_w"], axis=0)
    log["object_lin_vel_w"] = np.stack(log["object_lin_vel_w"], axis=0)
    log["object_ang_vel_w"] = np.stack(log["object_ang_vel_w"], axis=0)
    # Stack contact information
    if len(log["contact_positions"]) > 0:
      log["contact_positions"] = np.stack(log["contact_positions"], axis=0)  # (T, num_eefs, 3)
      log["contact_indicators"] = np.stack(log["contact_indicators"], axis=0)  # (T, num_eefs)
      
      # Print contact extraction summary
      print(f"\n{'='*60}")
      print(f"Contact Extraction Summary")
      print(f"{'='*60}")
      print(f"Method: {contact_method}")
      if contact_method == "distance":
        print(f"Distance threshold: {contact_threshold}m")
      
      print(f"\nContact data shape:")
      print(f"  Positions: {log['contact_positions'].shape} (T, num_eefs, 3)")
      print(f"  Indicators: {log['contact_indicators'].shape} (T, num_eefs)")
      
      num_frames = log["contact_indicators"].shape[0]
      num_eefs = log["contact_indicators"].shape[1]
      
      # Print diagnostic info for distance method if no contacts found
      if contact_method == "distance" and log["contact_indicators"].sum() == 0:
        print(f"\n⚠️  Diagnostic information (no contacts detected):")
        # Sample a few frames to check distances
        sample_frames = min(10, num_frames)
        if sample_frames > 0 and "body_pos_w" in log and "object_pos_w" in log:
          print(f"  Sample frame analysis (first {sample_frames} frames):")
          for frame_idx in range(sample_frames):
            for i, eef_name in enumerate(eef_names):
              if i < len(eef_indices) and eef_indices[i] >= 0:
                try:
                  eef_pos = log["body_pos_w"][frame_idx, eef_indices[i]]
                  obj_pos = log["object_pos_w"][frame_idx]
                  dist = np.linalg.norm(eef_pos - obj_pos)
                  print(f"    Frame {frame_idx}, {eef_name}: distance = {dist:.4f}m")
                except (IndexError, KeyError):
                  pass
      
      print(f"\nPer-end-effector contact statistics:")
      for i, eef_name in enumerate(eef_names):
        if i < num_eefs:
          contact_count = log["contact_indicators"][:, i].sum()
          contact_percentage = 100.0 * contact_count / num_frames if num_frames > 0 else 0.0
          print(f"  {eef_name}:")
          print(f"    Contact frames: {contact_count}/{num_frames} ({contact_percentage:.1f}%)")
          
          # Show contact positions for frames with contact
          contact_frames = np.where(log["contact_indicators"][:, i])[0]
          if len(contact_frames) > 0:
            # Show first few and last few contact positions
            num_samples = min(5, len(contact_frames))
            print(f"    Sample contact positions (first {num_samples}):")
            for frame_idx in contact_frames[:num_samples]:
              pos = log["contact_positions"][frame_idx, i]
              if not np.isnan(pos).any():
                print(f"      Frame {frame_idx}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            
            if len(contact_frames) > num_samples:
              print(f"    ... and {len(contact_frames) - num_samples} more contact frames")
      
      # Overall statistics
      total_contacts = log["contact_indicators"].sum()
      max_possible = num_frames * num_eefs
      overall_percentage = 100.0 * total_contacts / max_possible if max_possible > 0 else 0.0
      print(f"\nOverall statistics:")
      print(f"  Total contact detections: {total_contacts}/{max_possible} ({overall_percentage:.1f}%)")
      print(f"  Frames with at least one contact: {(log['contact_indicators'].sum(axis=1) > 0).sum()}/{num_frames}")
      print(f"  Frames with both contacts: {(log['contact_indicators'].sum(axis=1) == num_eefs).sum()}/{num_frames}")
      print(f"{'='*60}\n")

    
  np.savez(output_file, **log)  # type: ignore[arg-type]
  print(f"\nSaved mjlab motion to: {output_file}")
  print(f"Output contains {log['body_pos_w'].shape[0]} frames with {log['body_pos_w'].shape[1]} body links")
  print(f"  All robot bodies: {robot.body_names}")


def main(
  input_file: str,
  output_file: str,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  repeat_last_frame: int = 0,
  contact_method: str = "distance",
  contact_threshold: float = 0.05,
):
  """Convert trajectory NPZ file to mjlab motion format with contact extraction.
  
  Args:
    input_file: Path to input NPZ file
    output_file: Path to output NPZ file
    output_fps: Output frame rate
    device: Device to use for simulation
    repeat_last_frame: Number of times to repeat the last frame
    contact_method: Method to extract contacts - "mujoco" (preferred) or "distance"
    contact_threshold: Distance threshold in meters for "distance" method
  """
  convert(
    input_file=input_file,
    output_file=output_file,
    output_fps=output_fps,
    device=device,
    repeat_last_frame=repeat_last_frame,
    contact_method=contact_method,
    contact_threshold=contact_threshold,
  )


if __name__ == "__main__":
  tyro.cli(main)