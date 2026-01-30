import numpy as np

from sbto.utils.finite_diff import finite_diff_qpos_traj

def compute_obj_pos_error(obj_pos_traj, ref_obj_pos_traj):
    """
    Supports:
      obj_pos_traj:      [T, 3]   or [B, T, 3]
      ref_obj_pos_traj:  [T, 3]   or [B, T, 3]
    Returns:
      scalar (no batch) or [B] (batched)
    """

    # Ensure both arrays have batch dimension
    obj = np.asarray(obj_pos_traj)
    ref = np.asarray(ref_obj_pos_traj)
    # If no batch dimension, add one
    if obj.ndim == 2:   # [T, 3]
        obj = obj[None, ...]   # → [1, T, 3]
    if ref.ndim == 2:   # [T, 3]
        ref = ref[None, ...]   # → [1, T, 3]

    T = min(ref.shape[1], obj.shape[1])
    diff = obj[..., :T, :3] - ref[..., :T, :3]
    err = np.mean(np.linalg.norm(diff, axis=-1), axis=-1)

    # If original input was unbatched, return scalar
    if err.shape[0] == 1:
        return err[0]
    return err

def compute_obj_quat_error(obj_quat_traj, ref_obj_quat_traj):
    """
    Quaternion orientation error using:
        theta_t = arccos( 2 * <q_obj, q_ref>^2 - 1 )

    Supports:
      obj_quat_traj:      [T, 4]   or [B, T, 4]
      ref_obj_quat_traj:  [T, 4]   or [B, T, 4]

    Returns:
      scalar (no batch) or array [B]
    """

    obj = np.asarray(obj_quat_traj)
    ref = np.asarray(ref_obj_quat_traj)

    # Ensure batch dimension
    if obj.ndim == 2:   # [T, 4]
        obj = obj[None, ...]
    if ref.ndim == 2:   # [T, 4]
        ref = ref[None, ...]

    T = min(ref.shape[1], obj.shape[1])
    obj = obj[..., :T, -4:]
    ref = ref[..., :T, -4:]

    # Now obj, ref ∈ [B, T, 4] — broadcasting happens automatically if needed
    # Normalize quaternions to avoid numerical drift
    obj = obj / np.linalg.norm(obj, axis=-1, keepdims=True)
    ref = ref / np.linalg.norm(ref, axis=-1, keepdims=True)

    # Dot product per timestep: <q_obj, q_ref> ∈ [B, T]
    dot = np.sum(obj * ref, axis=-1)
    cos_theta = 2.0 * (dot ** 2) - 1.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # [B, T]
    err = np.mean(theta, axis=-1)

    # Return scalar when input had no batch dimension
    if err.shape[0] == 1:
        return err[0]

    return err

def compute_term_obj_pos_error(obj_pos_traj, ref_obj_pos_traj):
    last_pos = obj_pos_traj[None, -1, :3]
    last_pos_ref = ref_obj_pos_traj[None, -1, :3]
    return compute_obj_pos_error(last_pos, last_pos_ref)

def compute_term_obj_quat_error(obj_pos_traj, ref_obj_pos_traj):
    last_quat = obj_pos_traj[None, -1, -4:]
    last_quat_ref = ref_obj_pos_traj[None, -1, -4:]
    return compute_obj_quat_error(last_quat, last_quat_ref)

def compute_base_pos_error(base_pos_traj, ref_base_pos_traj):
    return compute_obj_pos_error(base_pos_traj, ref_base_pos_traj)

def compute_term_base_pos_error(base_pos_traj, ref_base_pos_traj):
    return compute_term_obj_pos_error(base_pos_traj, ref_base_pos_traj)

def compute_base_quat_error(base_quat_traj, ref_base_quat_traj):
    return compute_obj_quat_error(base_quat_traj, ref_base_quat_traj)

def compute_term_base_quat_error(base_quat_traj, ref_base_quat_traj):
    return compute_term_obj_quat_error(base_quat_traj, ref_base_quat_traj)

def compute_joint_pos_error(join_traj, ref_joint_traj):
    """
    Supports:
      join_traj:      [T, Nu]   or [B, T, Nu]
      ref_joint_traj: [T, Nu]   or [B, T, Nu]
    Returns:
      scalar (no batch) or [B] (batched)
    """

    # Ensure both arrays have batch dimension
    traj = np.asarray(join_traj)
    ref = np.asarray(ref_joint_traj)

    # If no batch dimension, add one
    if traj.ndim == 2:   # [T, 3]
        traj = traj[None, ...]   # → [1, T, 3]
    if ref.ndim == 2:   # [T, 3]
        ref = ref[None, ...]   # → [1, T, 3]

    T = min(ref.shape[1], traj.shape[1])
    diff = traj[..., :T, :] - ref[..., :T, :]
    err = np.mean(np.linalg.norm(diff, axis=-1), axis=-1)

    # If original input was unbatched, return scalar
    if err.shape[0] == 1:
        return err[0]

    return err

import numpy as np

def compute_acc_from_pos(qpos_traj, dt):
    """
    Compute second-order finite-difference acceleration from joint positions.

    qpos_traj: (T, nq) or (..., T, nq)
    dt: timestep
    """
    qpos = np.asarray(qpos_traj)

    # Second finite difference along time axis
    acc = (qpos[..., 2:, :] - 2 * qpos[..., 1:-1, :] + qpos[..., :-2, :]) / (dt ** 2)

    return acc


def compute_total_act_acc(qpos_traj, ref_qpos_traj, dt):
    """
    Compute smoothness metric for trajectory and reference.
    Returns mean L1 norm of joint accelerations per timestep.
    """

    traj = np.asarray(qpos_traj)
    ref  = np.asarray(ref_qpos_traj)

    traj_acc = compute_acc_from_pos(traj, dt)
    ref_acc  = compute_acc_from_pos(ref, dt)

    # L1 norm over joints, mean over time
    traj_acc_sum = np.mean(np.sum(np.abs(traj_acc), axis=-1))
    ref_acc_sum  = np.mean(np.sum(np.abs(ref_acc), axis=-1))

    return traj_acc_sum, ref_acc_sum