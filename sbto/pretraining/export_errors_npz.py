import numpy as np
from pathlib import Path
import torch
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    quat_conjugate as torch_quat_conjugate,
    quat_mul as torch_quat_mul,
    matrix_from_quat as torch_matrix_from_quat,
)


# Constants 

DEFAULT_JOINT_POS = np.array([
    -0.312, 0.0,    0.0,   0.669, -0.363, 0.0,
    -0.312, 0.0,    0.0,   0.669, -0.363, 0.0,
     0.0,   0.0,    0.0,
     0.2,   0.2,    0.0,   0.6,   0.0,   0.0,  -1.433,
     0.2,  -0.2,    0.0,   0.6,   0.0,   0.0,   1.433
], dtype=np.float32)  

ACTION_SCALE = np.array([
    0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
    0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
    0.548, 0.439, 0.439,
    0.439, 0.439, 0.439, 0.439, 0.439,
    0.075, 0.075,
    0.439, 0.439, 0.439, 0.439, 0.439,
    0.075, 0.075
], dtype=np.float32) 


# Orientation error
def orientation_error_6d_from_mjlab(
    q_current_np: np.ndarray,
    q_ref_np: np.ndarray,
) -> np.ndarray:
    """
    Compute 6D orientation error using the same quaternion math
    as mjlab 

    Returns:
        (N, T, 6): first two columns of rotation matrix of the relative rotation
    """
 
    q_current = torch.from_numpy(q_current_np.astype(np.float32))  

    if q_ref_np.ndim == 2:

        q_ref = torch.from_numpy(q_ref_np.astype(np.float32)).unsqueeze(0) 
        q_ref = q_ref.expand(q_current.shape[0], -1, -1) 
    else:

        q_ref = torch.from_numpy(q_ref_np.astype(np.float32))

    # Relative rotation: q_err = q_current * conj(q_ref)
    q_ref_conj = torch_quat_conjugate(q_ref)       # (N, T, 4)
    q_err = torch_quat_mul(q_current, q_ref_conj)  # (N, T, 4)

    # Rotation matrix 
    R = torch_matrix_from_quat(q_err)              # (N, T, 3, 3)
    first_two_cols = R[..., :, :2]                 # (N, T, 3, 2)
    N, T = q_current.shape[0], q_current.shape[1]
    err6d = first_two_cols.reshape(N, T, 6)        # (N, T, 6)

    return err6d.numpy()

# Main builder

def build_pretraining_npz(in_path: str, out_path: str) -> None:
    data = np.load(in_path)
    x = data.get("x", None)
    u = data.get("u", None)
    o = data.get("o", None)
    c = data.get("c", None)

    # Grab the fields from your current rollout npz
    joint_pos = data["joint_pos"].astype(np.float32)          # (N, T, 29)
    joint_vel = data["joint_vel"].astype(np.float32)          # (N, T, 29)
    anchor_pos_b = data["anchor_pos_b"].astype(np.float32)    # (N, T, 3)
    anchor_ori_b = data["anchor_ori_b"].astype(np.float32)    # (N, T, 4)
    base_lin_vel = data["base_lin_vel"].astype(np.float32)    # (N, T, 3)
    base_ang_vel = data["base_ang_vel"].astype(np.float32)    # (N, T, 3)
    object_pos_w = data["object_pos_w"].astype(np.float32)    # (N, T, 3)
    object_quat_w = data["object_quat_w"].astype(np.float32)  # (N, T, 4)
    c = data["c"].astype(np.float32)                          # (N,)

    N, T, A = joint_pos.shape
    assert A == 29, f"Expected 29 joints, got {A}"

    # reference trajectory index 
    ref_idx = int(np.argmin(c))
    print(f" Using trajectory {ref_idx} as reference")

    # reference anchor & object 
    ref_anchor_pos = anchor_pos_b[ref_idx]      # (T, 3)
    ref_anchor_quat = anchor_ori_b[ref_idx]     # (T, 4)
    ref_obj_pos = object_pos_w[ref_idx]         # (T, 3)
    ref_obj_quat = object_quat_w[ref_idx]       # (T, 4)


    error_anchor_pos = anchor_pos_b - ref_anchor_pos[None, :, :]  # (N, T, 3)

    error_anchor_b = orientation_error_6d_from_mjlab(
        anchor_ori_b,          
        ref_anchor_quat,       
    ) 

    joint_pos_rel = joint_pos - DEFAULT_JOINT_POS[None, None, :]  # (N, T, 29)

    # actions: previous joint target / ACTION_SCALE
    actions = np.zeros_like(joint_pos, dtype=np.float32)        
    actions[:, 1:, :] = joint_pos[:, :-1, :] / ACTION_SCALE[None, None, :]

    # object position in "robot frame"
    object_pos_b = object_pos_w - anchor_pos_b                   # (N, T, 3)

    # current object pos - ref object pos
    object_position_error = object_pos_w - ref_obj_pos[None, :, :]  # (N, T, 3)

    #  6D error between current object and ref object
    object_orientation_error = orientation_error_6d_from_mjlab(
        object_quat_w,        # (N, T, 4)
        ref_obj_quat,         # (T, 4)
    )  # -> (N, T, 6)

    feat_list = [
        joint_pos,                  # 29
        joint_vel,                  # 29
        error_anchor_pos,           # 3
        error_anchor_b,             # 6
        base_lin_vel,               # 3
        base_ang_vel,               # 3
        joint_pos_rel,              # 29
        actions,                    # 29
        object_pos_b,               # 3
        object_position_error,      # 3
        object_orientation_error,   # 6
    ]
    policy_obs = np.concatenate(feat_list, axis=-1)  # (N, T, 172)


    
    out = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "error_anchor_pos": error_anchor_pos,
        "error_anchor_b": error_anchor_b,
        "base_lin_vel": base_lin_vel,
        "base_ang_vel": base_ang_vel,
        "joint_pos_rel": joint_pos_rel,
        "actions": actions,
        "object_pos_b": object_pos_b,
        "object_position_error": object_position_error,
        "object_orientation_error": object_orientation_error,
        "policy_obs": policy_obs,
    }
    if x is not None:
      out["x"] = x
    if u is not None:
        out["u"] = u
    if o is not None:
        out["o"] = o
    if c is not None:
       out["c"] = c
       
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"Saved pretraining NPZ to: {out_path}")


def main():
    in_path = "sbto/data/rollout_time_x_u_obs_traj.npz"
    out_path = "sbto/data/pretraining_actor_input.npz"
    build_pretraining_npz(in_path, out_path)


if __name__ == "__main__":
    main()