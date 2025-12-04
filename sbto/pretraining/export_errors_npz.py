
import numpy as np
from pathlib import Path
import torch
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    quat_conjugate as torch_quat_conjugate,
    quat_mul as torch_quat_mul,
    matrix_from_quat as torch_matrix_from_quat,
    subtract_frame_transforms,
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
    c = data.get("c", None)
    joint_pos = data["joint_pos"].astype(np.float32)          # (N, T, 29)
    joint_vel = data["joint_vel"].astype(np.float32)          # (N, T, 29)
    anchor_pos_b = data["anchor_pos_b"].astype(np.float32)    # (N, T, 3)
    anchor_ori_b = data["anchor_ori_b"].astype(np.float32)    # (N, T, 4)
    base_lin_vel = data["base_lin_vel"].astype(np.float32)    # (N, T, 3)
    base_ang_vel = data["base_ang_vel"].astype(np.float32)    # (N, T, 3)
    object_pos_w = data["object_pos_w"].astype(np.float32)    # (N, T, 3)
    object_quat_w = data["object_quat_w"].astype(np.float32)  # (N, T, 4)
    c = data["c"].astype(np.float32)                          # (N,)
    robot_body_pos_w = data["robot_body_pos_w"].astype(np.float32)    # (N, T, B, 3)
    robot_body_quat_w = data["robot_body_quat_w"].astype(np.float32)  # (N, T, B, 4)

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

    object_position_error = object_pos_w - ref_obj_pos[None, :, :]  # (N, T, 3)

    #  6D error 
    object_orientation_error = orientation_error_6d_from_mjlab(
        object_quat_w,        # (N, T, 4)
        ref_obj_quat,         # (T, 4)
    )  
    
    # body_pos / body_ori in anchor frame
    N, T, B, _ = robot_body_pos_w.shape
    total = N * T
    # Flatten
    anchor_pos_flat  = anchor_pos_b.reshape(total, 3)        # (total, 3)
    anchor_quat_flat = anchor_ori_b.reshape(total, 4)        # (total, 4)
    body_pos_w_flat  = robot_body_pos_w.reshape(total, B, 3) # (total, B, 3)
    body_quat_w_flat = robot_body_quat_w.reshape(total, B, 4)# (total, B, 4)

    anchor_pos_t  = torch.from_numpy(anchor_pos_flat)        # (total, 3)
    anchor_quat_t = torch.from_numpy(anchor_quat_flat)       # (total, 4)
    body_pos_w_t  = torch.from_numpy(body_pos_w_flat)        # (total, B, 3)
    body_quat_w_t = torch.from_numpy(body_quat_w_flat)       # (total, B, 4)

    # Broadcast 
    t01 = anchor_pos_t[:, None, :].expand(total, B, 3)
    q01 = anchor_quat_t[:, None, :].expand(total, B, 4)

    # Subtract frames: world to anchor frame 
    t12, q12 = subtract_frame_transforms(
        t01,          # anchor in world
        q01,
        body_pos_w_t, # body in world
        body_quat_w_t,
    )

    # Convert quaternion -> 6D orientation: first 2 columns of rotation matrix
    R = torch_matrix_from_quat(q12)         # (total, B, 3, 3)
    first_two_cols = R[..., :2]            # (total, B, 3, 2)

 
    body_pos = t12.reshape(N, T, B * 3).cpu().numpy()        
    body_ori = first_two_cols.reshape(N, T, B * 6).cpu().numpy()  
   

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
    policy_obs = np.concatenate(feat_list, axis=-1)  

    out = dict(data)
    out["error_anchor_pos"]       = error_anchor_pos
    out["error_anchor_b"]         = error_anchor_b
    out["joint_pos_rel"]          = joint_pos_rel
    out["actions"]                = actions
    out["object_pos_b"]           = object_pos_b
    out["object_position_error"]  = object_position_error
    out["object_orientation_error"] = object_orientation_error
    out["policy_obs"]             = policy_obs
    out["body_pos"]               = body_pos
    out["body_ori"]               = body_ori
       
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(f"Saved pretraining NPZ to: {out_path}")


def main():
    in_path = "sbto/data/rollout_time_x_u_obs_traj.npz"
    out_path = "sbto/data/pretraining_actor_input.npz"
    build_pretraining_npz(in_path, out_path)


if __name__ == "__main__":
    main()