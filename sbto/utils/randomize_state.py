import numpy as np
from typing import Tuple, Union

def randomize_joint_pos(
    mj_model,
    N: int,
    x_0: np.ndarray,
    scale_q: Union[np.ndarray, float] = 0.1,
    scale_v: Union[np.ndarray, float] = 0.1,
) -> np.ndarray:
    """
    Randomizes joint positions and velocities around a nominal state x_0.

    Args:
        mj_model: MuJoCo model object with attributes nq (num positions) and nv (num velocities).
        N: Number of samples to generate.
        x_0: (nq + nv,) base state [q, v].
        scale_q: Position noise scale (float or array of shape (nq,)).
        scale_v: Velocity noise scale (float or array of shape (nv,)).

    Returns:
        x_rand: (N, nq + nv) array of randomized states.
    """
    nq, nv = mj_model.nq, mj_model.nv
    assert x_0.shape == (nq + nv,), f"x_0 must have shape ({nq + nv},), got {x_0.shape}"

    # Broadcastable scales
    scale_q = np.asarray(scale_q, dtype=float)
    scale_v = np.asarray(scale_v, dtype=float)

    # Split base state
    q_0, v_0 = np.split(x_0, [nq])

    # Tile base state
    q = np.tile(q_0, (N, 1))
    v = np.tile(v_0, (N, 1))

    # Add Gaussian noise
    q += np.random.randn(N, nq) * scale_q
    v += np.random.randn(N, nv) * scale_v

    # Combine back
    x_rand = np.concatenate([q, v], axis=-1)
    return x_rand

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
) -> np.ndarray:
    """
    Randomize the position (x, y) and yaw rotation of an object N times.

    Args:
        N: number of samples to generate
        obj_pos_quat: (7,) array-like base [x, y, z, qw, qx, qy, qz]
        x_range: tuple for x-offset range (in meters)
        y_range: tuple for y-offset range (in meters)
        w_range: tuple for yaw angle range (in radians)

    Returns:
        obj_pos_quat_randomized: (N, 7) array of randomized poses
    """
    obj_pos_quat = np.copy(np.asarray(obj_pos_quat))
    assert obj_pos_quat.shape == (7,), "obj_pos_quat must be a 7-element array (pos + quat)."

    # Split position and quaternion
    xyz = obj_pos_quat[:3]
    quat = obj_pos_quat[3:7]

    # Random offsets
    x_offsets = np.random.uniform(*x_range, N)
    y_offsets = np.random.uniform(*y_range, N)
    z_offsets = np.random.uniform(*z_range, N)
    w_offsets = np.random.uniform(*w_range, N)  # yaw rotations

    # Base positions
    xyz_rand = np.tile(xyz, (N, 1))
    xyz_rand[:, 0] += x_offsets
    xyz_rand[:, 1] += y_offsets
    xyz_rand[:, 2] += z_offsets

    # Compute yaw rotation quaternion for each sample
    qw = np.cos(w_offsets / 2.0)
    qz = np.sin(w_offsets / 2.0)
    yaw_quats = np.stack([qw, np.zeros_like(qw), np.zeros_like(qw), qz], axis=1)

    # Combine with base orientation
    quat_rand = np.zeros_like(yaw_quats)
    for i in range(N):
        quat_rand[i] = quat_multiply(yaw_quats[i], quat)

    # Combine pos + quat
    obj_pos_quat_rand = np.hstack([xyz_rand, quat_rand])
    return obj_pos_quat_rand

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