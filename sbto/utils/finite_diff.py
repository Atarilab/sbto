import numpy as np

def finite_diff_qpos_traj(qpos: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute joint velocities using second-order finite differencing.

    Args:
        qpos (np.ndarray): shape [T, nq] positions
        dt (float): timestep

    Returns:
        qvel (np.ndarray): shape [T, nq] velocities
    """
    T, nq = qpos.shape
    qvel = np.zeros_like(qpos)

    # Interior: central difference (2nd order)
    qvel[1:-1] = (qpos[2:] - qpos[:-2]) / (2 * dt)

    # Forward difference at t=0 (2nd order accurate)
    # v0 = (-3x0 + 4x1 - x2) / (2dt)
    if T >= 3:
        qvel[0] = (-3*qpos[0] + 4*qpos[1] - qpos[2]) / (2 * dt)
    elif T == 2:
        qvel[0] = (qpos[1] - qpos[0]) / dt

    # Backward difference at t=T-1 (2nd order accurate)
    # vN = (3xN - 4x(N-1) + x(N-2)) / (2dt)
    if T >= 3:
        qvel[-1] = (3*qpos[-1] - 4*qpos[-2] + qpos[-3]) / (2 * dt)
    elif T == 2:
        qvel[-1] = (qpos[-1] - qpos[-2]) / dt

    return qvel

def finite_diff_quat(q1, q2, dt):
    """
    Compute angular velocity (3D) from two quaternions using:
        ω = 2/dt * imag(q1* conjugate ⊗ q2)

    Args:
        q1, q2 : quaternions [4] as w,x,y,z
        dt     : timestep

    Returns:
        omega : [3] angular velocity vector
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    wx = w1*x2 - x1*w2 - y1*z2 + z1*y2
    wy = w1*y2 + x1*z2 - y1*w2 - z1*x2
    wz = w1*z2 - x1*y2 + y1*x2 - z1*w2

    return (2.0 / dt) * np.array([wx, wy, wz])

def finite_diff_quat_traj(qpos: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute angular velocity trajectory from quaternion trajectory
    using second-order finite differencing.

    Args:
        qpos : [T, 4] quaternion trajectory (wxyz)
        dt   : timestep

    Returns:
        omega : [T, 3] angular velocity
    """
    T = qpos.shape[0]
    omega = np.zeros((T, 3))

    # Normalize to be safe
    q_norm = qpos / np.linalg.norm(qpos, axis=1, keepdims=True)

    # Interior: central difference
    for t in range(1, T - 1):
        omega[t] = finite_diff_quat(q_norm[t - 1], q_norm[t + 1], 2 * dt)

    # Forward difference at t=0 (2nd-order)
    if T >= 3:
        omega[0] = finite_diff_quat(q_norm[0], q_norm[2], 2 * dt)
    elif T == 2:
        omega[0] = finite_diff_quat(q_norm[0], q_norm[1], dt)

    # Backward difference at t=T-1 (2nd-order)
    if T >= 3:
        omega[-1] = finite_diff_quat(q_norm[-3], q_norm[-1], 2 * dt)
    elif T == 2:
        omega[-1] = finite_diff_quat(q_norm[-2], q_norm[-1], dt)

    return omega
