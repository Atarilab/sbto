import numpy as np
import numpy.typing as npt
from numba import njit, prange

Array = npt.NDArray[np.float64]

def quadratic_cost(var: Array, ref: Array, weights: Array) -> float:
    return np.sum(weights[None, ...] * (var - ref[None, ...]) ** 2, axis=(-1, -2))

def contact_cost(cnt_status_rollout, cnt_plan, weights) -> float:
    cnt_status_rollout[cnt_status_rollout > 1] = 1
    return np.sum(weights[None, ...] * np.float32(cnt_status_rollout != cnt_plan[None, ...]), axis=(-1, -2))

def quat_dist(var, ref, weights) -> float:
    return np.sum(weights[:, 0] * (1.0 - np.square(np.sum(var * ref[None, ...], axis=-1))), axis=(-1))
    
@njit(parallel=True, fastmath=True, cache=True)
def quadratic_cost_nb(var, ref, weight):
    N, T, I = var.shape
    result = np.empty(N, np.float64)
    for n in prange(N):
        total = 0.0
        for t in range(T):
            for i in range(I):
                diff = var[n, t, i] - ref[t, i]
                total += weight[t, i] * diff * diff
        result[n] = total
    return result
    
@njit(parallel=True, fastmath=True, cache=True)
def quadratic_cost_nb_decay(var, ref, weight, decay):
    N, T, I = var.shape
    result = np.empty(N, np.float64)
    for n in prange(N):
        total = 0.0
        for t in range(T):
            for i in range(I):
                diff = var[n, t, i] - ref[t, i]
                total += weight[t, i] * diff * diff * decay[t]
        result[n] = total
    return result

@njit(parallel=True, fastmath=True, cache=True)
def quaternion_dist_nb(var, ref, weights):
    """
    Numba-accelerated version of quaternion distance cost.
    Shapes:
        var: (N, T, Nquat*4)       # quaternion rollout
        ref: (T, Nquat*4)          # reference quaternion trajectory
        weights: (T, 1) or (T,)  # scalar weights per timestep
    Returns:
        cost: (N,)
    """
    N, T, Q = var.shape
    result = np.empty(N, dtype=np.float64)
    QUAT_SIZE = 4
    Nquat = Q // QUAT_SIZE

    for n in prange(N):
        total = 0.0
        for t in range(T):
            for iquat in range(Nquat):
                dot = 0.0
                for k in range(iquat * QUAT_SIZE, (iquat+1) * QUAT_SIZE):
                    dot += var[n, t, k] * ref[t, k]
                diff = 1.0 - dot * dot
                total += weights[t, 0] * diff
        result[n] = total
    return result

@njit(parallel=True, fastmath=True, cache=True)
def quaternion_dist_nb_decay(var, ref, weights, decay):
    """
    Numba-accelerated version of quaternion distance cost.
    Shapes:
        var: (N, T, Nquat*4)       # quaternion rollout
        ref: (T, Nquat*4)          # reference quaternion trajectory
        weights: (T, 1) or (T,)  # scalar weights per timestep
        decay              : (T) array of float32 weights
    Returns:
        cost: (N,)
    """
    N, T, Q = var.shape
    result = np.empty(N, dtype=np.float64)
    QUAT_SIZE = 4
    Nquat = Q // QUAT_SIZE

    for n in prange(N):
        total = 0.0
        for t in range(T):
            for iquat in range(Nquat):
                dot = 0.0
                for k in range(iquat * QUAT_SIZE, (iquat+1) * QUAT_SIZE):
                    dot += var[n, t, k] * ref[t, k]
                diff = 1.0 - dot * dot
                total += weights[t, 0] * diff * decay[t]
        result[n] = total
    return result

@njit(parallel=True, fastmath=True, cache=True)
def hamming_dist_nb(cnt_rollout, cnt_plan, weights):
    """
    Efficient Hamming-distance-based contact cost.
    Args:
        cnt_rollout : (N, T, C) array, contact states (0, 1, maybe >1)
        cnt_plan           : (T, C) array, desired contact pattern (0 or 1)
        weights            : (T, C) array of float32 weights

    Returns:
        cost : (N,) array of float32
    """
    N, T, C = cnt_rollout.shape
    result = np.empty(N, dtype=np.float64)

    for n in prange(N):
        total = 0.0
        for t in range(T):
            for c in range(C):
                s = cnt_rollout[n, t, c]
                # Clamp contact status > 1 to 1, cast to integer
                if s > 1:
                    s = 1
                # XOR trick for mismatch detection (works with ints 0/1)
                diff = int(s) ^ int(cnt_plan[t, c])
                total += weights[t, c] * diff
        result[n] = total
    return result


@njit(parallel=True, fastmath=True, cache=True)
def hamming_dist_nb_decay(cnt_rollout, cnt_plan, weights, decay):
    """
    Efficient Hamming-distance-based contact cost.
    Args:
        cnt_rollout : (N, T, C) array, contact states (0, 1, maybe >1)
        cnt_plan           : (T, C) array, desired contact pattern (0 or 1)
        weights            : (T, C) array of float32 weights
        decay              : (T) array of float32 weights
    Returns:
        cost : (N,) array of float32
    """
    N, T, C = cnt_rollout.shape
    result = np.empty(N, dtype=np.float64)

    for n in prange(N):
        total = 0.0
        for t in range(T):
            for c in range(C):
                s = cnt_rollout[n, t, c]
                # Clamp contact status > 1 to 1, cast to integer
                if s > 1:
                    s = 1
                # XOR trick for mismatch detection (works with ints 0/1)
                diff = int(s) ^ int(cnt_plan[t, c])
                total += weights[t, c] * diff * decay[t]
        result[n] = total
    return result
