import numpy as np
import numpy.typing as npt
from numba import njit, prange

Array = npt.NDArray[np.float64]
    
@njit(parallel=True, fastmath=True, cache=True)
def quadratic_cost_nb(var, ref, weight):
    N, T, I = var.shape
    result = np.zeros(N, np.float64)
    for n in prange(N):
        for t in range(T):
            for i in range(I):
                diff = var[n, t, i] - ref[t, i]
                result[n] += weight[t, i] * diff * diff
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
    result = np.zeros(N, dtype=np.float64)
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

@njit(fastmath=True, inline="always")
def fast_acos(x):
    negate = 1.0 if x < 0.0 else 0.0
    x = abs(x)

    ret = -0.0187293
    ret = ret * x + 0.0742610
    ret = ret * x - 0.2121144
    ret = ret * x + 1.5707288

    ret = ret * np.sqrt(1.0 - x)

    # branchless reflection
    ret = ret - 2.0 * negate * ret

    return negate * np.pi + ret

@njit(parallel=True, fastmath=True, cache=True)
def quaternion_dist_logmap_nb(var, ref, weights):
    N, T, Q = var.shape
    QUAT_SIZE = 4
    Nquat = Q // QUAT_SIZE
    result = np.zeros(N, dtype=np.float64)

    for n in prange(N):
        for t in range(T):
            wgt = weights[t, 0]
            for iquat in range(Nquat):
                dot = 0.0
                for k in range(iquat * QUAT_SIZE, (iquat+1) * QUAT_SIZE):
                    dot += var[n, t, k] * ref[t, k]
                
                dot = abs(dot)
                dot = min(dot, 1.0)
                angle = 2.0 * fast_acos(dot)
                
                result[n] += wgt * angle * angle

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
    result = np.zeros(N, dtype=np.float64)

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

COST_FUNS = [quadratic_cost_nb, quaternion_dist_logmap_nb, hamming_dist_nb]

def get_cost_fn_idx(f):
    if f in COST_FUNS:
        return COST_FUNS.index(f)
    else:
        return None

@njit(parallel=True, fastmath=True, cache=True)
def compute_total_cost(
    x_traj,
    u_traj,
    obs_traj, 
    var_types,
    f_ids,
    idxs,
    refs,
    weights
    ):

    B = x_traj.shape[0]
    N_terms = len(var_types)
    total = np.zeros(B)
    for t in prange(N_terms):
        var_type = var_types[t]
        idx = idxs[t]
        ref = refs[t]
        w = weights[t]

        if var_type == 0:
            var = x_traj
        elif var_type == 1:
            var = u_traj
        else:
            var = obs_traj

        # dispatch function by integer
        if f_ids[t] == 0:
            total += quadratic_cost_nb(var[..., idx], ref, w)
        elif f_ids[t] == 1:
            total += quaternion_dist_logmap_nb(var[..., idx], ref, w)
        elif f_ids[t] == 2:
            total += hamming_dist_nb(var[..., idx], ref, w)

    return total