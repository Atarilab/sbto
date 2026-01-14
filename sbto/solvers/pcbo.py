import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass
import numba as nb

from sbto.solvers.cbo import ConfigCBO, CBO

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

KERNEL_TYPE = {
    "gaussian" : 0,
    "laplace" : 1,
    "cosine" : 2,
    "rational" : 3,
    "student" : 4,
    "cauchy" : 5,
    "confidence" : 6,
    "epa" : 7,
}


@nb.njit(fastmath=True)
def kernel_value(xi, cj, kappa, kernel_type):
    diff_l2_sq = 0.0
    diff_l1 = 0.0
    dot = 0.0
    nx_sq = 0.0
    ny_sq = 0.0

    D = xi.shape[0]

    for d in range(D):
        diff = xi[d] - cj[d]
        diff_l2_sq += diff * diff
        diff_l1 += np.abs(diff)

        dot += xi[d] * cj[d]
        nx_sq += xi[d] * xi[d]
        ny_sq += cj[d] * cj[d]

    # --- Dimension Scaling Logic ---
    # RMSD: Standardizes L2 distance across dimensions
    scaled_diff_l2_sq = diff_l2_sq / D 
    # Average L1: Standardizes Manhattan distance
    scaled_diff_l1 = diff_l1 / D 
    
    # Cosine distance is inherently normalized by magnitude, 
    # but we ensure numerical stability.
    cos_dist = 1.0 - (dot / (np.sqrt(nx_sq * ny_sq) + 1e-10))

    if kernel_type == 0:
        # Gaussian (using scaled L2 squared)
        return np.exp(-0.5 * scaled_diff_l2_sq / kappa)

    elif kernel_type == 1:
        # Laplace (using scaled L1)
        return np.exp(-scaled_diff_l1 / kappa)

    elif kernel_type == 2:
        # Cosine similarity
        return np.exp(-cos_dist / kappa)

    elif kernel_type == 3:
        # Rational quadratic (using scaled L2 squared)
        return 1.0 / (1.0 + scaled_diff_l2_sq / (2.0 * kappa))

    elif kernel_type == 4:
        # Student-t (using scaled L2 squared)
        return 1.0 / (1.0 + scaled_diff_l2_sq / kappa)

    elif kernel_type == 5:
        # Cauchy (using RMSD)
        rmsd = np.sqrt(scaled_diff_l2_sq)
        return 1.0 / (1.0 + rmsd / kappa)

    elif kernel_type == 6:
        # Indicator kernel (using RMSD)
        rmsd = np.sqrt(scaled_diff_l2_sq)
        return 1.0 if rmsd <= kappa else 0.0

    else:
        return 1.0


@nb.njit(parallel=True, fastmath=True)
def compute_consensus_numba(samples, costs, cmin, kappa, beta, kernel_type):
    N, D = samples.shape

    consensus = np.zeros((N, D))
    K = np.zeros((N, N))
    w = np.exp(-(costs - cmin) * beta)

    # 2. Compute Kernel Matrix using Symmetry
    # We use a nested loop structure that fills both K[i,j] and K[j,i]
    for i in nb.prange(N):
        # Set diagonal
        K[i, i] = kernel_value(samples[i], samples[i], kappa, kernel_type)
        for j in range(i + 1, N):
            val = kernel_value(samples[i], samples[j], kappa, kernel_type)
            K[i, j] = val
            K[j, i] = val

    # 2. Compute Consensus (Same as before)
    for i in nb.prange(N):
        denom = 0.0
        num = np.zeros(D)
        for j in range(N):
            wij = K[i, j] * w[j]
            denom += wij
            num += wij * samples[j]

        if denom > 1e-10:
            consensus[i] = num / denom
        else:
            consensus[i] = samples[i]

    return consensus, K

@dataclass
class ConfigPCBO(ConfigCBO):
    """
    kernel: kernel name to use
    kappa: witdth of the gaussian kernel
    """
    kernel: str = "gaussian"
    kappa_init: float = 0.1
    kappa_final: float = 0.01
    annealing_strategy: str = "exp"

class PCBO(CBO):
    def __init__(self, D, cfg):
        super().__init__(D, cfg)
        self._consensus = np.zeros((cfg.N_samples, D))
        self._kernel_type = KERNEL_TYPE[self.cfg.kernel]
        self.Nu = 29
        self.kappa = self.cfg.kappa_init
        self.current_iter = 0

    def _update_kappa(self, current_iter: int, max_iter: int) -> float:
        """Anneals the kernel bandwidth kappa."""
        if max_iter <= 0: 
            return self.cfg.kappa_init

        if self.cfg.annealing_strategy == "exp":
            # Exponential decay: k = k_init * (k_final/k_init)^(t/T)
            decay = np.log(self.cfg.kappa_final / self.cfg.kappa_init)
            alpha = decay / max_iter
            self.kappa = self.cfg.kappa_init * np.exp(alpha * current_iter)
            
        elif self.cfg.annealing_strategy == "linear":
            # Linear decay
            progress = current_iter / max_iter
            self.kappa = self.cfg.kappa_init + progress * (self.cfg.kappa_final - self.cfg.kappa_init)
            
        else:
            self.kappa = self.cfg.kappa_init

        # Clamp to ensure we don't go below final or negative
        return max(self.kappa, self.cfg.kappa_final)
    
    def opt_first_dim(self, n_dim: int = -1):
        self.current_iter = 0
        self.kappa = self.cfg.kappa_init
        super().opt_first_dim(n_dim)

    def update_mean(self, samples: Array, costs: Array) -> Tuple[int, float]:
        argmin = costs.argmin()
        cmin = costs[argmin]
        kappa_curr = self._update_kappa(self.current_iter, 100)
        self.current_iter += 1

        consensus, K = compute_consensus_numba(
            samples[:, :self.n_dim],
            costs,
            cmin,
            kappa_curr,
            self.cfg.beta,
            self._kernel_type
        )
        alpha = 0.3
        self._consensus[:, :self.n_dim] += alpha * (consensus - self._consensus[:, :self.n_dim])

        self.logs["k"] = kappa_curr
        self.logs["mean_K"] = np.mean(K)
        # print(np.mean(K))
        # print(np.max(K - np.eye(len(samples))))
        # print(np.min(K))
        return argmin, cmin