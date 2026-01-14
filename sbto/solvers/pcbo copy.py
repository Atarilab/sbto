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
    diff_l2 = 0.0
    diff_l1 = 0.0
    dot = 0.0
    nx = 0.0
    ny = 0.0

    D = xi.shape[0]

    for d in range(D):
        diff = xi[d] - cj[d]
        diff_l2 += diff * diff
        diff_l1 += np.abs(diff)

        dot += xi[d] * cj[d]
        nx += xi[d] * xi[d]
        ny += cj[d] * cj[d]

    dist_l2 = np.sqrt(diff_l2)

    if kernel_type == 0:
        # Gaussian
        return np.exp(-0.5 * diff_l2 / (kappa))

    elif kernel_type == 1:
        # Laplace L1
        return np.exp(-diff_l1 / kappa)

    elif kernel_type == 2:
        # Cosine similarity
        cos_sim = dot / (np.sqrt(nx * ny) + 1e-12)
        return np.exp(-(1.0 - cos_sim) / kappa)

    elif kernel_type == 3:
        # Rational quadratic
        return 1.0 / (1.0 + diff_l2 / (2.0 * kappa))

    elif kernel_type == 4:
        # Student-t
        return 1.0 / (1.0 + diff_l2 / kappa)

    elif kernel_type == 5:
        # Cauchy
        return 1.0 / (1.0 + dist_l2 / kappa)

    elif kernel_type == 6:
        # Indicator kernel
        return 1.0 if  np.sqrt(diff_l2 / D) <= kappa else 0.0

    else:
        return 1.0
@nb.njit(parallel=True, fastmath=True)
def compute_consensus_numba(samples, costs, cmin, kappa, beta, kernel_type):
    N, D = samples.shape
    consensus = np.zeros((N, D))
    
    # Pre-calculate weights (Gibbs distribution)
    w = np.exp(-(costs - cmin) * beta)

    # We iterate over each particle i to find its local consensus
    for i in nb.prange(N):
        denom = 0.0
        num = np.zeros(D)
        xi = samples[i]

        for j in range(N):
            # Localize the consensus using the kernel
            kij = kernel_value(xi, samples[j], kappa, kernel_type)
            wij = kij * w[j]
            denom += wij

            for d in range(D):
                num[d] += wij * samples[j, d]

        if denom > 1e-15:
            for d in range(D):
                consensus[i, d] = num[d] / denom
        else:
            # Fallback: if isolated, consensus is itself (no movement)
            consensus[i, :] = xi

    return consensus

@dataclass
class ConfigPCBO(ConfigCBO):
    """
    kernel: kernel name to use
    kappa: witdth of the gaussian kernel
    """
    kernel: str = "gaussian"
    kappa_init: float = 0.1
    kappa_final: float = 0.01

class PCBO(CBO):
    def __init__(self, D, cfg):
        super().__init__(D, cfg)
        self._consensus = np.zeros((cfg.N_samples, D))
        self._kernel_type = KERNEL_TYPE[self.cfg.kernel]
        self.Nu = 29
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
    
    def update_mean(self, samples: Array, costs: Array) -> Tuple[int, float]:
        argmin = costs.argmin()
        cmin = costs[argmin]
        kappa_curr = self._update_kappa(self.current_iter, 50)
        self.current_iter += 1
        
        self._consensus[:, self.n_dim-self.Nu:self.n_dim], K = compute_consensus_numba(
            samples[:, self.n_dim-self.Nu:self.n_dim],
            costs,
            cmin,
            kappa_curr,
            self.cfg.beta,
            self._kernel_type
        )
        self.logs["mean_K"] = np.mean(K)
        # print(np.mean(K))
        # print(np.max(K - np.eye(len(samples))))
        # print(np.min(K))
        return argmin, cmin