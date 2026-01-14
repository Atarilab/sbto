import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass
import numba as nb

from sbto.solvers.cbo import ConfigCBO, CBO, SolverState

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
import numba as nb
import numpy as np


@nb.njit(parallel=True, fastmath=True)
def compute_consensus_clustered_numba(
    samples,        # (N, D)
    costs,          # (N,)
    centers,        # (Nc, D)
    p,              # (N, Nc)
    cmin,
    kappa,
    beta,
    alpha,
):
    N, D = samples.shape
    Nc = centers.shape[0]

    # --- Precompute Boltzmann weights ---
    w = np.empty(N)
    for i in range(N):
        w[i] = np.exp(-(costs[i] - cmin) * beta)

    # --- Step 1: Update probabilities ---
    p_new = np.zeros((N, Nc))

    for i in nb.prange(N):
        # most likely cluster
        pmax = 0.0
        for j in range(Nc):
            if p[i, j] > pmax:
                pmax = p[i, j]

        # discounted + kernel-weighted probabilities
        norm = 0.0
        xi = samples[i]

        for j in range(Nc):
            # rij = (pij / pmax)^alpha
            if pmax > 0.0:
                rij = (p[i, j] / pmax) ** alpha
            else:
                rij = 0.0

            # Gaussian kernel k(x_i, c_j)
            diff_norm = 0.0
            for d in range(D):
                diff = xi[d] - centers[j, d]
                diff_norm += diff * diff
            diff_norm = np.sqrt(diff_norm)

            # Laplace kernel (L1 norm)
            # diff_norm = 0.0
            # for d in range(D):
            #     diff_norm += abs(xi[d] - centers[j, d])
                
            kij = np.exp(-0.5 * diff_norm / kappa)

            pij_tilde = rij * kij
            p_new[i, j] = pij_tilde
            norm += pij_tilde

        # normalize
        if norm > 0.0:
            for j in range(Nc):
                p_new[i, j] /= norm

    # --- Step 2: Update cluster centers ---
    centers_new = np.zeros_like(centers)

    for j in nb.prange(Nc):
        denom = 0.0
        num = np.zeros(D)

        for i in range(N):
            wij = p_new[i, j] * w[i]
            denom += wij
            for d in range(D):
                num[d] += wij * samples[i, d]

        if denom > 0.0:
            for d in range(D):
                centers_new[j, d] = num[d] / denom
        else:
            # fallback: keep old center
            for d in range(D):
                centers_new[j, d] = centers[j, d]

    # --- Step 3: Particle-wise consensus ---
    consensus = np.zeros((N, D))

    for i in nb.prange(N):
        for j in range(Nc):
            pij = p_new[i, j]
            for d in range(D):
                consensus[i, d] += pij * centers_new[j, d]

    return consensus, centers_new, p_new


@dataclass
class ConfigPCBOC(ConfigCBO):
    """
    kernel: kernel name to use
    kappa: witdth of the gaussian kernel
    """
    kernel: str = "gaussian"
    kappa: float = 0.1
    alpha: float = 1.
    cluster_frac: float = 0.1 

class PCBOC(CBO):
    def __init__(self, D, cfg):
        super().__init__(D, cfg)
        self.N_cluster = max(int(cfg.N_samples * cfg.cluster_frac), 1)
        self._consensus = np.zeros((cfg.N_samples, D))
        self._centers = np.zeros((self.N_cluster, D))
        self._p = self.sampler.rng.uniform(0., 1., (self.cfg.N_samples, self.N_cluster))
        self._p /= np.sum(self._p, axis=-1, keepdims=True)

    def _reset_proba(self):
        Nu = 29
        self._centers[:, :] += self.cfg.delta * self.sampler.sample(
                mean=self._zeros,
                cov=self._Id,
            )[:self.N_cluster, :]
        self._p[:] += self.cfg.cluster_frac
        self._p /= np.sum(self._p, axis=-1, keepdims=True)


    def opt_first_dim(self, n_dim: int = -1):
        super().opt_first_dim(n_dim)
        self._reset_proba()

    def update_distrib_param(self, state: SolverState, samples: Array) -> None:
        state.mean, _ = self.sampler.estimate_params(samples)
        entropy = -np.sum(self._p * np.log(self._p + 1e-10), axis=-1)
        mean_entropy = entropy.mean(axis=0)
        state.cov[:self.n_dim, :self.n_dim] = mean_entropy

    def update_mean(self, samples: Array, costs: Array) -> Tuple[int, float]:
        if self.first_it:
            self._centers[:] = self.state.mean
        
        argmin = costs.argmin()
        cmin = costs[argmin]

        self._consensus[:, :self.n_dim], self._centers[:, :self.n_dim], self._p[:] = \
            compute_consensus_clustered_numba(
                samples[:, :self.n_dim],
                costs,
                self._centers[:, :self.n_dim],
                self._p,
                cmin,
                self.cfg.kappa,
                self.cfg.beta,
                self.cfg.alpha,
            )
        # entropy = -np.sum(self._p * np.log(self._p + 1e-12), axis=-1)
        # mean_entropy = entropy.mean(axis=0)
        # print(mean_entropy)
        # print(np.mean(K))
        # print(np.max(K - np.eye(len(samples))))
        # print(np.min(K))
        return argmin, cmin