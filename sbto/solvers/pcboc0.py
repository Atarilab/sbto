import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass
import numba as nb

from sbto.solvers.cbo import ConfigCBO, CBO, SolverState

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
    
@nb.njit(fastmath=True)
def compute_cluster_kernel_dist(
    centers,        # (Nc, D)
    kappa,
    kernel_type,
    ):
    Nc = centers.shape[0]
    K = np.eye(Nc)
    # We use a nested loop structure that fills both K[i,j] and K[j,i]
    for i in nb.prange(Nc):
        for j in range(i + 1, Nc):
            val = kernel_value(centers[i], centers[j], kappa, kernel_type)
            K[i, j] = val
            K[j, i] = val

    return K

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
    kernel_type,
    repulsion_strength,   # NEW
):
    N, D = samples.shape
    Nc = centers.shape[0]

    # --- Boltzmann weights ---
    w = np.empty(N)
    for i in range(N):
        w[i] = np.exp(-(costs[i] - cmin) * beta)

    # --- Step 1: Update probabilities ---
    p_new = np.zeros((N, Nc))

    for i in nb.prange(N):
        pmax = 0.0
        for j in range(Nc):
            if p[i, j] > pmax:
                pmax = p[i, j]

        xi = samples[i]
        norm = 0.0

        for j in range(Nc):
            if pmax > 0.0:
                rij = (p[i, j] / pmax) ** alpha
            else:
                rij = 0.0

            kij = kernel_value(xi, centers[j], kappa, kernel_type)

            pij_tilde = rij * kij
            p_new[i, j] = pij_tilde
            norm += pij_tilde

        if norm > 0.0:
            for j in range(Nc):
                p_new[i, j] /= norm

    # --- Step 2: Update cluster centers (mean) ---
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
            for d in range(D):
                centers_new[j, d] = centers[j, d]

    # --- Step 2.5: Repulsion between cluster centers ---
    if repulsion_strength > 0.0:
        for j in nb.prange(Nc):
            for k in range(Nc):
                if k != j:
                    kij = kernel_value(
                        centers_new[j], centers_new[k],
                        kappa, kernel_type
                    )

                    for d in range(D):
                        centers_new[j, d] += (
                            repulsion_strength * kij *
                            (centers_new[j, d] - centers_new[k, d])
                        )

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
    alpha_p: float = 0.3
    alpha: float = 1.
    alpha_incr: float = 1.
    cluster_frac: float = 0.1 
    repulsion: float = 0.1 

class PCBOC(CBO):
    def __init__(self, D, cfg):
        super().__init__(D, cfg)
        self.N_cluster = max(int(cfg.N_samples * cfg.cluster_frac), 1)
        self._consensus = np.zeros((cfg.N_samples, D))
        self._centers = np.zeros((self.N_cluster, D))
        self._p = self.sampler.rng.uniform(0., 1., (self.cfg.N_samples, self.N_cluster))
        self._p /= np.sum(self._p, axis=-1, keepdims=True)
        self._kernel_type = KERNEL_TYPE[self.cfg.kernel]
        self._alpha = cfg.alpha

    def _reset_proba(self):
        # self._centers[:, :self.n_dim] += self.cfg.delta * self.sampler.sample(
        #         mean=self._zeros,
        #         cov=self._Id,
        #     )[:self.N_cluster, :self.n_dim]
        self._p += self.cfg.alpha_p * (1 / self.N_cluster - self._p)
        # self._p /= np.sum(self._p, axis=-1, keepdims=True)


    def opt_first_dim(self, n_dim: int = -1):
        self._reset_proba()
        self._alpha = self.cfg.alpha 
        super().opt_first_dim(n_dim)

    def update_distrib_param(self, state: SolverState, samples: Array) -> None:
        state.mean, _ = self.sampler.estimate_params(samples)
        state.cov[:self.n_dim, :self.n_dim] = self.mean_entropy
        # self.logs["entr"] = self.mean_entropy
        self.logs["mean_K"], _, _ = self.cluster_kernel_dist

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
                self._alpha,
                self._kernel_type,
                repulsion_strength=self.cfg.repulsion
            )
        # Nu = 29
        # self._centers[:, self.n_dim-Nu:self.n_dim] = centers[:, self.n_dim-Nu:self.n_dim]
        self._alpha += self.cfg.alpha_incr

        return argmin, cmin
    
    @property
    def mean_entropy(self):
        entropy = -np.sum(self._p * np.log(self._p + 1e-10), axis=-1)
        mean_entropy = entropy.mean(axis=0)
        return mean_entropy
    
    @property
    def cluster_kernel_dist(self):
        K = compute_cluster_kernel_dist(self._centers, self.cfg.kappa, self._kernel_type)
        mean = np.mean(K)
        min = np.min(K)
        max = np.max(K - np.eye(K.shape[0]))
        return mean, min, max