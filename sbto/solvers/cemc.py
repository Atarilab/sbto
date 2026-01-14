import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, Any
from dataclasses import dataclass
from numba import njit, prange
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver

Array = npt.NDArray[np.float64]

# -----------------------------------------------------------------------------
# NUMBA KERNELS (Optimized for Full Covariance D=600)
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def _solve_triangular_lower(L, b):
    """Custom Numba-friendly forward substitution for L @ x = b"""
    n = L.shape[0]
    x = np.zeros_like(b)
    for i in range(n):
        tmp = b[i]
        for j in range(i):
            tmp -= L[i, j] * x[j]
        x[i] = tmp / L[i, i]
    return x

@njit(fastmath=True, parallel=True)
def _gmm_assignment_and_stats_full(
    elites: Array,
    means: Array,
    covs: Array,
    weights: Array,
    reg_eps: float
) -> Tuple[Array, Array, Array]:
    n_elites, dim = elites.shape
    n_clusters = means.shape[0]
    
    assignments = np.zeros(n_elites, dtype=np.int32)
    
    # Pre-compute Cholesky and Log-Dets for all clusters
    L_mats = np.zeros((n_clusters, dim, dim))
    log_dets = np.zeros(n_clusters)
    
    for k in range(n_clusters):
        # Add regularization to ensure positive-definiteness
        reg_cov = covs[k].copy()
        for d in range(dim):
            reg_cov[d, d] += reg_eps
            
        L = np.linalg.cholesky(reg_cov)
        L_mats[k] = L
        # log|det(Sigma)| = 2 * sum(log(diag(L)))
        ldet = 0.0
        for d in range(dim):
            ldet += np.log(L[d, d])
        log_dets[k] = 2.0 * ldet

    # 1. Assignment Step (E-step)
    for i in prange(n_elites):
        best_k = -1
        best_log_prob = -np.inf
        
        for k in range(n_clusters):
            # Mahalanobis distance: (x-mu)^T @ Inv(Sigma) @ (x-mu)
            # Calculated as ||L^-1 @ (x-mu)||^2
            diff = elites[i] - means[k]
            y = _solve_triangular_lower(L_mats[k], diff)
            mahalanobis = np.sum(y**2)
            
            # Log-Likelihood (ignoring constant 2*pi terms)
            log_prob = np.log(weights[k] + 1e-10) - 0.5 * (log_dets[k] + mahalanobis)
            
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_k = k
        assignments[i] = best_k

    # 2. Update Statistics (M-step)
    new_means = np.zeros((n_clusters, dim), dtype=np.float32)
    new_covs = np.zeros((n_clusters, dim, dim), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.float32)
    
    for i in range(n_elites):
        k = assignments[i]
        counts[k] += 1.0
        new_means[k] += elites[i]

    new_std = np.mean(covs) * 0.1
    for k in range(n_clusters):
        if counts[k] > 0:
            new_means[k] /= counts[k]
        else:
            # Re-seed dead cluster to a random elite with wide variance
            rand_idx = np.random.randint(0, n_elites)
            new_means[k] = elites[rand_idx].copy()
            new_covs[k, ...] = covs[assignments[rand_idx], ...].copy()

    # Compute Sample Covariance
    for i in range(n_elites):
        k = assignments[i]
        diff = elites[i] - new_means[k]
        # Outer product update
        for r in range(dim):
            for c in range(dim):
                new_covs[k, r, c] += diff[r] * diff[c]

    for k in range(n_clusters):
        if counts[k] > 1:
            new_covs[k] /= (counts[k] - 1)
        
    return new_means, new_covs, counts

# -----------------------------------------------------------------------------
# CLUSTERIZED CEM IMPLEMENTATION (FULL COVARIANCE)
# -----------------------------------------------------------------------------

@dataclass
class ConfigCEMC(ConfigSolver):
    elite_frac: float = 0.1
    alpha_mean: float = 0.8
    alpha_cov: float = 0.2
    std_incr: float = 1e-5  # Regularization epsilon
    keep_frac: float = 0.05
    n_clusters: int = 5

class ClusterizedCEM(SamplingBasedSolver):
    def __init__(self, D: int, cfg: ConfigCEMC):
        self.n_clusters = cfg.n_clusters
        super().__init__(D, cfg)
        self.cfg = cfg
        self.N_elite = int(cfg.elite_frac * cfg.N_samples)
        self.N_keep = int(self.N_elite * cfg.keep_frac)
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        self.samples = np.zeros((cfg.N_samples, D))
        self.first_it = True

    def opt_first_dim(self, n_dim = -1):
        self.weights = np.ones(self.n_clusters) / self.n_clusters
        return super().opt_first_dim(n_dim)

    def set_state(self, state: SolverState):
        cov = np.tile(state.cov[None, :], (self.n_clusters, 1, 1))
        mean = np.tile(state.mean[None, :], (self.n_clusters, 1)) + np.random.randn(self.n_clusters, self.D) * 0.15
        self.state = SolverState(
        mean=np.float32(mean),
        cov=np.float32(cov),
        best=state.best,
        best_all=state.best_all,
        min_cost=state.min_cost,
        min_cost_all=state.min_cost_all,
        )

    def increment_criteria(self):
        return np.max(self.state.cov[:, :self.n_dim, :self.n_dim])
        
    def init_state(self, mean=None, cov=None) -> SolverState:
        if mean is None:
            # Spread initial means slightly to encourage separation
            mean = np.random.randn(self.n_clusters, self.D) * 0.1
        if cov is None:
            # Full Covariance Initialization (K, D, D)
            cov = np.zeros((self.n_clusters, self.D, self.D))
            for k in range(self.n_clusters):
                cov[k] = np.eye(self.D) * (self.cfg.sigma0**2)

        return SolverState(
            mean=mean, cov=cov,
            best=np.zeros(self.D), best_all=np.zeros(self.D),
            min_cost=np.inf, min_cost_all=np.inf,
        )

    def get_samples(self) -> Array:
        N_total = self.cfg.N_samples
        start_idx = 0 if self.first_it else self.N_keep
        N_to_sample = N_total - start_idx
        
        if N_to_sample <= 0: return self.samples
        
        counts = np.random.multinomial(N_to_sample, self.weights)
        current_idx = start_idx
        
        for k in range(self.n_clusters):
            count = counts[k]
            if count <= 0: continue
            
            # Sample using Cholesky: x = mu + L @ z
            cluster_samples = self.sampler.sample(self.state.mean[k][:self.n_dim], self.state.cov[k, :self.n_dim, :self.n_dim])[:count]
            self.samples[current_idx:current_idx + count, :self.n_dim] = cluster_samples
            current_idx += count
            
        return self.samples

    def update(self, samples: Array, costs: Array) -> None:
        # 1. Selection
        idx = np.argpartition(costs, self.N_elite)[:self.N_elite]
        idx = idx[np.argsort(costs[idx])]
        elites = samples[idx]
        
        # Update Global Best
        self.update_min_cost_best(self.state, costs[idx[0]], elites[0])

        if self.N_keep > 0:
            self.samples[:self.N_keep] = elites[:self.N_keep]

        # 2. GMM Fitting
        new_means, new_covs, new_counts = _gmm_assignment_and_stats_full(
            elites[:, :self.n_dim], self.state.mean[:, :self.n_dim], self.state.cov[:, :self.n_dim, :self.n_dim], self.weights, 1e-6
        )
        self.logs["count"] = np.min(new_counts)
        # 3. Weights update with "Entropy Injection" (prevents weight collapse)
        if np.sum(new_counts) > 0:
            target_weights = new_counts / np.sum(new_counts)
            self.weights = (1 - self.cfg.alpha_cov) * self.weights + self.cfg.alpha_cov * target_weights
            self.weights = np.maximum(self.weights, 0.15) # Floor weight at 5%
            alpha = 0.9
            self.weights = self.weights ** alpha
            self.weights /= np.sum(self.weights)

        # 4. State Update (Smoothing)
        mask = (new_counts > -1)
        for k in range(self.n_clusters):
            if mask[k]:
                self.state.mean[k, :self.n_dim] = (1 - self.cfg.alpha_mean) * self.state.mean[k, :self.n_dim] + self.cfg.alpha_mean * new_means[k, :self.n_dim]
                self.state.cov[k, :self.n_dim, :self.n_dim] = (1 - self.cfg.alpha_cov) * self.state.cov[k, :self.n_dim, :self.n_dim] + self.cfg.alpha_cov * new_covs[k, :self.n_dim, :self.n_dim]
        
        self.first_it = False