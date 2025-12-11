import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass
import time
from numba import njit
    
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

@dataclass
class ConfigMPPI(ConfigSolver):
    """
    MPPI configuration.

    lambda_: Temperature parameter for weight calculation (often called lambda or gamma).
    alpha_mean: Smoothing coefficient for mean update (exponential smoothing).
    alpha_cov: Smoothing coefficient for covariance update.
    std_incr: Add to diagonal of covariance for numerical stability / exploration boost.
    keep_frac: fraction of highest-weighted samples to keep (optional — works like CEM keep).
    """
    lambda_: float = 10.
    alpha_mean: float = 0.9
    alpha_cov: float = 0.1
    std_incr: float = 0.0
    keep_frac: float = 0.0
    _target_: str = "sbto.solvers.mppi.MPPI"


class MPPI(SamplingBasedSolver):
    """
    Model Predictive Path Integral (MPPI) solver with covariance adaptation.

    This solver samples controls from the current Gaussian (mean, cov), evaluates costs,
    computes importance weights w_i = exp(-(cost_i - min_cost)/lambda_), and uses the
    weighted mean & covariance to adapt the distribution. Exponential smoothing is applied
    to the parameters (matching your CEM's style).
    """
    def __init__(self, D, cfg: ConfigMPPI):
        super().__init__(D, cfg)
        self.cfg: ConfigMPPI = cfg
        # number of samples to keep (optional)
        self.N_keep = int(cfg.keep_frac * cfg.N_samples)
        # small diagonal regularization for covariance/adaptation
        self.Id = np.diag(np.full(self.D, cfg.std_incr))
        self.reg_cov = cfg.std_incr > 0.

        self.first_it = True
        # Pre-allocate samples buffer
        self.samples = np.zeros((cfg.N_samples, D))
        # pre-allocate weights
        self.weights = np.zeros(cfg.N_samples, dtype=np.float64)

    def get_samples(self) -> Array:
        """
        Draw samples from current state distribution.
        For MPPI we sample noise (zero-mean) and add the current mean to produce samples.
        On the very first iteration we still sample (no warm start).
        """
        # draw zero-mean samples then shift by current mean
        # sampler.sample accepts mean and cov; pass zero-mean to get noise
        noise = self.sampler.sample(
            mean=np.zeros(self.D),
            cov=self.state.cov
        )
        # Shift by mean to get actual samples
        self.samples = noise + self.state.mean[np.newaxis, :]
        return self.samples

    def _compute_weights(self, costs: Array) -> Array:
        """
        Importance weights for MPPI:
            w_i = exp( - (cost_i - min_cost) / lambda_ )
        Normalize to sum to 1. Uses numerical stabilisation (subtract min cost).
        """
        lam = float(self.cfg.lambda_)
        # ensure costs is 1D
        costs = costs.ravel()
        cmin = costs.min()
        # stabilized exponent
        exponents = -(costs - cmin) / lam
        # to avoid overflow/underflow clip exponents (large negative -> ~0)
        # but clipping is optional; we clip to a reasonable range
        exponents = np.clip(exponents, -700, 700)
        w = np.exp(exponents)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            # fallback: uniform weights or all mass on best sample
            w = np.zeros_like(w)
            best_idx = int(np.argmin(costs))
            w[best_idx] = 1.0
        else:
            w /= s
        return w

    @staticmethod
    def _weighted_mean_and_cov(samples: Array, weights: Array) -> Tuple[Array, Array]:
        """
        Compute weighted mean and covariance of samples.
        Weighted covariance uses the formula:
            cov = sum_i w_i (x_i - mu)(x_i - mu)^T
        where weights sum to 1.
        """
        mu = (weights[:, None] * samples).sum(axis=0)
        # diffs = samples - mu[np.newaxis, :]
        # compute weighted covariance
        # shape (D, D)
        cov = np.cov(samples, aweights=weights, rowvar=False)
        # cov = (weights[:, None, None] * (diffs[:, :, None] * diffs[:, None, :])).sum(axis=0)
        return mu, cov

    def update_distrib_param(self, state: SolverState, mu: Array, cov: Array) -> None:
        """
        Update state params with exponential smoothing (mimics CEM style updates).
        """
        if self.reg_cov:
            cov = cov + self.Id

        # Exponential smoothing toward weighted estimates
        state.mean += self._mask_mean * self.cfg.alpha_mean * (mu - state.mean)
        state.cov += self._mask_cov * self.cfg.alpha_cov * (cov - state.cov)

    def update(self,
               samples: Array,
               costs: Array,
               ) -> None:
        """
        Update based on MPPI weights computed from the given costs.
        Also updates best/min cost tracking in the same way your CEM does.
        """
        # compute weights
        w = self._compute_weights(costs)
        self.weights = w

        # Optionally keep top-K samples (highest weights) for next iteration (like CEM keep)
        if self.N_keep > 0 and not self.first_it:
            # copy highest-weighted samples into beginning of buffer for reuse if desired
            keep_idx = np.argsort(-w)[: self.N_keep]  # descending sort
            self.samples[:self.N_keep] = samples[keep_idx]

        # weighted mean & covariance
        mu_w, cov_w = self._weighted_mean_and_cov(samples, w)

        # update distribution parameters with smoothing
        self.update_distrib_param(self.state, mu_w, cov_w)

        # identify best sample (minimum cost) to update solver's best known
        arg_min = int(np.argmin(costs))
        best = samples[arg_min]
        min_cost = float(costs[arg_min])
        self.update_min_cost_best(self.state, min_cost, best)

        self.first_it = False
