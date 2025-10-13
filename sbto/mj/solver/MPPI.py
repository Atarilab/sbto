# sbto/mj/solver/mppi.py

import numpy as np
from typing import Tuple

from sbto.mj.nlp_base import NLPBase, Array
from sbto.mj.solver_base import SamplingBasedSolver, SolverState


class MPPI(SamplingBasedSolver):
    """
    Model Predictive Path Integral (MPPI) solver.

    Update rule (per iteration):
        - Sample control sequences around current mean ~ N(mean, cov)
        - Roll out each sample and compute total costs
        - Compute softmin weights w_i = exp(-(S_i - S_min)/lambda_)
        - Update mean with weighted noise: mean <- mean + sum_i w_i * (u_i - mean)
        - Update covariance with weighted covariance for exploration
    """
    def __init__(self,
                 nlp: NLPBase,
                 N_samples: int = 100,
                 lambda_: float = 1.0,      # temperature (lower -> greedier)
                 alpha_mean: float = 1.0,   # smoothing for mean update
                 alpha_cov: float = 0.2,    # smoothing for covariance update
                 seed: int = 0,
                 quasi_random: bool = True):
        super().__init__(nlp, N_samples, seed, quasi_random)
        self.lambda_ = float(lambda_)
        self.alpha_mean = float(alpha_mean)
        self.alpha_cov = float(alpha_cov)

        # small diagonal regularization for covariance (per-knot scaling)
        a, b = 1e-4, 1e-3
        self.Id = np.diag(np.linspace(a, b, self.nlp.Nknots).repeat(self.nlp.Nu))

    def update(self, state: SolverState, eps: Array) -> Tuple[SolverState, Array, Array]:
        """
        MPPI update using softmin weights.
       
        """
        # Evaluating all samples
        costs = self.nlp.cost(*self.nlp.rollout(eps))  # shape (N_samples,)

        # Softmin weights 
        c_min = float(np.min(costs))
        # guard lambda_ to avoid div-by-zero
        lam = max(self.lambda_, 1e-8)
        w = np.exp(-(costs - c_min) / lam)
        w_sum = float(np.sum(w)) + 1e-12
        w /= w_sum  

        # Mean update 
        #    delta_i = (u_i - mean); mean_new = mean + sum_i w_i * delta_i
        delta = eps - state.mean  # (N_samples, D)
        delta_mean = w @ delta    # (D,)
        new_mean = state.mean + self.alpha_mean * delta_mean

        # Covariance update 
        diff = eps - new_mean                         # (N_samples, D)
        cov_w = (diff.T * w) @ diff + self.Id        # (D, D)
        new_cov = (1.0 - self.alpha_cov) * state.cov + self.alpha_cov * cov_w

        #  best sample
        arg_min = int(np.argmin(costs))
        min_cost = float(costs[arg_min])
        best_control = eps[arg_min]

        #  Commit updates
        state.mean = new_mean
        state.cov = new_cov
        state = self.update_min_cost(state, min_cost)

        return state, costs, best_control