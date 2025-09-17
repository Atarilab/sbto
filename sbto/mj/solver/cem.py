import numpy as np
from typing import Tuple

from sbto.mj.nlp_base import NLPBase, Array
from sbto.mj.solver_base import SamplingBasedSolver, SolverState

class CEM(SamplingBasedSolver):
    """
    Cross-Entropy Method (CEM) solver.
    """
    def __init__(self,
                 nlp: NLPBase,
                 N_samples: int = 100,
                 elite_frac: float = 0.1,
                 alpha_mean: float = 0.8,
                 alpha_cov: float = 0.3,
                 seed: int = 0):
        super().__init__(nlp, N_samples, seed)
        self.elite_frac = elite_frac
        self.N_elite = int(self.elite_frac * self.Nsamples)
        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov

        # small diagonal regularization for covariance
        a, b = 1e-4, 1e-3
        self.Id = np.diag(np.linspace(a, b, self.nlp.Nknots).repeat(self.nlp.Nu))

    def update(self, state: SolverState, eps: Array) -> Tuple[SolverState, float, Array]:
        """
        Update the solver state using the elite samples.
        """
        costs = self.nlp.cost(*self.nlp.rollout(eps))
        
        elite_idx = np.argsort(costs)[:self.N_elite]
        elites = eps[elite_idx]

        mean = np.mean(elites, axis=0)
        cov = np.cov(elites, rowvar=False) + self.Id

        # Best sample
        arg_min = elite_idx[0]
        min_cost = float(costs[arg_min])
        best_control = eps[arg_min]

        # Update state with exponential smoothing
        state = state.replace(
            mean = state.mean + self.alpha_mean * (mean - state.mean),
            cov = state.cov + self.alpha_cov * (cov - state.cov),
        )
        state = self.update_min_cost(state, min_cost)

        return state, costs, best_control