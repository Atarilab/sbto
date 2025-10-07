import numpy as np
from typing import Tuple, Optional

from sbto.mj.nlp_base import NLPBase, Array
from sbto.mj.solver_base import SamplingBasedSolver, SolverState


class EfficientCEM(SamplingBasedSolver):
    """
    Cross-Entropy Method (CEM) solver with elite set computed from all past samples.
    """

    def __init__(self,
                 nlp: NLPBase,
                 N_samples: int = 100,
                 elite_frac: float = 0.1,
                 alpha_mean: float = 0.8,
                 alpha_cov: float = 0.3,
                 seed: int = 0,
                 max_history: Optional[int] = None):
        """
        Args:
            nlp: NLP problem instance.
            N_samples: Number of samples per iteration.
            elite_frac: Fraction of samples considered elite.
            alpha_mean: Smoothing coefficient for mean update.
            alpha_cov: Smoothing coefficient for covariance update.
            seed: Random seed.
            max_history: Optional maximum number of total samples to keep in history.
        """
        # Keep and shift N_elite samples
        self.elite_frac = elite_frac
        self.N_elite = int(self.elite_frac * N_samples)
        super().__init__(nlp, N_samples + 2 * self.N_elite, seed)
        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov
        self.max_history = max_history

        # Small diagonal regularization for covariance
        a, b = 1e-4, 1e-3
        self.Id = np.diag(np.linspace(a, b, self.nlp.Nknots).repeat(self.nlp.Nu))

        # History of all samples and their costs
        self.all_costs = np.full(self.Nsamples, np.inf)
        self._elite_hist = None
        self._cost_elite_hist = np.full(self.N_elite, np.inf)

    def random_interpolate_elites(self) -> Array:
        weights = self.rng.random(size=(self.N_elite, self.N_elite))
        scaled_weights = np.exp(-weights)
        scaled_weights /= np.sum(scaled_weights, axis=1, keepdims=True)  # normalize rows
        return scaled_weights @ self._elite_hist

    def update(self, state: SolverState, eps: Array) -> Tuple[SolverState, Array, Array]:
        """
        Update solver state using elite samples accumulated over history.
        """
        # Shift half elites
        if not self._elite_hist is None:
            eps[:self.N_elite] = self.random_interpolate_elites()
        
        costs = self.nlp.cost(*self.nlp.rollout(eps[:-self.N_elite]))
        self.all_costs[:-self.N_elite] = costs

        # Add last elites
        eps[-self.N_elite:] = self._elite_hist
        self.all_costs[-self.N_elite:] = self._cost_elite_hist
    
        # Compute elite set
        elite_idx = np.argsort(self.all_costs)[:self.N_elite]
        elites = eps[elite_idx]
        elite_costs = self.all_costs[elite_idx]

        self._elite_hist = elites
        self._cost_elite_hist = elite_costs

        # Mean and covariance from elites
        mean = np.mean(elites, axis=0)
        cov = np.cov(elites, rowvar=False)

        # Best current sample
        min_cost = elite_costs[0]
        best_control = elites[0]

        # Exponential smoothing update
        state = state.replace(
            mean=state.mean + self.alpha_mean * (mean - state.mean),
            cov=state.cov + self.alpha_cov * (cov - state.cov),
        )
        state = self.update_min_cost(state, min_cost)

        return state, self.all_costs, best_control
