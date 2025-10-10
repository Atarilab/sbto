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
                 quasi_random: bool = True,
                 ):
        """
        Args:
            nlp: NLP problem instance.
            N_samples: Number of samples per iteration.
            elite_frac: Fraction of samples considered elite.
            alpha_mean: Smoothing coefficient for mean update.
            alpha_cov: Smoothing coefficient for covariance update.
            seed: Random seed.
        """
        # Keep and shift N_elite samples
        self.elite_frac = elite_frac
        self.N_elite = int(self.elite_frac * N_samples)
        super().__init__(nlp, N_samples, seed, quasi_random)
        self.alpha_mean = alpha_mean
        self.alpha_cov = alpha_cov

        # Small diagonal regularization for covariance
        self.std = 2e-4
        self.Id = np.eye(nlp.Nvars_u)
        self.it_no_improvement = 0

        # History of all samples and their costs
        self.all_samples = np.zeros((self.N_samples + 1 * self.N_elite, self.nlp.Nvars_u))
        self.all_costs = np.full(self.N_samples + 1 * self.N_elite, np.inf)
        self._elite_hist = None
        self._cost_elite_hist = np.full(self.N_elite, np.inf)

    def random_interpolate_elites(self) -> Array:
        lmbda = self.N_elite / 3 # Higher means selecting one of the elites, Lower means mixing
        weights = self.rng.uniform(size=(self.N_elite, self.N_elite)) * lmbda
        scaled_weights = np.exp(-weights)
        scaled_weights /= np.sum(scaled_weights, axis=1, keepdims=True)  # normalize rows
        return scaled_weights @ self._elite_hist

    def update(self, state: SolverState, eps: Array) -> Tuple[SolverState, Array, Array]:
        """
        Update solver state using elite samples accumulated over history.
        """
        self.all_samples[self.N_elite:] = eps

        # Shift half elites
        if not self._elite_hist is None:
            self.all_samples[:self.N_elite] = self.random_interpolate_elites()
            # Add best and mean
            self.all_samples[0] = state.mean

        # self.all_samples[self.N_elite:-self.N_elite] = eps
        # costs = self.nlp.cost(*self.nlp.rollout(self.all_samples[:-self.N_elite]))
        # self.all_costs[:-self.N_elite] = costs

        # Add last elites
        # self.all_samples[-self.N_elite:] = self._elite_hist
        # self.all_costs[-self.N_elite:] = self._cost_elite_hist

        # Add mean and best
        costs = self.nlp.cost(*self.nlp.rollout(self.all_samples))
        self.all_costs = costs

        # Compute elite set
        argsort_idx = np.argsort(self.all_costs)
        elite_idx = argsort_idx[:self.N_elite]
        elites = self.all_samples[elite_idx]
        elite_costs = self.all_costs[elite_idx]

        self._elite_hist = elites
        # self._cost_elite_hist = elite_costs

        # Mean and covariance from elites
        mean = np.mean(elites, axis=0)
        cov = np.cov(elites, rowvar=False)

        # Best current sample
        min_cost = elite_costs[0]
        best_control = elites[0]

        # Increase noise if no improvement
        if min_cost >= state.min_cost:
            self.it_no_improvement += 1
            id_elite = self.rng.choice(self.N_elite)
            mean=elites[id_elite] + self.alpha_mean * (mean - elites[id_elite])
            cov += self.Id * self.std * self.it_no_improvement
        else:
            self.it_no_improvement = 0

        # Exponential smoothing update
        state.mean=state.mean + self.alpha_mean * (mean - state.mean)
        state.cov=state.cov + self.alpha_cov * (cov - state.cov)

        state = self.update_min_cost(state, min_cost)

        return state, self.all_costs[argsort_idx[:self.N_samples]], best_control
