import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from sbto.mj.nlp_base import NLPBase, Array
from sbto.mj.solver_base import SamplingBasedSolver, SolverState, SolverConfig

@dataclass
class EfficientCEMConfig(SolverConfig):
    """
    elite_frac: Fraction of samples considered elite.
    alpha_mean: Smoothing coefficient for mean update.
    alpha_cov: Smoothing coefficient for covariance update.
    std_incr: Increment std when no improvement.
    """
    elite_frac: float = 0.1
    alpha_mean: float = 0.8
    alpha_cov: float = 0.3
    std_incr: float = 1e-4

class EfficientCEM(SamplingBasedSolver):
    """
    Cross-Entropy Method (CEM) solver with elite set computed from all past samples.
    """

    def __init__(self,
                 nlp: NLPBase,
                 cfg: EfficientCEMConfig,
                 ):

        # Keep and shift N_elite samples
        self.elite_frac = cfg.elite_frac
        self.N_elite = int(self.elite_frac * cfg.N_samples)
        self.alpha_mean = cfg.alpha_mean
        self.alpha_cov = cfg.alpha_cov
        self.std_incr = cfg.std_incr
        super().__init__(nlp, cfg)

        # Small diagonal regularization for covariance
        self.Id = np.eye(nlp.Nvars_u) * self.std_incr
        self.it_no_improvement = 0

        # History of all samples and their costs
        self.all_samples = np.zeros((self.N_samples + self.N_elite, self.nlp.Nvars_u))
        self.elites = np.empty((self.N_elite, self.nlp.Nvars_u))
        self.elite_costs = np.full(self.N_elite, np.inf)

    def update(self, state: SolverState, eps: Array) -> Tuple[SolverState, Array, Array]:
        """
        Update solver state using elite samples accumulated over history.
        """
        self.all_samples[self.N_elite:] = eps

        # Shift away elites: more exploration
        if not self.elites is None:
            id_elite = self.rng.choice(self.N_elite)
            self.all_samples[:self.N_elite] = self.elites + (self.elites - self.elites[id_elite]) * self.rng.uniform(0., 1., size=(self.N_elite, 1)) * np.sqrt(1+self.it_no_improvement)
        # First iteration
        else:
            self.all_samples[:self.N_elite] = 0.5 * (eps[-self.N_elite:] + eps[:self.N_elite])

        costs = self.nlp.cost(*self.nlp.rollout(self.all_samples))

        # Compute elite set
        argsort_idx = np.argsort(costs)
        elite_idx = argsort_idx[:self.N_elite]
        self.elites = self.all_samples[elite_idx]
        self.elite_costs = costs[elite_idx]

        # Add last elites
        self.elites = self.elites
        self.cost_elites = self.elite_costs

        # Mean and covariance from elites
        mean = np.mean(self.elites, axis=0)
        cov = np.cov(self.elites, rowvar=False)

        # Best current sample
        min_cost = self.elite_costs[0]
        best_control = self.elites[0]

        # Increase noise if no improvement 
        # And shift the mean 
        if min_cost >= state.min_cost_all:
            self.it_no_improvement += 1
            id_elite = self.rng.choice(self.N_elite)
            alpha = self.rng.uniform(0.8, 1.)
            mean = self.elites[id_elite] + alpha * (mean - self.elites[id_elite])
            cov += self.Id * self.it_no_improvement
        else:
            self.it_no_improvement = 0

        self.pbar_postfix["it_no_improv"] = self.it_no_improvement
        self.pbar_postfix["n_e_interp"] = np.sum(elite_idx < self.N_elite)

        # Exponential smoothing update
        state.mean=state.mean + self.alpha_mean * (mean - state.mean)
        state.cov=state.cov + self.alpha_cov * (cov - state.cov)

        state = self.update_min_cost(state, min_cost)

        return state, costs, best_control
