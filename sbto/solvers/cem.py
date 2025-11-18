import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass

from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

@dataclass
class ConfigCEM(ConfigSolver):
    """
    elite_frac: Fraction of samples considered elite.
    alpha_mean: Smoothing coefficient for mean update.
    alpha_cov: Smoothing coefficient for covariance update.
    std_incr: Increase the diag of the cov matrix.
    """
    elite_frac: float = 0.05
    alpha_mean: float = 0.9
    alpha_cov: float = 0.1
    std_incr: float = 0.
    _target_:str = "sbto.solvers.cem.CEM"
    
class CEM(SamplingBasedSolver):
    """
    Cross-Entropy Method (CEM) solver.
    """
    def __init__(self, D, cfg: ConfigCEM):
        super().__init__(D, cfg)
        self.N_elite = int(cfg.elite_frac * cfg.N_samples)
        # small diagonal regularization for covariance
        self.Id = np.diag(np.full(self.D, cfg.std_incr))
        self.reg_cov = cfg.std_incr > 0.

    def get_elites(self, samples: Array, costs: Array) -> Tuple[Array, IntArray]:
        """
        Returns (elites, elite_idx)
        """
        elites_idx = np.argpartition(costs, self.N_elite)[:self.N_elite]
        elites_idx = elites_idx[np.argsort(costs[elites_idx])]

        elites = samples[elites_idx]
        return elites, elites_idx
    
    def update_distrib_param(self, state: SolverState, elites: Array) -> None:
        mean, cov = self.sampler.estimate_params(elites)
        if self.reg_cov:
            cov += self.Id
        # Update state params with exponential smoothing
        state.mean += self.cfg.alpha_mean * (mean - state.mean)
        state.cov += self.cfg.alpha_cov * (cov - state.cov)

    def update(self,
               samples: Array,
               costs: Array,
               ) -> None:
        """
        Update the solver state from elite samples.
        """
        elites, elites_idx = self.get_elites(samples, costs)
        self.update_distrib_param(self.state, elites)

        arg_min = elites_idx[0]
        best = samples[arg_min]
        min_cost = costs[arg_min]
        self.update_min_cost_best(self.state, min_cost, best)