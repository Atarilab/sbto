import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass

from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

@dataclass
class ConfigCEMMod(ConfigSolver):
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
    keep_frac: float = 0.
    _target_:str = "sbto.solvers.cem.CEM"
    
class CEMMod(SamplingBasedSolver):
    """
    Cross-Entropy Method (CEM) solver.
    """
    def __init__(self, D, cfg: ConfigCEMMod):
        super().__init__(D, cfg)
        self.N_elite = int(cfg.elite_frac * cfg.N_samples)
        self.N_keep = int(self.N_elite * cfg.keep_frac)
        # small diagonal regularization for covariance
        self.Id = np.diag(np.full(self.D, cfg.std_incr))
        self.I = np.eye(D)
        self.reg_cov = cfg.std_incr > 0.
        
        self.first_it = True
        if self.N_keep > 0:
            self.samples = np.empty((cfg.N_samples, D))
    
    def update_mod(self, decay_t_knots: Array):
        Nknots = len(decay_t_knots)
        Nu = self.D // Nknots
        alpha_min = 0.
        decay_t_knots /= np.max(decay_t_knots)
        decay_knots = np.repeat(decay_t_knots, Nu)
        self.alpha_mean = (self.cfg.alpha_mean - alpha_min) * decay_knots + alpha_min
        
        std_min = 0.0
        self.alpha_cov = np.full((self.D, self.D), std_min)
        for k in range(self.D):
            if k > 0:
                self.alpha_cov += np.diag(decay_knots[k:] * (self.cfg.alpha_cov-std_min), k) 
            self.alpha_cov += np.diag(decay_knots[k:] * (self.cfg.alpha_cov-std_min), -k)
        # Make sure alpha_cov have positive eigen values
        # Numerically unstable otherwise
        # E = np.linalg.eigvalsh(self.alpha_cov)
        # self.alpha_cov += self.I * np.abs(np.min(E))

    def get_samples(self) -> Array:
        """
        Get samples from distribution parametrized
        by the current state.
        """
        samples = super().get_samples()

        if self.N_keep > 0 and not self.first_it:
            self.samples[self.N_keep:] = samples[self.N_keep:]
            return self.samples
        else:
            return samples
        
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
        state.mean += self.alpha_mean * (mean - state.mean)
        state.cov += self.alpha_cov * (cov - state.cov)

    def update(self,
               samples: Array,
               costs: Array,
               ) -> None:
        """
        Update the solver state from elite samples.
        """
        elites, elites_idx = self.get_elites(samples, costs)
        self.update_distrib_param(self.state, elites)
        if self.N_keep > 0:
            self.samples[:self.N_keep] = elites[:self.N_keep]

        arg_min = elites_idx[0]
        best = samples[arg_min]
        min_cost = costs[arg_min]
        self.update_min_cost_best(self.state, min_cost, best)

        self.first_it = False