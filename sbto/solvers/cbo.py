import numpy as np
import numpy.typing as npt
from typing import Tuple
from dataclasses import dataclass

from sbto.solvers.solver_base import SamplingBasedSolver, SolverState, ConfigSolver

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

@dataclass
class ConfigCBO(ConfigSolver):
    """
    beta: Inverse temperature
    noise_model: standard | coordinate
    delta: diffusion term
    dt: step size
    """
    beta: float = 1.
    noise_model: str = "standard"
    delta: float = 1.e-2
    dt: float = 1.e-2

class CBO(SamplingBasedSolver):
    """
    Cross-Entropy Method (CEM) solver.
    """
    def __init__(self, D, cfg: ConfigCBO):
        super().__init__(D, cfg)
        self.Id = np.eye(D)
        
        self.first_it = True
        # if self.N_keep > 0:
        self._zeros = np.zeros(D)
        self._Id = np.eye(D) * self.cfg.dt
        self._x = np.zeros((cfg.N_samples, D))
        self._consensus = np.zeros((1, D))
        self._delta = self.cfg.delta
        self._dt = self.cfg.dt

    def update_mean(self, samples: Array, costs: Array) -> Tuple[int, float]:
        argmin = costs.argmin()
        cmin = costs[argmin]
        exponents = -(costs - cmin) * self.cfg.beta
        w = np.exp(exponents)
        s = w.sum()
        w /= s
        self._consensus[:self.n_dim] = w @ samples[:, :self.n_dim]
        return argmin, cmin
    
    def get_samples(self) -> Array:
        """
        Get samples from distribution parametrized
        by the current state.
        """
        if self.first_it:
            noise = self.sampler.sample(
                mean=self._zeros,
                cov=self._Id,
            )
            self._consensus[:] = self.state.mean
            self._x[:] = self.state.mean + self.cfg.delta * noise
            return self._x

        noise = self._delta[:self.n_dim] * self.sampler.sample(
            mean=self._zeros[:self.n_dim],
            cov=self._Id[:self.n_dim, :self.n_dim],
        )

        diff = self._x[:, :self.n_dim] - self._consensus[:, :self.n_dim]

        if self.cfg.noise_model == "standard":
            scale = np.linalg.norm(diff, axis=-1, keepdims=True) / np.sqrt(self.n_dim)
            self.logs["s"] = np.max(np.abs(scale))
            max = 0.05
            noise *= np.clip(scale, -max, max)
        elif self.cfg.noise_model == "coordinate":
            self.logs["s"] = np.max(np.abs(diff))
            max = 0.05
            noise *= np.clip(diff, -max, max)
        elif self.cfg.noise_model == "combined":
            scale = np.linalg.norm(diff, axis=-1, keepdims=True) / np.sqrt(self.n_dim) * diff
            self.logs["s"] = np.max(np.abs(scale))
            max = 0.05
            noise *= np.clip(diff, -max, max)
        else:
            raise ValueError(f"Invalid noise config ({self.cfg.noise_model}).")

        self._x[:, :self.n_dim] -= self._dt[:self.n_dim] * diff - noise
        
        return self._x

    def update_distrib_param(self, state: SolverState, samples: Array) -> None:
        state.mean, state.cov = self.sampler.estimate_params(samples)

    def update(self,
               samples: Array,
               costs: Array,
               ) -> None:
        """
        Update the solver state from elite samples.
        """
        arg_min, min_cost = self.update_mean(samples, costs)
        best = samples[arg_min]
        self.update_min_cost_best(self.state, min_cost, best)
        self.update_distrib_param(self.state, samples)

        self.first_it = False