import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Any
import copy

from sbto.solvers.sampler import SamplerAbstract, AVAILABLE_SAMPLERS

Array = npt.NDArray[np.float64]

@dataclass
class SolverState():
    """
    State parameters for the solver.
    Default:
        - mean (Array) : best sample  [D]
        - cov (Array) : covariance matrix [D, D]
        - best (Array) : best sample [D]
        - min_cost (float) : minimum cost of current iteration
        - min_cost_all (float) : minimum cost of all iterations
    """
    mean: Array
    cov: Array
    best: Array
    min_cost: float
    min_cost_all: float

@dataclass
class ConfigSolver():
    N_samples: int = 1024
    seed: int = 0
    quasi_random: bool = True
    N_it: int = 100
    sigma0: float = 0.2
    sampler: str = "normal"

class SamplingBasedSolver(ABC):
    """
    Abstract base class for sampling-based solvers.
    """
    def __init__(self,
                 D : int, cfg : ConfigSolver):
        self.D = D
        self.cfg = cfg

        self.sampler = self._get_sampler()
        self.state = self.init_state()

    def _get_sampler(self) -> SamplerAbstract:
        sampler_name = self.cfg.sampler
        if not sampler_name in AVAILABLE_SAMPLERS.keys():
            raise ValueError(
                f"Sampler {sampler_name} not available. "
                f"Choose from {" ".join(AVAILABLE_SAMPLERS.keys())}"
            )
        SamplerClass = AVAILABLE_SAMPLERS[sampler_name]
        return SamplerClass(**self.cfg.__dict__)

    def init_state(self,
                   mean: Array | Any = None,
                   cov: Array | Any = None,
                   ) -> SolverState:
        """
        Initialize the solver state.
        """
        if mean is None:
            mean = np.zeros(self.D)
        if cov is None:
            cov = np.eye(self.D) * self.cfg.sigma0**2
        best = np.empty_like(mean)

        return SolverState(
            mean=mean,
            cov=cov,
            best=best,
            min_cost=np.inf,
            min_cost_all=np.inf,
        )

    def update_min_cost_best(
            self,
            state: SolverState,
            min_cost_rollout : float,
            best: Array
            ) -> None:
        """
        Update solver state's min_cost and best sample inplace.
        """
        state.min_cost=float(min_cost_rollout)
        if min_cost_rollout < state.min_cost_all:
            state.best = best
            state.min_cost_all=float(min_cost_rollout)

    def get_samples(self) -> Array:
        """
        Get samples from distribution parametrized
        by the current state.
        """
        return self.sampler.sample(**asdict(self.state))
        
    @abstractmethod
    def update(self,
               samples: Array,
               costs: Array,
               ) -> None:
        """
        Update inplace the solver state based on costs.
        Including minimum cost and best control.
        """
        pass
