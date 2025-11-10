import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Tuple
import time
from tqdm import trange
from copy import deepcopy

from sbto.mj.nlp_mj import NLPBase, Array
from sbto.utils.config import ConfigBase, ConfigNPZBase
from sbto.utils.sampler import SamplerAbstract, AVAILABLE_SAMPLERS

@dataclass
class SolverState(ConfigNPZBase):
    """
    State parameters for the solver.
    e.g. mean, covariance, temperature, etc.
    """
    mean: Array
    cov: Array
    min_cost: float
    min_cost_all: float

    def __post_init__(self):
        self._filename = "solver_state.npz"

@dataclass
class SolverConfig(ConfigBase):
    N_samples: int = 100
    seed: int = 0
    quasi_random: bool = True
    N_it: int = 100
    sigma0: float = 0.2
    sampler: str = "normal"

    def __post_init__(self):
        self._filename = "config_solver.yaml"

class SamplingBasedSolver(ABC):
    """
    Abstract base class for sampling-based solvers.
    """
    
    def __init__(self,
                 nlp : NLPBase,
                 cfg : SolverConfig,
                 ):
        self.nlp = nlp
        self.N_it = cfg.N_it
        self.N_samples = cfg.N_samples
        self.sigma0 = cfg.sigma0
        self.sampler = self._get_sampler(cfg)

        self.it = 0
        self.pbar_postfix = {}

    def _get_sampler(self, cfg: SolverConfig) -> SamplerAbstract:
        sampler_name = cfg.sampler
        if not sampler_name in AVAILABLE_SAMPLERS.keys():
            raise ValueError(
                f"Sampler {sampler_name} not available. "
                f"Choose from {" ".join(AVAILABLE_SAMPLERS.keys())}"
            )
        SamplerClass = AVAILABLE_SAMPLERS[sampler_name]
        return SamplerClass(**cfg.args)

    def init_state(self,
                   mean: Array | Any = None,
                   cov: Array | Any = None,
                   ) -> SolverState:
        """
        Initialize the solver state.
        """
        if mean is None:
            mean = np.zeros(self.nlp.Nvars_u)
        if cov is None:
            cov = np.eye(self.nlp.Nvars_u) * self.sigma0**2

        return SolverState(
            mean=mean,
            cov=cov,
            min_cost=np.inf,
            min_cost_all=np.inf,
        )

    def solve(self, state: SolverState,) -> Tuple[SolverState, Array, float, Array]:
        """
        Solve the optimization problem.
        
        Args:
            state (SolverState): Initial state of the solver.
            N_steps (int): Number of optimization steps.
        
        Returns:
            SolverState: Final state after optimization.
            Array: Best control knots
            float: Cost of best control
            Array: All costs of all iterations [Nit, N_samples]
        """
        states = [deepcopy(state)]
        all_costs = []
        all_samples = []
        min_cost_all = np.inf
        best_u_all = None
        pbar = trange(self.N_it, desc="Optimizing", leave=True)

        start = time.time()
        for self.it in pbar:
            eps = self.sampler.sample(**state.args)

            state, costs, best_u = self.update(state, eps)

            states.append(deepcopy(state))
            all_costs.append(costs)
            all_samples.append(eps)

            if state.min_cost_all < min_cost_all:
                min_cost_all = state.min_cost_all
                best_u_all = best_u

            self.pbar_postfix["min_cost"] = min_cost_all
            pbar.set_postfix(self.pbar_postfix)

        end = time.time()
        duration = end - start
        print(f"Solving time: {duration:.2f}s")

        all_costs = np.asarray(all_costs).reshape(self.N_it, -1)
        return states, best_u_all, min_cost_all, all_costs, all_samples
        
    def evaluate(self, samples_knots: Array) -> Tuple[Array, Array]:
        """
        Evaluate sampled knots and returns rollout data.
        Args:
            -sampled_knots [N_samples, Nknots*Nu]
        """
        return self.nlp.cost(*self.nlp.rollout(samples_knots))
    
    def update_min_cost(self, state: SolverState, min_cost_rollout : float) -> None:
        """
        Update solver state's min_cost inplace.
        """
        min_cost_rollout = float(min_cost_rollout)
        new_min_cost_all = min(min_cost_rollout, state.min_cost_all)
        state.min_cost=min_cost_rollout
        state.min_cost_all=new_min_cost_all
            
    @abstractmethod
    def update(self,
               state: SolverState,
               eps: Array) -> Tuple[SolverState, float, Array]:
        """
        Update solver state based on rollouts.

        Returns:
            Tuple[SolverState, float, Array]: Updated state and minimum cost and best control.
        """
        pass
