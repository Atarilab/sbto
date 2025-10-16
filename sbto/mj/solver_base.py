import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Tuple
import time
from tqdm import trange
from scipy.stats import qmc
from copy import deepcopy

from sbto.mj.nlp_mj import NLPBase, Array
from sbto.utils.config import ConfigBase, ConfigNPZBase

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
        self.N_samples = cfg.N_samples
        self.N_it = cfg.N_it
        self.seed = np.array([cfg.seed])
        self.rng = np.random.default_rng(self.seed)
        self.quasi_random = cfg.quasi_random

        self.it = 0
        self.pbar_postfix = {}

    def init_state(self,
                   mean: Array | Any = None,
                   cov: Array | Any = None,
                   sigma_mult: float = 1.0
                   ) -> SolverState:
        """
        Initialize the solver state.
        """
        if mean is None:
            mean = np.zeros(self.nlp.Nvars_u)
        if cov is None:
            cov = np.eye(self.nlp.Nvars_u) * sigma_mult**2

        return SolverState(
            mean=mean,
            cov=cov,
            min_cost=np.inf,
            min_cost_all=np.inf,
        )

    def multivariate_normal(self, state: SolverState) -> Tuple[Array, SolverState]:
        """
        Sample from the current state distribution.
        """
        if not self.quasi_random:
            noise = self.rng.multivariate_normal(
                mean=state.mean,
                cov=state.cov,
                size=(self.N_samples,),
                check_valid="ignore",
                method="cholesky"
            )
        else:
            sampler = qmc.MultivariateNormalQMC(
                mean=state.mean,
                cov=state.cov,
                rng=self.rng,
                inv_transform=False,
            )
            noise = sampler.random(self.N_samples)

        return noise, state

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
        min_cost_all = np.inf
        best_u_all = None
        pbar = trange(self.N_it, desc="Optimizing", leave=True)

        start = time.time()
        for self.it in pbar:
            eps, state = self.multivariate_normal(state)

            state, costs, best_u = self.update(state, eps)
            states.append(deepcopy(state))
            
            if state.min_cost_all < min_cost_all:
                min_cost_all = state.min_cost_all
                best_u_all = best_u
            all_costs.append(costs)

            self.pbar_postfix["min_cost"] = min_cost_all
            pbar.set_postfix(self.pbar_postfix)

        end = time.time()
        duration = end - start
        print(f"Solving time: {duration:.2f}s")

        all_costs = np.asarray(all_costs).reshape(self.N_it, -1)
        return states, best_u_all, min_cost_all, all_costs 
    
    def evaluate(self, u_traj: Array) -> Tuple[Array, Array, Array, float]:
        """
        Evaluate trajectory and returns rollout data.
        """
        x_traj, u_traj, obs_traj = self.nlp.rollout(u_traj)
        cost = self.nlp.cost(x_traj, u_traj, obs_traj)
        return (
            np.squeeze(x_traj),
            np.squeeze(u_traj),
            np.squeeze(obs_traj),
            np.squeeze(cost),
        )
    
    def update_min_cost(self, state: SolverState, min_cost_rollout : float) -> SolverState:
        """
        Update the mean cost in the solver state.
        """
        new_min_cost_all = min(min_cost_rollout, state.min_cost_all)
        state.min_cost=min_cost_rollout
        state.min_cost_all=new_min_cost_all
        return state
            
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
