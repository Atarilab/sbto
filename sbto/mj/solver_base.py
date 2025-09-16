import numpy as np
from flax.struct import dataclass
from abc import ABC, abstractmethod
from typing import Any, Tuple
import time
from tqdm import trange

from sbto.mj.nlp_mj import NLPBase, Array

@dataclass
class SolverState:
    """
    State parameters for the solver.
    e.g. mean, covariance, temperature, etc.
    """
    mean: Array
    cov: Array
    rng: Array
    temperature: float
    min_cost: float

class SamplingBasedSolver(ABC):
    """
    Abstract base class for sampling-based solvers.
    """
    
    def __init__(self,
                 nlp : NLPBase,
                 N_samples: int = 100,
                 seed : int = 0):
        self.nlp = nlp
        self.N_samples = N_samples
        self.seed = np.array([seed])
        self.rng = np.random.default_rng(self.seed)

    def init_state(self,
                   mean: Array | Any = None,
                   cov: Array | Any = None,
                   temperature: float = 1.0,
                   sigma_mult: float = 1.0) -> SolverState:
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
            rng=self.seed,
            temperature=temperature,
            min_cost=np.inf,
        )

    def sample(self, state: SolverState) -> Tuple[Array, SolverState]:
        """
        Sample from the current state distribution.
        """
        noise = self.rng.multivariate_normal(
            mean=state.mean,
            cov=state.cov,
            size=(self.N_samples,),
            check_valid="ignore",
            method="cholesky"
        )
        return noise, state

    def solve(self,
              state: SolverState,
              Nit: int = 100,
              ) -> SolverState:
        """
        Solve the optimization problem.
        
        Args:
            state (SolverState): Initial state of the solver.
            N_steps (int): Number of optimization steps.
        
        Returns:
            SolverState: Final state after optimization.
        """
        states = []
        min_cost_all = np.inf
        best_u_all = None
        pbar = trange(Nit, desc="Optimizing", leave=True)

        start = time.time()
        for _ in pbar:
            eps, state = self.sample(state)
            state, min_cost, best_u = self.update(state, eps)
            states.append(state)

            if min_cost < min_cost_all:
                min_cost_all = min_cost
                best_u_all = best_u

            pbar.set_postfix(best_cost=min_cost_all)

        end = time.time()
        duration = end - start
        print(f"Solving time: {duration}s")
        return states, min_cost_all, best_u_all
    
    def evaluate(self, state: SolverState) -> Tuple[Array, Array, Array, float]:
        """
        Evaluate the solver state.
        """
        x_traj, u_traj, obs_traj = self.nlp.rollout(state.mean[None, ...])
        cost = self.nlp.cost(x_traj, u_traj, obs_traj)
        return x_traj, u_traj, obs_traj, cost
    
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
