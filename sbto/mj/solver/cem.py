import numpy as np
from typing import Tuple, Any
from dataclasses import dataclass
from scipy.optimize import brentq

from sbto.mj.nlp_base import NLPBase, Array
from sbto.mj.solver_base import SamplingBasedSolver, SolverState, SolverConfig

@dataclass
class CEMConfig(SolverConfig):
    """
    elite_frac: Fraction of samples considered elite.
    alpha_mean: Smoothing coefficient for mean update.
    alpha_cov: Smoothing coefficient for covariance update.
    """
    elite_frac: float = 0.1
    alpha_mean: float = 0.8
    alpha_cov: float = 0.3
    a: float = 1e-4
    b: float = 1e-3

class CEM(SamplingBasedSolver):
    """
    Cross-Entropy Method (CEM) solver.
    """
    def __init__(self,
                 nlp: NLPBase,
                 cfg: CEMConfig,
                 ):
        super().__init__(nlp, cfg)
        self.elite_frac = cfg.elite_frac
        self.N_elite = int(self.elite_frac * self.N_samples)
        self.alpha_mean = cfg.alpha_mean
        self.alpha_cov = cfg.alpha_cov

        # small diagonal regularization for covariance
        a, b = cfg.a, cfg.b
        self.Id = np.diag(np.linspace(a, b, self.nlp.Nknots).repeat(self.nlp.Nu))

    def get_elites(self, samples: Array, costs: Array) -> Tuple[Array, int]:
        """
        Returns (elites, arg_min)
        """
        elite_idx = np.argsort(costs)[:self.N_elite]
        elites = samples[elite_idx]
        # Best sample
        arg_min = elite_idx[0]
        return elites, arg_min
    
    def update_distrib_param(self, state: SolverState, elites: Array) -> SolverState:
        mean, cov = self.sampler.estimate_params(elites)
        # Update state params with exponential smoothing
        state.mean += self.alpha_mean * (mean - state.mean)
        state.cov += self.alpha_cov * (cov + self.Id - state.cov)

    def update(self, state: SolverState, samples: Array) -> Tuple[SolverState, float, Array]:
        """
        Update the solver state using the elite samples.
        """
        samples_qdes, costs = self.evaluate(samples)
        elites, arg_min = self.get_elites(samples, costs)
        best = samples_qdes[arg_min]
        min_cost = costs[arg_min]
        self.update_distrib_param(state, elites)
        self.update_min_cost(state, min_cost)
                
        return state, costs, best

@dataclass
class CEMBetaConfig(CEMConfig):
    """
    kappa=a+b to initialize beta distribution 
    """
    kappa: float = 10.

    def __post_init__(self):
        super().__post_init__()
        # beta distribution
        self.sampler = "beta"
        # beta samples in [0,1]
        # has to be linear scaling to [q_min, q_max]
        self.scaling = "linear"

@dataclass
class CEMBetaState(SolverState):
    a: Array
    b: Array
    Sigma: Array

class CEMBeta(CEM):
    """
    Cross-Entropy Method (CEM) solver.
    """
    def __init__(self,
                 nlp: NLPBase,
                 cfg: CEMBetaConfig,
                 ):
        super().__init__(nlp, cfg)
        self.kappa = cfg.kappa

    def _get_mode_in_unit_interval(self) -> Array:
        return (self.q_nom - self.q_min) / self.q_range

    def get_a_b_from_mode_and_kappa(self, kappa=6.0):
        """
        Compute Beta(a,b) parameters so that the Beta mode maps to q_nom after
        linear rescaling from [0,1] -> [q_min, q_max].
        kappa = a + b (must be > 2).
        Returns (a, b).
        """
        if kappa <= 2.0:
            raise ValueError("kappa must be > 2 to place an interior mode (a,b > 1).")

        # target mode in unit interval
        m = self._get_mode_in_unit_interval()

        # a, b per actuator
        a_act = m * (kappa - 2.0) + 1.0
        b_act = (1.0 - m) * (kappa - 2.0) + 1.0

        # a, b for each knots
        a = np.tile(a_act, self.nlp.Nknots)
        b = np.tile(b_act, self.nlp.Nknots)
        return a, b

    def init_state(self,
                   Sigma : Array | Any = None,
                   ) -> SolverState:
        """
        Initialize the solver state.
        """
        if Sigma is None:
            Sigma = np.eye(self.nlp.Nvars_u)
        if not Sigma.shape == (self.nlp.Nvars_u, self.nlp.Nvars_u):
            raise ValueError(f"Invalid Sigma shape")
        if not np.all(np.diag(Sigma) == np.ones(self.nlp.Nvars_u)):
            raise ValueError(f"Diagonal values of Sigma should be 1.")
        
        a, b = self.get_a_b_from_mode_and_kappa(self.kappa)
        mean, cov = self.compute_mean_cov_scaled(a, b)

        return CEMBetaState(
            mean=mean,
            cov=cov,
            a=a,
            b=b,
            Sigma=Sigma,
            min_cost=np.inf,
            min_cost_all=np.inf,
        )

    def compute_mean_cov_scaled(self, a, b):
        """
        Compute mean and variance of beta(a, b).
        Rescaled to the joint range.
        """
        mean = a / (a + b)
        mean_q = self.f_rescale(mean).reshape(-1)

        var = (a * b) / ((a + b)**2 * (a + b + 1))
        var = var.reshape(-1, self.nlp.Nu)
        # Var[a*X] = a**2 * X
        var *= self.q_range**2
        var = var.reshape(-1)
        diag_cov = np.diag(var)

        return mean_q, diag_cov
    
    def update_distrib_param(self, state: CEMBetaState, elites: Array):
        a, b, Sigma = self.sampler.estimate_params(elites)
        # Update state with weighted geometric average
        w1, w2 = self.alpha_mean, 1 - self.alpha_mean
        state.a = a**w1 * state.a**w2
        state.b = b**w1 * state.b**w2
        state.Sigma += self.alpha_cov * (Sigma - state.Sigma)
        # Update mean and cov for plotting
        state.mean, state.cov = self.compute_mean_cov_scaled(a, b)


@dataclass
class CEMKumaraswamyConfig(CEMConfig):
    """
    kappa=a+b to initialize beta distribution 
    """
    kappa: float = 10.

    def __post_init__(self):
        super().__post_init__()
        # beta distribution
        self.sampler = "kumaraswamy"
        # beta samples in [0,1]
        # has to be linear scaling to [q_min, q_max]
        self.scaling = "linear"

class CEMKumaraswamy(CEMBeta):
    def get_a_b_from_mode_and_kappa(self, kappa=6):

        m = self._get_mode_in_unit_interval()
        p = self.kappa

        a_list, b_list = [], []

        for mi in m:
            if not (0.0 < mi < 1.0) or p <= 1.0:
                a_list.append(np.nan)
                b_list.append(np.nan)
                continue

            def f(a):
                return a - 1.0 - (p - 1.0) * (mi ** a)

            # Bracket the root
            a_low = 1.0 + 1e-10
            a_high = 1e2

            a_root = brentq(f, a_low, a_high, xtol=1e-10, rtol=1e-10, maxiter=200)
            b_root = p / a_root

            a_list.append(a_root)
            b_list.append(b_root)

        a = np.tile(a_list, self.nlp.Nknots)
        b = np.tile(b_list, self.nlp.Nknots)
        return a, b
    
    def compute_mean_cov_scaled(self, a, b):
        """
        Compute mean and variance of beta(a, b).
        Rescaled to the joint range.
        """
        mean = self.sampler.moment_n(a, b, 1)
        mean_q = self.f_rescale(mean).reshape(-1)

        var = self.sampler.moment_n(a, b, 2) - mean**2
        var = var.reshape(-1, self.nlp.Nu)
        # Var[a*X] = a**2 * X
        var *= self.q_range**2
        var = var.reshape(-1)
        diag_cov = np.diag(var)

        return mean_q, diag_cov
    
    def update_distrib_param(self, state: CEMBetaState, elites: Array):
        a, b, Sigma = self.sampler.estimate_params(elites)
        # Update state with weighted geometric average
        w1, w2 = self.alpha_mean, 1 - self.alpha_mean
        # state.a = a**w1 * state.a**w2
        # state.b = b**w1 * state.b**w2
        state.a += self.alpha_mean * (np.clip(a, 1e-1, 1e2) - state.a)
        state.b += self.alpha_mean * (np.clip(b, 1e-1, 1e2) - state.b)
        state.Sigma += self.alpha_cov * (Sigma - state.Sigma)
        # Update mean and cov for plotting
        state.mean, state.cov = self.compute_mean_cov_scaled(a, b)
