import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
from typing import Any, Tuple, Callable, Optional
from scipy.interpolate import interp1d
from functools import partial

from sbto.sim.action_scaling import Scaling

Array = npt.NDArray[np.float64]

class SimRolloutBase(ABC):
    def __init__(
        self,
        Nq: int,
        Nv: int,
        Nu: int,
        T: int,
        Nknots: int = 0,
        interp_kind: str = "linear",
        scaling: Optional[Scaling] = None,
        ):
        # problem size
        self.T = T
        self.Nq = Nq
        self.Nv = Nv
        self.Nu = Nu
        self.Nx = self.Nq + self.Nv
        self.Nknots = Nknots if T >= Nknots > 0 else T
        # x_0 not a decision variable
        self.Nvars_x = self.Nx * T
        self.Nvars_u = self.Nu * self.Nknots

        # spline interpolation
        self.interp_kind = interp_kind
        self.t_all = np.int32(np.linspace(0, 1, T, endpoint=True) * T)
        self.t_knots = np.int32(np.linspace(0, 1, Nknots, endpoint=True) * T)

        self.x_0 = np.zeros((self.Nx, ))
        # pd target scaling to joint range
        self.q_min = np.zeros((self.Nu, ))
        self.q_max = np.zeros((self.Nu, ))
        self.q_nom = np.zeros((self.Nu, ))
        self.q_range = np.ones_like(self.q_nom)
        # no scaling by default
        self.scaling = scaling

    def set_act_limits(self, q_min: Array, q_max: Array, q_nom: Optional[Array] = None):
        self.q_range = q_max - q_min
        if np.any(self.q_range < 0.):
            raise ValueError("Joint range should be positive.")
        self.q_min = q_min
        self.q_max = q_max
        self.q_nom = q_nom if q_nom is not None else (q_max - q_min) / 2.
        if self.scaling:
            self.scaling.set_range(self.q_min, self.q_max, self.q_nom)

    def _check_state_shape(self, x: Array) -> None:
        valid_shape = (self.Nx,)
        n = len(valid_shape)
        if x.shape[-n:] != valid_shape:
            raise ValueError(f"x should be a {valid_shape} vector, got {x.shape}")

    def _check_u_knots_shape(self, u_knots: Array) -> None:
        valid_shape = (self.Nknots, self.Nu)
        n = len(valid_shape)
        if u_knots.shape[-n:] != valid_shape:
            raise ValueError(f"u_knots should be a {valid_shape} vector, got {u_knots.shape}")
    
    def _check_u_traj_shape(self, u_traj: Array) -> None:
        valid_shape = (self.T, self.Nu)
        n = len(valid_shape)
        if u_traj.shape[-n:] != valid_shape:
            raise ValueError(f"u_traj should be a {valid_shape} vector, got {u_traj.shape}")

    def set_initial_state(self, x_0: Array) -> None:
        self.x_0[:] = x_0

    def interpolate(self, u_knots):
        """
        interpolate u_knots [-1, Nknots, Nu] in an array of shape [-1, T, Nu]
        """
        self._check_u_knots_shape(u_knots)
        # Interpolate along each column
        f = interp1d(
            self.t_knots,
            u_knots,
            kind=self.interp_kind,
            copy=False,
            bounds_error=False,
            assume_sorted=True,
            axis=-2,
            )

        return f(self.t_all)
    
    def rollout(self, u_knots : Array, with_x0: bool = False) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control knots u_knots [-1, Nknots, Nu].
        Interpolate and rescale the knots to the desired joint range.
        """
        u_knots = u_knots.reshape(-1, self.Nknots, self.Nu)
        if self.scaling:
            u_knots = self.scaling(u_knots) 
        u_traj = self.interpolate(u_knots)
        self._check_u_traj_shape(u_traj)
        return self._rollout_dynamics(u_traj, with_x0)
    
    def rollout_traj(self, u_traj : Array, with_x0: bool = False) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control trajecotries u_traj [-1, T, Nu].
        """
        self._check_u_traj_shape(u_traj)
        return self._rollout_dynamics(u_traj, with_x0)
    
    @abstractmethod
    def _rollout_dynamics(self, u_traj: Array, with_x0: bool = False) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control trajecotries [-1, T, Nu].
        Returns
            - time [T]
            - state [-1, T, Nx] or [-1, T+1, Nx] if with_x0
            - control [-1, T, Nu]
            - observations [-1, T, Nobs].
        """
        pass
