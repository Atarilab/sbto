from abc import ABC, abstractmethod
from typing import Tuple, Union, Callable, TypeAlias, List, Optional
import time
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from typing import TypeAlias
from enum import Enum
from functools import wraps, partial

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
CostFn: TypeAlias = Callable[[Tuple[Array, Array, Array]], float]

class VarType(Enum):
    STATE = 0
    CONTROL = 1
    OBS = 2

class NLPBase(ABC):
    def __init__(
        self,
        Nq: int,
        Nv: int,
        Nu: int,
        T: int,
        Nknots: int = 0,
        interp_kind = "linear"
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

        # inital state
        self.x_0 = np.zeros((self.Nx,))

        # cost functions
        self._costs_names: List[str] = []
        self._costs_fn: List[CostFn] = []

    def _check_state_array_shape(self, x: Array) -> None:
        valid_shape = (self.Nx,)
        if x.shape != valid_shape:
            raise ValueError(f"x should be a {valid_shape} vector, got {x.shape}")
        
    def set_initial_state(self, x_0: Array) -> None:
        self._check_state_array_shape(x_0)
        self.x_0 = x_0
        
    def _check_cost_fn(self, f: CostFn, ref_values: Array, weights: Array) -> None:
        if not callable(f):
            raise ValueError("Cost function should be callable")

    @staticmethod
    def _normalize_cost_array(arr: Union[Array, float],
                            T: int,
                            I: int,
                            *,
                            name: str) -> Array:
        """
        Normalize a cost array into shape (T, I).

        Cases handled:
        - scalar -> fill with scalar
        - shape (I,) -> repeat across T (-> shape (T, I))
        - shape (T,) -> repeat across I (-> shape (T, I))
        - shape (T, I) -> use as-is
        Otherwise: raise ValueError
        """
        if np.isscalar(arr):
            return np.full((T, I), arr, dtype=np.float64)

        arr = np.asarray(arr, dtype=np.float64)

        if arr.shape == (I,):
            return np.tile(arr[None, :], (T, 1))
        elif arr.shape == (T,):
            return np.tile(arr[:, None], (1, I))
        elif arr.shape == (T, I):
            return arr
        else:
            if T == 1:
                raise ValueError(
                    f"{name} must have shape (I,) "
                    f"but got {arr.shape} (T={T}, I={I})"
                )
            else:
                raise ValueError(
                    f"{name} must have shape (I,), (T-1,), or (T-1, I), "
                    f"but got {arr.shape} (T-1={T}, I={I})"
                )
        
    @staticmethod
    def _get_terminal_values(
        arr: Union[Array, float],
        I: int,
        ) -> None:
        if np.isscalar(arr):
            return arr
        arr = np.asarray(arr, dtype=np.float64)
        if len(arr.shape) == 1:
            # Shape [I]
            if arr.shape[-1] == I:
                return arr
            # Shape [T], take last element
            else:
                return arr[-1]
        # Shape [T, I]
        # Take last column
        elif len(arr.shape) == 2 and arr.shape[-1] == I:
            return arr[-1:, :]
        else:
            raise ValueError(
                f"Invalid array shape {arr.shape}."
            )
        
    @staticmethod
    def _extract_var(rollout_var: Array, idx: IntArray, terminal: bool) -> Array:
        """
        Select the relevant slice from a variable array depending on terminal flag.
        """
        if terminal:
            return np.take_along_axis(rollout_var[:, -1:, :], idx, axis=-1)
        else:
            return np.take_along_axis(rollout_var[:, :-1, :], idx, axis=-1)

    def _add_cost(self,
                type: VarType,
                name: str,
                f: CostFn,
                idx: Union[IntArray, int],
                ref_values: Union[Array, float],
                weights: Union[Array, float],
                terminal: bool,
                ) -> None:
        TERMINAL_STR = "_terminal"

        if terminal and name in self._costs_names:
            name = name + TERMINAL_STR
        if name in self._costs_names:
            raise ValueError(f"Cost with name '{name}' already exists.")

        I = len(idx) if isinstance(idx, (list, np.ndarray)) else 1
        T = 1 if terminal else self.T-1

        ref_values = self._normalize_cost_array(ref_values, T, I, name=f"ref_values of {name}")
        weights    = self._normalize_cost_array(weights,    T, I, name=f"weights of {name}")
        idx = np.asarray(idx, dtype=np.int32).reshape(1, 1, -1)

        extractor = partial(self._extract_var, idx=idx, terminal=terminal)
        mapping = {
            VarType.STATE: lambda x, u, o: f(extractor(x), ref_values, weights),
            VarType.CONTROL: lambda x, u, o: f(extractor(u), ref_values, weights),
            VarType.OBS: lambda x, u, o: f(extractor(o), ref_values, weights),
        }
        self._costs_fn.append(mapping[type])
        self._costs_names.append(name)

    def _add_cost_and_terminal_cost(
        self,
        type: VarType,
        name: str,
        f: CostFn,
        idx: Union[IntArray, int],
        ref_values: Union[Array, float] = 0.,
        weights: Union[Array, float] = 1.,
        ref_values_terminal: Optional[Union[Array, float]] = None,
        weights_terminal: Optional[Union[Array, float]] = None,
        ) -> None:
        I = len(idx) if isinstance(idx, (list, np.ndarray)) else 1
        if ref_values_terminal is None:
            ref_values_terminal = self._get_terminal_values(ref_values, I)
        if weights_terminal is None:
            weights_terminal = self._get_terminal_values(weights, I)

        if not np.all(weights == 0.):
            self._add_cost(type, name, f, idx, ref_values, weights, False)
        if not np.all(weights_terminal == 0.):
            self._add_cost(type, name, f, idx, ref_values_terminal, weights_terminal, True)

    @staticmethod
    def _type_cost(var_type: VarType):
        """
        Decorator factory to create add_*_cost methods for a given VarType.
        Injects the var_type while preserving signature and docstring.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self: 'NLPBase', *args, **kwargs):
                return self._add_cost_and_terminal_cost(var_type, *args, **kwargs)
            return wrapper
        return decorator

    @_type_cost(VarType.CONTROL)
    def add_control_cost(
        self,
        name: str,
        f: CostFn,
        idx_u: Union[IntArray, int],
        ref_values: Union[Array, float] = 0.,
        weights: Union[Array, float] = 1.,
        ref_values_terminal: Optional[Union[Array, float]] = None,
        weights_terminal: Optional[Union[Array, float]] = None,
    ) -> None:
        """Add a control cost with optional terminal component."""

    @_type_cost(VarType.STATE)
    def add_state_cost(
        self,
        name: str,
        f: CostFn,
        idx_x: Union[IntArray, int],
        ref_values: Union[Array, float] = 0.,
        weights: Union[Array, float] = 1.,
        ref_values_terminal: Optional[Union[Array, float]] = None,
        weights_terminal: Optional[Union[Array, float]] = None,
    ) -> None:
        """Add a state cost with optional terminal component."""

    @_type_cost(VarType.OBS)
    def add_obs_cost(
        self,
        name: str,
        f: CostFn,
        idx_o: Union[IntArray, int],
        ref_values: Union[Array, float] = 0.,
        weights: Union[Array, float] = 1.,
        ref_values_terminal: Optional[Union[Array, float]] = None,
        weights_terminal: Optional[Union[Array, float]] = None,
    ) -> None:
        """Add an observation cost with optional terminal component."""

    @staticmethod
    def quadratic_cost(var: Array, ref: Array, weights: Array) -> float:
        return np.sum(weights[None, ...] * (var - ref[None, ...]) ** 2, axis=(-1, -2))
    
    def cost(self, x_traj : Array, u_traj : Array, obs_traj : Array) -> float:
        """
        Compute cost based on:
        - state trajectories [-1, T, Nu]
        - control trajectories [-1, T, Nu]
        - observations trajectories [-1, T, Nobs]
        """
        return sum(cost_fn(x_traj, u_traj, obs_traj) for cost_fn in self._costs_fn)

    def interpolate(self, arr):
        """
        interpolate arr [-1, Nknots, Nu] in an array of shape [-1, T, Nu]
        """
        # Interpolate along each column
        f = interp1d(
            self.t_knots,
            arr,
            kind=self.interp_kind,
            copy=False,
            bounds_error=False,
            assume_sorted=True,
            axis=-2,
            )
        return f(self.t_all)
    
    def rollout(self, u_knots : Array) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control knots.
        Returns state, control and observations trajectories.
        """
        u_traj = self.interpolate(u_knots.reshape(-1, self.Nknots, self.Nu))
        return self._rollout_dynamics(u_traj)
    
    def get_rollout_data(self, knots_qd_traj: Array) -> Tuple[Array, Array, Array, float]:
        """
        Evaluate joint target trajectory and returns rollout data.
        Args:
            -qd_traj [N, Nknots*Nu]
        """
        knots_qd_traj = knots_qd_traj.reshape(-1, self.Nvars_u)
        x_traj, knots_qd_traj, obs_traj = self.rollout(knots_qd_traj)
        cost = self.cost(x_traj, knots_qd_traj, obs_traj)
        return (
            np.squeeze(x_traj),
            np.squeeze(knots_qd_traj),
            np.squeeze(obs_traj),
            np.squeeze(cost),
        )

    @abstractmethod
    def _rollout_dynamics(self, u_traj: Array) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control trajecotries [-1, T, Nu].
        Returns state [-1, T, Nu], control [-1, T, Nu] and observations [-1, T, Nobs] trajectories.
        """
        pass
