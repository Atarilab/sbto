from abc import ABC, abstractmethod
from typing import Tuple, Union, Callable, TypeAlias, List
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d
from typing import TypeAlias
from enum import Enum

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
            raise ValueError(
                f"{name} must have shape (I,), (T,), or (T, I), "
                f"but got {arr.shape} (T={T}, I={I})"
            )
        
    @staticmethod
    def _extract_var(rollout_var: Array, idx: IntArray, terminal: bool) -> Array:
        """
        Select the relevant slice from a variable array depending on terminal flag.
        """
        if terminal:
            return np.take_along_axis(rollout_var[:, -1:, :], idx, axis=-1)
        else:
            return np.take_along_axis(rollout_var, idx, axis=-1)

    def _add_cost(self,
                type: VarType,
                name: str,
                f: CostFn,
                idx: Union[IntArray, int],
                ref_values: Union[Array, float],
                weights: Union[Array, float],
                terminal: bool,
                ) -> None:
        
        I = len(idx) if isinstance(idx, (list, np.ndarray)) else 1
        T = 1 if terminal else self.T

        ref_values = self._normalize_cost_array(ref_values, T, I, name=f"ref_values of {name}")
        weights    = self._normalize_cost_array(weights,    T, I, name=f"weights of {name}")
        idx = np.asarray(idx, dtype=np.int32).reshape(1, 1, -1)
            
        match type:
            case VarType.STATE:
                cost_fn = lambda x, u, o: f(
                    self._extract_var(x, idx, terminal),
                    ref_values,
                    weights
                    )
            case VarType.CONTROL:
                cost_fn = lambda x, u, o: f(
                    self._extract_var(u, idx, terminal),
                    ref_values,
                    weights
                    )
            case VarType.OBS:
                cost_fn = lambda x, u, o: f(
                    self._extract_var(o, idx, terminal),
                    ref_values,
                    weights
                    )
            case _:
                raise ValueError(f"Unknown variable type: {type}")
        
        self._costs_fn.append(cost_fn)
        self._costs_names.append(name)

    def add_obs_cost(self,
                     name: str,
                     f: CostFn,
                     idx_obs: Union[IntArray, int],
                     ref_values: Union[Array, float] = 0.,
                     weights: Union[Array, float] = 1.,
                     terminal: bool = False,
                     ) -> None:
        self._add_cost(VarType.OBS, name, f, idx_obs, ref_values, weights, terminal)

    def add_state_cost(self,
                     name: str,
                     f: CostFn,
                     idx_state: Union[IntArray, int],
                     ref_values: Union[Array, float] = 0.,
                     weights: Union[Array, float] = 1.,
                     terminal: bool = False,
                     ) -> None:
        self._add_cost(VarType.STATE, name, f, idx_state, ref_values, weights, terminal)

    def add_control_cost(self,
                     name: str,
                     f: CostFn,
                     idx_u: Union[IntArray, int],
                     ref_values: Union[Array, float] = 0.,
                     weights: Union[Array, float] = 1.,
                     terminal: bool = False,
                     ) -> None:
        self._add_cost(VarType.CONTROL, name, f, idx_u, ref_values, weights, terminal)

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

    @abstractmethod
    def _rollout_dynamics(self, u_traj: Array) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control trajecotries [-1, T, Nu].
        Returns state [-1, T, Nu], control [-1, T, Nu] and observations [-1, T, Nobs] trajectories.
        """
        pass
