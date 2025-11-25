from abc import ABC
from typing import Tuple, Union, Callable, TypeAlias, List, Optional
import numpy as np
import numpy.typing as npt
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

class OCPBase(ABC):
    def __init__(self, T: int):
        # Nuber of timesteps
        self.T = T

        # cost functions
        self._costs_names: List[str] = []
        self._costs_fn: List[CostFn] = []

    def _check_cost_fn(self, f: CostFn, ref_values: Array, weights: Array) -> None:
        if not callable(f):
            raise ValueError("Cost function should be callable")

    @staticmethod
    def _normalize_cost_array(
        arr: Union[Array, float],
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
        elif arr.shape == (T+1,):
            return np.tile(arr[:-1, None], (1, I))
        elif arr.shape == (T,):
            return np.tile(arr[:, None], (1, I))
        elif arr.shape == (T+1, I):
            return arr[:-1]
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
    def _extract_var(rollout_var: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """
        Efficiently extract the relevant slice from rollout_var without using np.take_along_axis.
        idx: expected shape (1, 1, I) or (I,)
        """
        # Flatten idx since it always targets the last axis
        idx_flat = np.ravel(idx)
        return rollout_var[:, :, idx_flat]

    def _add_cost(self,
                type: VarType,
                name: str,
                f: CostFn,
                idx: Union[IntArray, int],
                ref_values: Union[Array, float],
                weights: Union[Array, float],
                ) -> None:
        
        extractor = partial(self._extract_var, idx=idx)
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

        if name in self._costs_names:
            raise ValueError(f"Cost with name '{name}' already exists.")

        I = len(idx) if isinstance(idx, (list, np.ndarray)) else 1

        ref_values_r = self._normalize_cost_array(ref_values, self.T-1, I, name=f"ref_values of {name}")
        weights_r    = self._normalize_cost_array(weights,    self.T-1, I, name=f"weights of {name}")
        ref_values_t = self._normalize_cost_array(ref_values_terminal, 1, I, name=f"ref_values of {name}")
        weights_t    = self._normalize_cost_array(weights_terminal,    1, I, name=f"weights of {name}")

        ref_values = np.concatenate((ref_values_r, ref_values_t), axis=0)
        weights = np.concatenate((weights_r, weights_t), axis=0)

        if not np.all(weights == 0.):
            self._add_cost(
            type,
            name,
            f,
            idx,
            ref_values,
            weights,
            )

    @staticmethod
    def _type_cost(var_type: VarType):
        """
        Decorator factory to create add_*_cost methods for a given VarType.
        Injects the var_type while preserving signature and docstring.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(self: 'OCPBase', *args, **kwargs):
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

    def cost(self, x_traj : Array, u_traj : Array, obs_traj : Array) -> float:
        """
        Compute cost based on:
        - state trajectories [-1, T, Nu]
        - control trajectories [-1, T, Nu]
        - observations trajectories [-1, T, Nobs]
        """
        return sum(cost_fn(x_traj, u_traj, obs_traj) for cost_fn in self._costs_fn)