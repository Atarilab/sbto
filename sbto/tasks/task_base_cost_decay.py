from abc import ABC
from typing import Tuple, Union, Callable, TypeAlias, List, Optional
import numpy as np
import numpy.typing as npt
from typing import TypeAlias
from enum import Enum
from functools import wraps, partial

from .task_base import OCPBase, VarType

Array = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
CostFn: TypeAlias = Callable[[Tuple[Array, Array, Array]], float]

class OCPCostDecay(OCPBase):
    def __init__(self, T: int):
        super().__init__(T)
        # Nuber of timesteps
        self.T = T

        # cost functions
        self._costs_names: List[str] = []
        self._costs_fn: List[CostFn] = []

        self.decay = np.ones(T)

    def update_decay(self, decay):
        self.decay = decay

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

        extractor = partial(self._extract_var, idx=idx, terminal=terminal)
        mapping = {
            VarType.STATE: lambda x, u, o, d: f(extractor(x), ref_values, weights, d),
            VarType.CONTROL: lambda x, u, o, d: f(extractor(u), ref_values, weights, d),
            VarType.OBS: lambda x, u, o, d: f(extractor(o), ref_values, weights, d),
        }
        self._costs_fn.append(mapping[type])
        self._costs_names.append(name)

    def cost(self, x_traj : Array, u_traj : Array, obs_traj : Array) -> float:
        """
        Compute cost based on:
        - state trajectories [-1, T, Nu]
        - control trajectories [-1, T, Nu]
        - observations trajectories [-1, T, Nobs]
        - decay [T]: multiplier to each time step of the cost
        """
        return sum(cost_fn(x_traj, u_traj, obs_traj, self.decay) for cost_fn in self._costs_fn)