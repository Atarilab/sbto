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

class OCPCostMod(OCPBase):
    def __init__(self, T: int):
        super().__init__(T)
        # Nuber of timesteps
        self.T = T

        # cost functions
        self._costs_names: List[str] = []
        self._costs_fn: List[CostFn] = []

        self.mod = np.ones(T)

    def update_mod(self, mod):
        self.mod = mod.copy()
        # Normalize
        self.mod /= np.sum(self.mod)
        self.mod *= self.T
        
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
            VarType.STATE: lambda x, u, o, m: f(extractor(x), ref_values, weights, m),
            VarType.CONTROL: lambda x, u, o, m: f(extractor(u), ref_values, weights, m),
            VarType.OBS: lambda x, u, o, m: f(extractor(o), ref_values, weights, m),
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
        return sum(cost_fn(x_traj, u_traj, obs_traj, self.mod) for cost_fn in self._costs_fn)
    
    def _cost_debug(self, x_traj : Array, u_traj : Array, obs_traj : Array, decay : Array, N_print: int = 1) -> float:
        """
        Compute cost based on:
        - state trajectories [-1, T, Nu]
        - control trajectories [-1, T, Nu]
        - observations trajectories [-1, T, Nobs]
        """
        N, T_traj, _ = u_traj.shape
        total = np.zeros(N)
        keep_terminal = T_traj == self.T and decay[-1] != 0.
        print(T_traj, self.T, keep_terminal)
        for name, f in zip(self._costs_names, self._costs_fn):
            if not keep_terminal and "terminal" in name:
                continue 
            c = f(x_traj, u_traj, obs_traj, decay)
            total += c
            print(name)
            print(c[:N_print])

        print("--- total", total[:N_print])
        return total