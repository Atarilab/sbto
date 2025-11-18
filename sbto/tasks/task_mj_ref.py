import numpy as np
from typing import Union, Optional, List

from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_mj import TaskMj, Array, CostFn, IntArray
from sbto.utils.extract_ref import ReferenceMotion

class TaskMjRef(TaskMj):
    def __init__(self,
                 sim: SimMjRollout,
                 ):
        super().__init__(sim)
        self.ref : ReferenceMotion = None

    def init_reference(
        self,
        ref_motion_path: str,
        t0: float = 0.,
        speedup: float = 1.0,
        z_offset: float = 0.0,
        ):
        self.ref = ReferenceMotion(
            ref_motion_path,
            self.mj_scene.cfg.xml_scene_path,
            t0,
            speedup,
            z_offset,
        )
        self.ref.extend_to_length(self.T)

    def _check_reference_is_set(self):
        if self.ref is None:
            raise Warning("ReferenceMotion has not been set. Call 'init_reference'")

    def add_state_cost_from_ref(
            self,
            name: str,
            f: CostFn,
            idx_x: Union[IntArray, int],
            weights: Union[Array, float] = 1.,
            weights_terminal: Optional[Union[Array, float]] = None,
        ) -> None:
        self._check_reference_is_set()

        idx_x = np.atleast_1d(idx_x)
        if np.any(idx_x >= self.mj_scene.Nx):
            raise ValueError(f"Invalid state index. Above {self.mj_scene.Nx}.")

        ref_values = self.ref.x[:self.T-1, idx_x]
        ref_values_terminal = self.ref.x[self.T-1, idx_x]

        super().add_state_cost(
            name,
            f,
            idx_x,
            ref_values,
            weights,
            ref_values_terminal,
            weights_terminal,
            )
    
    def add_sensor_cost_from_ref(
            self,
            sensor_name: Union[str, List[str]],
            f: CostFn,
            sub_idx_sensor: Union[IntArray, int] = -1,
            weights: Union[Array, float] = 1.,
            weights_terminal: Optional[Union[Array, float]] = None,
        ) -> None:
        self._check_reference_is_set()

        # Get sensordata idx
        idx_o = self.get_sensors_adr(sensor_name, sub_idx_sensor)

        # Set cost name and ref values
        if not isinstance(sensor_name, str):
            ref_values = np.empty((self.T-1, len(idx_o)))
            ref_values_terminal = np.empty((1, len(idx_o)))

            i = 0
            for sns_name in sensor_name:
                data = self.ref.data[sns_name]
                n_sns = data.shape[-1]
                ref_values[:, i:i+n_sns] = data[:self.T-1, :]
                ref_values_terminal[:, i:i+n_sns] = data[self.T-1, :]
                i += n_sns

            name = "+".join(sensor_name)
        else:
            name = sensor_name

            ref_values = self.ref.data[sensor_name][:self.T-1, :]
            ref_values_terminal = self.ref.data[sensor_name][self.T-1, :]

        name_suffix = '_'.join(map(str, idx_o.tolist()))
        name = name + '_' + name_suffix
        count_name = sum(1 for n in self._costs_names if name == n)
        name += '_' + str(count_name)

        super().add_obs_cost(
            name,
            f,
            idx_o,
            ref_values,
            weights,
            ref_values_terminal,
            weights_terminal,
            )