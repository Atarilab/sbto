import numpy as np
import mujoco
from typing import Tuple, Union, Optional, List

from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.sim.scene_mj import MjScene
from sbto.tasks.task_base import OCPBase, Array, CostFn, IntArray
from sbto.tasks.task_base_cost_decay import OCPCostDecay

class TaskMj(OCPCostDecay):
    def __init__(self, sim: SimMjRollout):
        super().__init__(sim.T)
        self.mj_scene : MjScene = sim.mj_scene

    @staticmethod
    def get_state_full(model, data):
        full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
        state = np.zeros((mujoco.mj_stateSize(model, full_physics),))
        mujoco.mj_getState(model, data, state, full_physics)
        return state
    
    def get_sensors_adr(self,
                        sensor_names: Union[str, list[str]],
                        sub_idx_sensor: Union[IntArray, int] = -1,
                        ) -> Array:
        """Gets sensor adr given one or multiple sensor names."""
        if isinstance(sensor_names, str):
            sensor_names = [sensor_names]
        adr = []
        for name in sensor_names:
            sensor_id = self.mj_scene.mj_model.sensor(name).id
            sensor_adr = self.mj_scene.mj_model.sensor_adr[sensor_id]
            sensor_dim = self.mj_scene.mj_model.sensor_dim[sensor_id]
            adr.extend(range(sensor_adr, sensor_adr + sensor_dim))
        sensor_idx = np.asarray(adr)

        # sub_idx_sensor is the index to consider among sensor_idx
        if sub_idx_sensor != -1:
            if isinstance(sub_idx_sensor, int):
                sub_idx_sensor = [sub_idx_sensor]
            
            sub_idx_sensor = np.asarray(sub_idx_sensor, dtype=np.int64)
            
            idx_o = np.take(sensor_idx, sub_idx_sensor)
        else:
            idx_o = sensor_idx

        return idx_o

    def add_state_cost(self,
                     name: str,
                     f: CostFn,
                     idx_x: Union[IntArray, int],
                     ref_values: Union[Array, float] = 0.,
                     weights: Union[Array, float] = 1.,
                     ref_values_terminal: Optional[Union[Array, float]] = None,
                     weights_terminal: Optional[Union[Array, float]] = None,
                     use_intial_as_ref: bool = False,
                     ) -> None:
        idx_x = np.atleast_1d(idx_x)
        if np.any(idx_x >= self.mj_scene.Nx):
            raise ValueError(f"Invalid state index. Above {self.mj_scene.Nx}.")

        if use_intial_as_ref:
            state = self.get_state_full(self.mj_scene.mj_model, self.mj_scene.mj_data)
            ref_values = state[idx_x+1] # state with time -> +1

        super().add_state_cost(
            name,
            f,
            idx_x,
            ref_values,
            weights,
            ref_values_terminal,
            weights_terminal,
            )
    
    def add_sensor_cost(self,
                        sensor_name: Union[str, List[str]],
                        f: CostFn,
                        sub_idx_sensor: Union[IntArray, int] = -1,
                        ref_values: Union[Array, float] = 0.,
                        weights: Union[Array, float] = 1.,
                        ref_values_terminal: Optional[Union[Array, float]] = None,
                        weights_terminal: Optional[Union[Array, float]] = None,
                        use_intial_as_ref: bool = False,
                        ) -> None:
        # Get sensordata idx
        idx_o = self.get_sensors_adr(sensor_name, sub_idx_sensor)

        # Set cost name
        if not isinstance(sensor_name, str):
            name = "+".join(sensor_name)
        else:
            name = sensor_name

        name_suffix = '_'.join(map(str, idx_o.tolist()))
        name = name + '_' + name_suffix
        count_name = sum(1 for n in self._costs_names if name == n)
        name += '_' + str(count_name)

        # Use sensor data values as reference
        if use_intial_as_ref:
            ref_values = self.mj_scene.mj_data.sensordata[idx_o]
            
        super().add_obs_cost(
            name,
            f,
            idx_o,
            ref_values,
            weights,
            ref_values_terminal,
            weights_terminal,
            )

    def are_initial_states_valid(self, states: Array, obs: Array) -> Array:
        """
        Checks which candidate initial states are valid.
        Args:
            state (Array): [N, Nx]
            obs (Array): [N, Nobs]
        Returns:
            valid (Array): [N], boolean array 
        """
        N = states.shape[0]
        all_valid = np.full(N, True)
        return all_valid

    def set_contact_sensor_id(
        self,
        cnt_sensor_names: str | List[str],
        cnt_sub_idx_sensor: int | List[int] = -1
        ) -> None:
        self.contact_obs_id = self.get_sensors_adr(cnt_sensor_names, cnt_sub_idx_sensor)
    
    def get_contact_status(
        self,
        obs_traj,
        ) -> Array:
        if self.contact_obs_id is None:
            print("Warning: self.contact_obs_id is not set.")
            return []
        return obs_traj[:, self.contact_obs_id]
    
    def get_sensor_data(
        self,
        obs: Array,
        sensor_names: str | List[str],
        sub_idx_sensor: int | List[int] = -1,
        ) -> Array:

        idx_o = self.get_sensors_adr(sensor_names, sub_idx_sensor)
        return obs[:, idx_o]