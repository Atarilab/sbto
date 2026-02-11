import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Union, Optional, List

from sbto.sim.sim_mj_rollout import SimMjRollout, MjScene
from sbto.tasks.task_mj import TaskMj, Array, CostFn, IntArray
from sbto.utils.extract_ref import ReferenceMotion

@dataclass
class ConfigRefMotion():
        motion_path: str
        t0: float = 0.
        t_end: float = 0.
        speedup: float = 1.0
        z_offset: float = 0.0
        flip_quat_pos: bool = True
        quat_wxyz: bool = True
        override_T_sim: bool = True

class TaskMjRef(TaskMj):
    def __init__(self,
                 sim: SimMjRollout,
                 cfg_ref: ConfigRefMotion,
                 mj_scene_ref: Optional[MjScene] = None
                 ):
        super().__init__(sim)
        self.mj_scene_ref = mj_scene_ref

        # Take mj_scene_ref model if exists
        # If the object in the scene is changed:
        # It computes the reference data with the original model
        self.ref = ReferenceMotion(
            self.mj_scene_ref if self.mj_scene_ref else self.mj_scene,
            cfg_ref.motion_path,
            cfg_ref.t0,
            cfg_ref.t_end,
            cfg_ref.speedup,
            cfg_ref.z_offset,
            cfg_ref.flip_quat_pos,
            cfg_ref.quat_wxyz,
        )

        # Update simulator (T, Nknots) parameters:
        # (The reference may be longer than sim.T)
        T_steps_ref = self.ref.T - 1
        super(SimMjRollout, sim).__init__(
            sim.mj_scene.Nq,
            sim.mj_scene.Nv,
            sim.mj_scene.Nu,
            T_steps_ref,
            sim.step_knots,
            sim.cfg.interp_kind,
            sim.scaling,
        )
        self.T = T_steps_ref

    def init_reference(
        self,
        ref_motion_path: str,
        t0: float = 0.,
        speedup: float = 1.0,
        z_offset: float = 0.0,
        ):
        mj_model = self.mj_scene_ref.mj_model if self.mj_scene_ref else self.mj_scene.mj_model
        self.ref = ReferenceMotion(
            ref_motion_path,
            mj_model,
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
        if self._are_weights_zero(weights, weights_terminal):
            return
        
        self._check_reference_is_set()

        idx_x = np.atleast_1d(idx_x)
        if np.any(idx_x >= self.mj_scene.Nx):
            raise ValueError(f"Invalid state index. Above {self.mj_scene.Nx}.")

        # initial state is not in the rollout data -> start from 1
        ref_values = self.ref.x[1:self.T, idx_x]
        ref_values_terminal = self.ref.x[self.T, idx_x]

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
        if self._are_weights_zero(weights, weights_terminal):
            return
        
        self._check_reference_is_set()

        # Get sensordata idx
        idx_o = self.get_sensors_adr(sensor_name, sub_idx_sensor)

        # Set cost name and ref values
        if not isinstance(sensor_name, str):
            ref_values = np.zeros((self.T-1, len(idx_o)))
            ref_values_terminal = np.zeros((1, len(idx_o)))

            i = 0
            for sns_name in sensor_name:
                data = self.ref.sensor_data[sns_name]
                n_sns = data.shape[-1]
                ref_values[:, i:i+n_sns] = data[1:self.T, :]
                ref_values_terminal[:, i:i+n_sns] = data[self.T, :]
                i += n_sns

            name = "+".join(sensor_name)
        else:
            name = sensor_name

            ref_values = self.ref.sensor_data[sensor_name][1:self.T, :]
            ref_values_terminal = self.ref.sensor_data[sensor_name][self.T, :]

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