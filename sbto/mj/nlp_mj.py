import numpy as np
import mujoco
from mujoco import rollout
from typing import Tuple, Union, Optional, List
import copy
from multiprocessing import cpu_count
from sbto.mj.nlp_base import NLPBase, Array, CostFn, IntArray
from sbto.utils.config import ConfigBase, dataclass

@dataclass
class ConfigNLP_Mj(ConfigBase):
    T: int
    Nknots: int
    interp_kind: str = "linear"
    Nthread: int = -1

    def __post_init__(self):
        self._filename = "config_nlp.yaml"

class NLP_MuJoCo(NLPBase):
    def __init__(
        self,
        xml_path: str,
        T: int,
        Nknots: int = 0,
        interp_kind = "linear",
        Nthread: int = -1,
        ):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        
        super().__init__(
            self.mj_model.nq,
            self.mj_model.nv,
            self.mj_model.nu,
            T,
            Nknots,
            interp_kind
            )
        
        if Nthread == -1:
            self.Nthread = cpu_count()
        else:
            self.Nthread = Nthread if cpu_count() > Nthread > 0 else cpu_count()
        print(f"Using {self.Nthread} threads for MuJoCo simulation.")

        self.dt = self.mj_model.opt.timestep
        self.duration = self.T * self.dt

        # Set actuator limits
        self.q_min = np.array(self.mj_model.jnt_range)[1:, 0]
        self.q_max = np.array(self.mj_model.jnt_range)[1:, 1]

        self.a = 0.5 * (self.q_min + self.q_max)[None, None, ...]
        self.b = 0.5 * (self.q_max - self.q_min)[None, None, ...]

        # preallocate results
        self.mj_models = None
        self.mj_datas = None
        self.initial_states : Array = None
        self.state_rollout : Array = None
        self.sensordata_rollout : Array = None
        self.N_allocated = -1
        self.Nobs = 0

        # rollout variables
        self._chunk_size = 2
        self._persistent_pool = True

        # contact sensors obs_id
        self.contact_obs_id : Array = None

    def set_initial_state_from_keyframe(self, keyframe_name: str) -> None:
        keyframe = self.mj_model.keyframe(keyframe_name)
        x_p_0 = np.array(keyframe.qpos)
        x_v_0 = np.array(keyframe.qvel)
        x_0 = np.concatenate((x_p_0, x_v_0))
        self.set_initial_state(x_0)
        self.mj_data.qpos = keyframe.qpos
        self.mj_data.qvel = keyframe.qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _init_batches(self, N: int) -> None:
        self.N_allocated = N
        self.Nobs = self.mj_model.nsensordata
        self.mj_models = [self.mj_model] * self.N_allocated
        self.mj_datas = [copy.copy(self.mj_data) for _ in range(self.Nthread)]
        t0 = [0.]
        # [N, Nx+1], include time as the first state
        self.initial_states = np.tile(np.concatenate((t0, self.x_0)), (self.N_allocated, 1))
        # [N, T, Nx+1]
        self.state_rollout = np.empty((self.N_allocated, self.T, self.Nx+1))
        # [N, T, Nobs]
        self.sensordata_rollout = np.empty((self.N_allocated, self.T, self.Nobs))

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
            sensor_id = self.mj_model.sensor(name).id
            sensor_adr = self.mj_model.sensor_adr[sensor_id]
            sensor_dim = self.mj_model.sensor_dim[sensor_id]
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
    
    def _reset_data(self) -> None:
        for data in self.mj_datas:
            mujoco.mj_resetData(self.mj_model, data)
    
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
        idx_x = np.asarray(idx_x)
        if np.any(idx_x >= self.Nx):
            raise ValueError(f"Invalid state index. Above {self.Nx}.")
        # +1 for time in the state
        idx_x = idx_x + 1
        if use_intial_as_ref:
            state = self.get_state_full(self.mj_model, self.mj_data)
            ref_values = state[idx_x]

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
            ref_values = self.mj_data.sensordata[idx_o]
            
        super().add_obs_cost(
            name,
            f,
            idx_o,
            ref_values,
            weights,
            ref_values_terminal,
            weights_terminal,
            )

    def _rollout_dynamics(self, pd_target_traj: Array) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control trajecotries [-1, T, Nu].
        Returns state [-1, T, Nu], control [-1, T, Nu] and observations [-1, T, Nobs] trajectories.
        """
        if self.N_allocated != pd_target_traj.shape[0]:
            self._init_batches(pd_target_traj.shape[0])
        else:
            self._reset_data()

        rollout.rollout(self.mj_models,
                        self.mj_datas,
                        self.initial_states,
                        control=pd_target_traj,
                        nstep=self.T,
                        state=self.state_rollout,
                        sensordata=self.sensordata_rollout, 
                        skip_checks=False,
                        persistent_pool=self._persistent_pool,
                        chunk_size=self._chunk_size
                        )
        return self.state_rollout, pd_target_traj, self.sensordata_rollout
    
    def get_sensor_data(
        self,
        obs: Array,
        sensor_names: str | List[str],
        sub_idx_sensor: int | List[int] = -1,
        ) -> Array:

        idx_o = self.get_sensors_adr(sensor_names, sub_idx_sensor)
        return obs[:, idx_o]
    
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