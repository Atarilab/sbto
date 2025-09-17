import numpy as np
import mujoco
from mujoco import rollout
from typing import Tuple, Union, Callable, TypeAlias, List
from sbto.mj.nlp_base import NLPBase, Array
import copy
from multiprocessing import cpu_count

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
            self.mj_model.nq, # +1 for time 
            self.mj_model.nv,
            self.mj_model.nu,
            T,
            Nknots,
            interp_kind
            )
        
        if Nthread == -1:
            self.Nthread = cpu_count()
        else:
            self.Nthread = Nthread if cpu_count() > Nthread > 0 else 1
        print(f"Using {self.Nthread} threads for MuJoCo simulation.")

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
        self._chunk_size = 16
        self._persistent_pool = True

    def set_initial_state_from_keyframe(self, keyframe_name: str) -> None:
        keyframe = self.mj_model.keyframe(keyframe_name)
        x_p_0 = np.array(keyframe.qpos)
        x_v_0 = np.array(keyframe.qvel)
        x_0 = np.concatenate((x_p_0, x_v_0))
        self.set_initial_state(x_0)

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
    def get_state(model, data, nbatch=1):
        full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
        state = np.zeros((mujoco.mj_stateSize(model, full_physics),))
        mujoco.mj_getState(model, data, state, full_physics)
        return np.tile(state, (nbatch, 1))

    def _reset_data(self) -> None:
        for data in self.mj_datas:
            mujoco.mj_resetData(self.mj_model, data)

    def get_q_des_from_u_traj(self, act: Array) -> Array:
        action_scale = 0.5
        q_des = np.clip(
            self.a + action_scale * act * self.b,
            self.q_min,
            self.q_max
            )
        return q_des
    
    def add_state_cost(self, name, f, idx_state, ref_values = 0, weights = 1, terminal = False):
        if np.any(idx_state >= self.Nx):
            raise ValueError(f"Invalid state index. Above {self.Nx}.")
        # +1 for time in the state
        return super().add_state_cost(name, f, idx_state+1, ref_values, weights, terminal)

    def _rollout_dynamics(self, u_traj: Array) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control trajecotries [-1, T, Nu].
        Returns state [-1, T, Nu], control [-1, T, Nu] and observations [-1, T, Nobs] trajectories.
        """
        if self.N_allocated != u_traj.shape[0]:
            self._init_batches(u_traj.shape[0])
        else:
            self._reset_data()

        rollout.rollout(self.mj_models,
                        self.mj_datas,
                        self.initial_states,
                        control=self.get_q_des_from_u_traj(u_traj),
                        nstep=self.T,
                        state=self.state_rollout,
                        sensordata=self.sensordata_rollout, 
                        skip_checks=False,
                        persistent_pool=self._persistent_pool,
                        chunk_size=self._chunk_size
                        )
        return self.state_rollout, u_traj, self.sensordata_rollout