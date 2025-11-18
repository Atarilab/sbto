import numpy as np
import numpy.typing as npt
from mujoco import rollout
from typing import Tuple, Optional
import copy
from dataclasses import dataclass
from multiprocessing import cpu_count

from sbto.sim.sim_base import SimRolloutBase, Array
from sbto.sim.scene_mj import MjScene
from sbto.sim.action_scaling import Scaling

IntArray = npt.NDArray[np.int64]

@dataclass
class ConfigMjRollout():
    """
    Configuration class for defining simulation rollout parameters.

    Attributes
    ----------
    T : int
        Number of control simulation steps per rollouts.
    
    Nknots : int
        Number of control or interpolation knots used in trajectory generation.
    
    keyframe_x0 : str
        Name of a keyframe_x0 to use for initializing simulation state.

    interp_kind : str
        Type of interpolation to use between knots (e.g. "linear", "cubic").
    
    scaling_kind : str
        Type of scaling to rescale samples to joint range.
    
    Nthread : int
        Number of thread for parallel rollouts (defaullt to cpu_count).
    """
    T: int
    Nknots: int = 0
    keyframe_x0: str = ""
    interp_kind: str = "linear"
    scaling_kind: str = ""
    Nthread: int = -1
    _chunk_size: int = 2
    _target_: str = "sbto.sim.sim_mj_rollout.SimMjRollout"

    # def __post_init__(self):
    #     self._filename = "config_sim_rollout.yaml"
    #     self.class_path = "sbto.sim.sim_mj_rollout.SimRolloutMj"

class SimMjRollout(SimRolloutBase):
    def __init__(
        self,
        mj_scene: MjScene,
        cfg: ConfigMjRollout,
        scaling: Optional[Scaling] = None
        ):

        self.cfg = cfg
        self.mj_scene = mj_scene
        super().__init__(
            self.mj_scene.Nq,
            self.mj_scene.Nv,
            self.mj_scene.Nu,
            cfg.T,
            cfg.Nknots,
            cfg.interp_kind,
            scaling,
            )
        
        # Set initial state
        if cfg.keyframe_x0:
            self.set_initial_state_from_keyframe(cfg.keyframe_x0)

        # Set actuator limits
        self.set_act_limits(
            self.mj_scene.q_min,
            self.mj_scene.q_max,
            self.x_0[self.mj_scene.act_joint_ids], # Nominal is x_0 by default
        )

        if cfg.Nthread == -1:
            self.Nthread = cpu_count()
        else:
            self.Nthread = cfg.Nthread if cpu_count() > cfg.Nthread > 0 else cpu_count()


        # preallocate results
        self.mj_models = None
        self.mj_datas = None
        self.initial_states : Array = None
        self.state_rollout : Array = None
        self.sensordata_rollout : Array = None
        self.initial_warmstart : Array = None
        self.N_allocated = -1
        self.T_allocated = -1
        # mujoco rollout variables
        self._chunk_size = cfg._chunk_size
        self._persistent_pool = True

    @property
    def duration(self):
        return self.mj_scene.dt * self.T
    
    def set_act_limits(self, q_min, q_max, q_nom = None):
        if q_nom is None and not np.all(self.x_0 == 0.):
            q_nom = self.x_0[self.mj_scene.act_qposadr]
        super().set_act_limits(q_min, q_max, q_nom)

    def set_initial_state_from_keyframe(self, keyframe_name: str, with_obj: bool = False) -> None:
        keyframe = self.mj_scene.mj_model.keyframe(keyframe_name)
        
        if not with_obj:
            x_p_0 = self.mj_scene.mj_data.qpos
            x_v_0 = self.mj_scene.mj_data.qvel
            qpos_adr = self.mj_scene.act_qposadr
            qvel_adr = self.mj_scene.act_dofadr

            # If floating base, add base pose
            if self.mj_scene.is_floating_base:
                qpos_base = np.arange(qpos_adr[0])
                qvel_base = np.arange(qvel_adr[0])
                qpos_adr = np.concatenate((qpos_base, qpos_adr))
                qvel_adr = np.concatenate((qvel_base, qvel_adr))

            # obj pose is not considered
            x_p_0[qpos_adr] = np.array(keyframe.qpos)[qpos_adr]
            x_v_0[qvel_adr] = np.array(keyframe.qvel)[qvel_adr]
        else:
            x_p_0 = np.array(keyframe.qpos)
            x_v_0 = np.array(keyframe.qvel)

        # Update data so that it can be used in taks definition
        self.mj_scene.update_data(x_p_0, x_v_0)

        # Set initial rollout state
        x_0 = np.concatenate((x_p_0, x_v_0))
        self.set_initial_state(x_0)

    def _init_batches(self, N: int, T:int) -> None:
        self.N_allocated = N
        self.T_allocated = T
        self.mj_models = [self.mj_scene.mj_model] * self.N_allocated
        self.mj_datas = [copy.copy(self.mj_scene.mj_data) for _ in range(self.Nthread)]
        t0 = [0.]
        # [N, Nx+1], include time as the first state
        self.initial_states = np.tile(np.concatenate((t0, self.x_0)), (self.N_allocated, 1))
        # [N, T, Nx+1]
        self.state_rollout = np.empty((self.N_allocated, T, self.Nx+1))
        # [N, T, Nobs]
        self.sensordata_rollout = np.empty((self.N_allocated, T, self.mj_scene.Nobs))

    def _rollout_dynamics(self, u_traj: Array, with_x0) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control trajecotries u_traj [-1, T, Nu].
        Returns state [-1, T, Nu], control [-1, T, Nu] and observations [-1, T, Nobs] trajectories.
        """
        N, T, Nu = u_traj.shape
        if self.N_allocated != N or self.T_allocated != T:
            self._init_batches(N, T)

        rollout.rollout(self.mj_models,
                        self.mj_datas,
                        self.initial_states,
                        control=u_traj,
                        nstep=T,
                        initial_warmstart=self.initial_warmstart,
                        state=self.state_rollout,
                        sensordata=self.sensordata_rollout, 
                        skip_checks=True,
                        persistent_pool=self._persistent_pool,
                        chunk_size=self._chunk_size
                        )
        
        if with_x0:
            x_traj_full = np.concatenate((self.initial_states[:, None, :], self.state_rollout), axis=1)
            return (
            x_traj_full[:, :, :1],
            x_traj_full[:, :, 1:],
            u_traj,
            self.sensordata_rollout
        )

        return (
            self.state_rollout[:, :, :1],
            self.state_rollout[:, :, 1:],
            u_traj,
            self.sensordata_rollout
        )
    
    def rollout_multiple_shooting(self, u_knots : Array, x_shooting: Array, with_x0: bool = False) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control knots u_knots [-1, Nknots, Nu].
        Interpolate and rescale the knots to the desired joint range.
        """
        u_knots = u_knots.reshape(-1, self.Nknots, self.Nu)
        if self.scaling:
            u_knots = self.scaling(u_knots)
        u_traj = self.interpolate(u_knots)
        
        t_full = []
        x_full = []
        u_full = []
        obs_full = []

        for t_start, t_end, x_start in zip(self.t_knots[:-1], self.t_knots[1:], x_shooting):
            # Reset initial state and data
            self.T_allocated = -1
            self.set_initial_state(x_start)

            if len(t_full) > 0:
                with_x0 = False

            t, x, u, obs = self._rollout_dynamics(u_traj[:, t_start: t_end, :], with_x0)

            t_full.append(t + t_start * self.mj_scene.dt)
            x_full.append(x)
            u_full.append(u)
            obs_full.append(obs)
            
            self.initial_warmstart = np.tile(self.mj_datas[0].qacc_warmstart, (self.N_allocated, 1))

        t_full = np.concatenate(t_full, axis=1)
        x_full = np.concatenate(x_full, axis=1)
        u_full = np.concatenate(u_full, axis=1)
        obs_full = np.concatenate(obs_full, axis=1)

        self.set_initial_state(x_shooting[0])
        self.T_allocated = -1
        self.initial_warmstart = None
        
        return t_full, x_full, u_full, obs_full
        
