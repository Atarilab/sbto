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
    
    step_knots : int
        Number of sim step between two knots
    
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
    step_knots: int = 25
    keyframe_x0: str = ""
    interp_kind: str = "linear"
    scaling_kind: str = ""
    Nthread: int = -1
    _chunk_size: int = 2

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
            cfg.step_knots,
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
        self.mj_models = []
        self.mj_datas = []
        self.x_rollout : Array = None
        self.x_rollout_full : Array = None
        self.sensordata_rollout : Array = None
        self.sensordata_rollout_full : Array = None
        self.initial_warmstart : Array = None
        self.t0 = 0.
        self.steps_to_skip = 0
        self.N_allocated = -1
        self.last_T = -1
        self.nstep_allocated = -1
        # mujoco rollout variables
        self._chunk_size = cfg._chunk_size
        self._persistent_pool = True
        self._pending_steps_to_skip = 0

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

    def _allocate_data_arrays(self, N: int, nstep: int) -> None:

        if len(self.mj_datas) != self.Nthread:
            self.mj_datas = [copy.copy(self.mj_scene.mj_data) for _ in range(self.Nthread)]

        if N != self.N_allocated:
            self.mj_models = [self.mj_scene.mj_model] * N
            # [N, Nx+1], include time as the first state
            # [N, T+1, Nx+1]
            self.x_rollout_full = np.empty((N, self.T+1, self.Nx+1))
            # Set x_0
            x_0 = np.concatenate(([self.t0], self.x_0))[None, :]
            self.x_rollout_full[:, 0, :] = x_0
            # [N, T, Nobs]
            self.sensordata_rollout_full = np.empty((N, self.T, self.mj_scene.Nobs))

        # [N, T_eff, Nx+1]
        self.x_rollout = np.empty((N, nstep, self.Nx+1))
        # [N, T_eff, Nobs]
        self.sensordata_rollout = np.empty((N, nstep, self.mj_scene.Nobs))
            
        self.N_allocated = N
        self.nstep_allocated = nstep

    def skip_first_rollout_steps(self, knots_to_skip):
        """
        Skip the first rollout steps by taking the state and sensor data.
        """
        # Even though the first sampled knots are the same,
        # the interpolated controls between those knots might be
        # different as the later knots have different values.
        match self.interp_kind:
            case "pchip":
                offset_skip = 2
            case "linear":
                offset_skip = 0
            case _:
                offset_skip = 2
        knots_to_skip -= offset_skip

        if knots_to_skip > 0:
            # Do not immediately skip the rollout.
            # This ensures x_rollout_full[:, :t_knots[knots_to_skip], :]
            # is initialized with the right data before skipping it
            self.steps_to_skip = self.t_knots[knots_to_skip]
            self._pending_steps_to_skip = self.t_knots[knots_to_skip]

    def _rollout_dynamics(self, u_traj: Array, with_x0) -> Tuple[Array, Array, Array]:
        """
        Rollout the dynamics with the given control trajecotries u_traj [-1, T, Nu].
        Returns state [-1, T, Nu], control [-1, T, Nu] and observations [-1, T, Nobs] trajectories.
        """
        N, T, Nu = u_traj.shape
        nstep = int(T - self.steps_to_skip)

        if (
            self.N_allocated != N or
            self.nstep_allocated != nstep or
            self.last_T != T
            ):
            # Full horizon rollout when incrementing
            # to ensure consistency of the state rollouts
            # (due to numerical instabilities)
            if T != self.last_T and self.steps_to_skip > 0:
                self.steps_to_skip = 0
                nstep = T
            self._allocate_data_arrays(N, nstep)

        self.last_T = T

        rollout.rollout(self.mj_models,
                        self.mj_datas,
                        self.x_rollout_full[:, self.steps_to_skip, :],
                        control=u_traj[:, self.steps_to_skip:, :],
                        nstep=nstep,
                        initial_warmstart=None,
                        state=self.x_rollout,
                        sensordata=self.sensordata_rollout,
                        skip_checks=True,
                        persistent_pool=self._persistent_pool,
                        chunk_size=self._chunk_size
                        )

        self.x_rollout_full[:, self.steps_to_skip+1:T+1, :] = self.x_rollout
        self.sensordata_rollout_full[:, self.steps_to_skip:T, :] = self.sensordata_rollout

        # skip_first_rollout_steps needs to be called before each rollout
        self.steps_to_skip = 0
        first_timestep = 0 if with_x0 else 1

        return (
            self.x_rollout_full[:, first_timestep:T+1, :1],
            self.x_rollout_full[:, first_timestep:T+1, 1:],
            u_traj,
            self.sensordata_rollout_full[:, :T, :]
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
            self.last_T = -1
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
        self.last_T = -1
        self.initial_warmstart = None
        
        return t_full, x_full, u_full, obs_full
        
