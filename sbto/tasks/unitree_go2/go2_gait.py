
import os
import numpy as np
import mujoco
from sbto.mj.nlp_mj import NLP_MuJoCo
from sbto.utils.config import ConfigBase, dataclass
import numpy as np
import sbto.tasks.unitree_go2.go2_constants as GO2
from sbto.utils.gait import GaitConfig, generate_contact_plan, quad_trot

@dataclass
class ConfigGo2Gait(ConfigBase):
    T: int
    Nknots: int
    interp_kind: str = "linear"
    Nthread: int = -1
    scene: str = "scene_position.xml"
    contact_weight: float = 10 #Penalizes deviation between planned and achieved contact
    contact_weight_term: float = 10 #Additional penalty at the final timestep to ensure stable final stance
    #current best 10 for both
    contact_force_weight: float = 1e-5 #Penalize large ground forces/smooths contact transitions and avoids huge impulse/Between 1e-5 and 1e-3
    stance_ratio: list = (0.6, 0.6, 0.6, 0.6)  #removed stance and phase and cost already got better
    phase_offset: list = (0.5, 0.0, 0.0, 0.5) #Relative timing between legs
    nominal_period: float = 0.7 #pattern repeats every 0.5 seconds.


    def __post_init__(self):
        self._filename = "config_go2_gait.yaml"  
    
class Go2_Gait(NLP_MuJoCo):
    SCENE = "scene_position.xml"
    def __init__(self, cfg):
        
        T = cfg.T
        Nknots = cfg.Nknots
        interp_kind = cfg.interp_kind
        Nthread = cfg.Nthread

        xml_path = os.path.join(GO2.XML_DIR_PATH, cfg.scene)
        super().__init__(xml_path, T, Nknots, interp_kind, Nthread)
        #  initial state
        self.set_initial_state_from_keyframe("home")

        #  model sizes & indices 
        Nq = self.mj_model.nq # number of generalized positions
        Nv = self.mj_model.nv # number of generalized velocities
        base_q = 7 if self.mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0 # position state: 3:xyz + 4:quaternion(how it rotates)
        base_v = 6 if base_q == 7 else 0 # velocity state: 3:linear vel+3:angular vel 

        # joint qpos (after base) and qvel indices
        idx_joint_pos = np.arange(base_q, Nq)
        Nj = len(idx_joint_pos)
        idx_joint_vel = np.arange(Nq + base_v, Nq + base_v + Nj)

        #  joint limits
        qmin, qmax = [], []
        for j in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            lo, hi = self.mj_model.jnt_range[j]
            if lo == 0.0 and hi == 0.0:  
                lo, hi = -1.0, 1.0
            qmin.append(lo); qmax.append(hi)
        self.q_min = np.array(qmin, dtype=float)
        self.q_max = np.array(qmax, dtype=float)

        # nominal joint pose = initial
        self.q_nom = self.x_0[idx_joint_pos]
        self.a_min = self.q_nom - self.q_min
        self.a_max = self.q_max - self.q_nom

        #  gait targets (what we want the robot to do)
        self.v_des = np.array([0.5, 0.0, 0.0], dtype=float) #Goal: make the robot move forward at 0.5 m/s along X


        # time grid for position ramp
        t_grid = np.linspace(0.0, self.duration, num=T)[: self.T - 1]
        xy_ref = t_grid[:, None] * self.v_des[None, :2]  # (T-1, 2)

        idx_base_z = 2
        z0 = self.x_0[idx_base_z]
        z_ref = np.full(self.T - 1, z0, dtype=float) #keeps base height near the initial value so the robot doesn’t sag or jump

        # base orientation 
        idx_base_quat = np.arange(3, 7)
        quat_ref = np.tile(self.x_0[idx_base_quat], (self.T - 1, 1))  #keeps the robot upright (orientation quaternion fixed to initial)

        # base linear velocity indices 
        idx_base_linvel = np.arange(Nq, Nq + 3)

        #  costs 
        # joint pos near nominal 
        self.add_state_cost(
            "joint_pos", #Penalizes deviation of joint angles from self.q_nom (initial pose)
            self.quadratic_cost,
            idx_joint_pos,
            weights=5.0, # weight on joint position error
            use_intial_as_ref=True,
            weights_terminal=30.0, #more important at the end of the trajectory
        )  #if weight is high, legs won’t move much. If too low, legs can flop wildly

        #  joint vel damping
        self.add_state_cost(
            "joint_vel", #Penalizes joint velocities -> damps movement

            self.quadratic_cost,
            idx_joint_vel,
            weights=1e-2, 
            weights_terminal=1e-2 #better when both 1e-2

        )

        # base height tracking
        self.add_state_cost(
            "base_z",
            self.quadratic_cost,
            idx_base_z,
            ref_values=z_ref,
            weights=30.0, #	Strongly punishes the base height deviating from z_ref.

            weights_terminal=80.0,
        )

        #  base orientation near initial 
        self.add_state_cost(
            "base_quat", #Keeps the trunk upright (orientation close to initial).
            self.quadratic_cost,
            idx_base_quat,
            ref_values=quat_ref,
            weights=5.0,
            weights_terminal=50.0,
        )

        #  base linear velocity tracking (vx ~ 0.5 m/s)
        self.add_state_cost(
            "base_linvel", #Encourages the base velocity to match v_des (most importantly vx)
            self.quadratic_cost,
            idx_base_linvel,
            ref_values=self.v_des,            # broadcasted
            weights=[5.0, 1.0, 1.0], #The first weight  is on vx, so going forward is prioritized; y,z less so.
            weights_terminal=[0.0, 0.0, 50.0],
        )

        #  base XY position ramp (consistent with v_des)
        self.add_state_cost(
            "base_xy", #Tracks base X,Y position to follow the ramp produced by v_des.

            self.quadratic_cost,
            [0, 1],
            ref_values=xy_ref,          
            weights=[1.0, 1.0],
            weights_terminal=[30.0, 30.0], #Terminal weights huge → the optimizer strongly prefers ending near the expected final xy.
        )

        #  small control effort
        self.add_control_cost(
            "u_traj", #Penalizes big control commands (smooths actions, avoids thrashy solutions).
            self.quadratic_cost,
            idx=list(range(self.Nu)),
            weights=0.02,
        )
        # --- Contact plan ---

        gait = quad_trot 

        self.set_contact_sensor_id(
            GO2.Sensors.FEET_CONTACTS,
            GO2.Sensors.cnt_status_id
        )

        self.contact_plan = generate_contact_plan(cfg.T, self.dt, gait)
        self.contact_plan = self.contact_plan.repeat(GO2.cnt_sensor_per_foot, axis=-1)
        self.add_sensor_cost(
        GO2.Sensors.FEET_CONTACTS,
        self.contact_cost,
        sub_idx_sensor=GO2.Sensors.cnt_status_id,
        ref_values=self.contact_plan[:-1],
        ref_values_terminal=self.contact_plan[-1:],
        weights=cfg.contact_weight,
        weights_terminal=cfg.contact_weight_term,
        )

        self.add_sensor_cost(
        GO2.Sensors.BASE_QUAT,    
        self.quat_dist,
        weights=0.1,                
        weights_terminal=30.0,
        use_intial_as_ref=True,
        )
        
        self.add_sensor_cost( #Improve overall balance stability/Dampen sudden trunk rotations
        GO2.Sensors.BASE_ANGVEL,  
        self.quadratic_cost,
        weights=1.0,
        weights_terminal=10.0,
        )

    @staticmethod
    def contact_cost(cnt_status_rollout, cnt_plan, weights) -> float:
        cnt_status_rollout[cnt_status_rollout > 1] = 1
        return np.sum(
            weights[None, ...] * np.float32(cnt_status_rollout != cnt_plan[None, ...]),
            axis=(-1, -2),
        )

    @staticmethod
    def quat_dist(var, ref, weights) -> float:
        return np.sum(
            weights[:, 0] * (1.0 - np.square(np.sum(var * ref[None, ...], axis=-1))),
            axis=(-1),
        )
