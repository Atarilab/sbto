import os
import numpy as np
from sbto.mj.nlp_mj import NLP_MuJoCo
import sbto.tasks.unitree_g1.g1_constants as const
from sbto.utils.gait import humanoid_trot, generate_contact_plan

class G1_Gait(NLP_MuJoCo):
    SCENE = "scene_mjx_23dof_custom_collisions.xml"
    def __init__(self,
                 T,
                 Nknots = 0,
                 interp_kind="linear",
                 Nthread = -1
                 ):
        xml_path = os.path.join(const.XML_DIR_PATH, G1_Gait.SCENE)
        super().__init__(xml_path, T, Nknots, interp_kind, Nthread)

        keyframe_name = "knees_bent"
        self.set_initial_state_from_keyframe(keyframe_name)

        self.q_min = np.array(const.RESTRICTED_JOINT_RANGE)[:, 0]
        self.q_max = np.array(const.RESTRICTED_JOINT_RANGE)[:, 1]

        idx_joint_pos = np.arange(7, 7 + 23)
        idx_joint_vel = np.arange(36, 36 + 23)
        self.q_nom = self.x_0[idx_joint_pos]
        self.a_min = self.q_nom - self.q_min
        self.a_max = self.q_max - self.q_nom

        self.v_des = np.array([0.5, 0., 0.])

        self.add_state_cost(
            "joint_pos",
            self.quadratic_cost,
            idx_joint_pos,
            weights=0.,
            use_intial_as_ref=True,
            weights_terminal=50
            )
        self.add_state_cost(
            "joint_vel",
            self.quadratic_cost,
            idx_joint_vel,
            weights=0.001,
            )
        self.add_sensor_cost(
            const.G1Sensors.TORSO_POS,
            self.quadratic_cost,
            2,
            weights=5.,
            weights_terminal=50.,
            use_intial_as_ref=True
            )
        self.add_sensor_cost(
            const.G1Sensors.TORSO_POS,
            self.quadratic_cost,
            [0,1],
            ref_values=self.v_des[None, :2] * np.linspace(0., self.duration, num=T)[:self.T-1, None],
            weights=1.,
            ref_values_terminal=self.v_des[:2] * self.duration,
            weights_terminal=100.,
            )
        self.add_sensor_cost(
            const.G1Sensors.TORSO_LINVEL,
            self.quadratic_cost,
            ref_values=self.v_des,
            weights=[5., 5., 1.],
            ref_values_terminal=0.,
            weights_terminal=[0., 0., 50],
            )
        self.add_sensor_cost(
            const.G1Sensors.TORSO_ANGVEL,
            self.quadratic_cost,
            weights=1.,
            weights_terminal=10.,
            )
        self.add_sensor_cost(
            const.G1Sensors.TORSO_QUAT,
            self.quat_dist,
            weights=2.,
            weights_terminal=100.,
            use_intial_as_ref=True
            )
        
        self.contact_plan = generate_contact_plan(T, self.dt, humanoid_trot).repeat(const.cnt_sensor_per_foot, axis=-1)
        self.add_sensor_cost(
            const.G1Sensors.FEET_CONTACTS,
            self.contact_cost,
            sub_idx_sensor=const.G1Sensors.cnt_status_id,
            ref_values=self.contact_plan[:-1],
            ref_values_terminal=self.contact_plan[-1:],
            weights=15.,
        )
        self.add_sensor_cost(
            const.G1Sensors.FEET_CONTACTS,
            self.quadratic_cost,
            sub_idx_sensor=const.G1Sensors.cnt_force_id,
            weights=1.0e-5,
        )
        self.add_control_cost(
            "u_traj",
            self.quadratic_cost,
            idx=list(range(len(idx_joint_pos))),
            weights=0.05,
        )

    @staticmethod
    def contact_cost(cnt_status_rollout, cnt_plan, weights) -> float:
        cnt_status_rollout[cnt_status_rollout>1] = 1
        return np.sum(weights[None, ...] * np.float32(cnt_status_rollout != cnt_plan[None, ...]), axis=(-1, -2))
    
    @staticmethod
    def quat_dist(var, ref, weights) -> float:
        return np.sum(weights[:, 0] * (1.0 - np.square(np.sum(var * ref[None, ...], axis=-1))), axis=(-1))  # (N,)
    
    def get_q_des_from_u_traj(self, act):
        action_scale = 0.5
        act *= action_scale
        act = np.clip(act, -1, 1)        
        q_des = np.where(act < 0, act * self.a_min, act * self.a_max) + self.q_nom
        return q_des