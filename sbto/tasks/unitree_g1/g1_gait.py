import os
import numpy as np
from sbto.mj.nlp_mj import NLP_MuJoCo
import sbto.tasks.unitree_g1.g1_constants as G1
from sbto.utils.gait import GaitConfig, generate_contact_plan
from sbto.mj.nlp_mj import ConfigNLP_Mj, dataclass

@dataclass
class ConfigG1Gait(ConfigNLP_Mj):
    # Scene
    scene_file: str = "scene_mjx_23dof_custom_collisions.xml"

    # --- Joint reference ---
    keyframe_name: str = "knees_bent"

    # --- Desired motion parameters ---
    v_des: tuple = (0.5, 0.0, 0.0)  # Desired torso linear velocity [vx, vy, vz]

    # --- Desired gait parameters ---
    stance_ratio: tuple = (0.55, 0.55)
    phase_offset: tuple = (0.5, 0.0)
    nominal_period = 0.9

    # --- State costs ---
    joint_pos_weight: float = 0.
    joint_pos_weight_terminal: float = 50.

    joint_vel_weight: float = 0.001

    # --- Torso position cost ---
    torso_pos_weight: float = 1.
    torso_pos_weight_terminal: float = 1000.0

    # --- Torso XY tracking cost ---
    torso_xy_weight: float = 10.
    torso_xy_weight_terminal: float = 100.0

    # --- Torso linear velocity cost ---
    torso_linvel_weight: tuple = (2.0, 2.0, 2.0)
    torso_linvel_weight_terminal: tuple = (10.0, 10.0, 20.0)

    # --- Torso angular velocity cost ---
    torso_angvel_weight: float = 1.0
    torso_angvel_weight_terminal: float = 10.0

    # --- Torso orientation cost ---
    torso_quat_weight: float = .1
    torso_quat_weight_terminal: float = 1000

    # --- Contact plan and cost ---
    contact_weight: float = 10.0
    contact_weight_term: float = 10.0
    contact_force_weight: float = 1.0e-5

    # --- Control cost ---
    u_weight_default: float = 1.
    u_weight_hip_knee_scale: float = 0.1
    u_weight_upperbody_scale: float = 2.0
    u_torques: float = 1.0e-5

    # --- Action scaling ---
    action_scale: float = 0.5


class G1_Gait(NLP_MuJoCo):

    def __init__(self, cfg: ConfigG1Gait):
        xml_path = os.path.join(G1.XML_DIR_PATH, cfg.scene_file)
        super().__init__(xml_path, cfg.T, cfg.Nknots, cfg.interp_kind, cfg.Nthread)

        # --- Initial state setup ---
        self.set_initial_state_from_keyframe(cfg.keyframe_name)

        self.q_min = np.array(G1.RESTRICTED_JOINT_RANGE)[:, 0]
        self.q_max = np.array(G1.RESTRICTED_JOINT_RANGE)[:, 1]

        self.q_nom = self.x_0[G1.IDX_JOINT_POS]
        self.a_min = self.q_nom - self.q_min
        self.a_max = self.q_max - self.q_nom

        self.v_des = np.array(cfg.v_des)

        # --- Add costs ---
        self.add_state_cost(
            "joint_pos",
            self.quadratic_cost,
            G1.IDX_JOINT_POS,
            weights=cfg.joint_pos_weight,
            use_intial_as_ref=True,
            weights_terminal=cfg.joint_pos_weight_terminal,
        )
        self.add_state_cost(
            "joint_vel",
            self.quadratic_cost,
            G1.IDX_JOINT_VEL,
            weights=cfg.joint_vel_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_POS,
            self.quadratic_cost,
            2,
            weights=cfg.torso_pos_weight,
            weights_terminal=cfg.torso_pos_weight_terminal,
            use_intial_as_ref=True,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_POS,
            self.quadratic_cost,
            [0, 1],
            ref_values=self.v_des[None, :2] * np.linspace(0., self.duration, num=cfg.T)[:self.T-1, None],
            weights=cfg.torso_xy_weight,
            ref_values_terminal=self.v_des[:2] * self.duration,
            weights_terminal=cfg.torso_xy_weight_terminal,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_LINVEL,
            self.quadratic_cost,
            ref_values=self.v_des,
            weights=cfg.torso_linvel_weight,
            ref_values_terminal=0.,
            weights_terminal=cfg.torso_linvel_weight_terminal,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_ANGVEL,
            self.quadratic_cost,
            weights=cfg.torso_angvel_weight,
            weights_terminal=cfg.torso_angvel_weight_terminal,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_QUAT,
            self.quat_dist,
            weights=cfg.torso_quat_weight,
            weights_terminal=cfg.torso_quat_weight_terminal,
            use_intial_as_ref=True,
        )

        # --- Contact plan ---
        gait = GaitConfig(
            G1.N_FEET,
            cfg.stance_ratio,
            cfg.phase_offset,
            cfg.nominal_period
            )
        self.set_contact_sensor_id(G1.Sensors.FEET_CONTACTS, G1.Sensors.cnt_status_id)
        self.contact_plan = generate_contact_plan(cfg.T, self.dt, gait)
        self.contact_plan = self.contact_plan.repeat(G1.cnt_sensor_per_foot, axis=-1)
        
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            self.contact_cost,
            sub_idx_sensor=G1.Sensors.cnt_status_id,
            ref_values=self.contact_plan[:-1],
            ref_values_terminal=self.contact_plan[-1:],
            weights=cfg.contact_weight,
            weights_terminal=cfg.contact_weight_term,
        )
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            self.quadratic_cost,
            sub_idx_sensor=G1.Sensors.cnt_force_id,
            weights=cfg.contact_force_weight,
        )

        # --- Control cost ---
        w_u_traj = np.full(self.Nu, cfg.u_weight_default)
        w_u_traj[list(G1.HIP_KNEE)] *= cfg.u_weight_hip_knee_scale
        w_u_traj[13:] *= cfg.u_weight_upperbody_scale
        self.add_control_cost(
            "u_traj",
            self.quadratic_cost,
            idx=list(range(self.Nu)),
            weights=w_u_traj,
        )
        self.add_sensor_cost(
            G1.Sensors.TORQUES,
            self.quadratic_cost,
            weights=cfg.u_torques
            )

        # --- Action scaling ---
        self.action_scale = cfg.action_scale

    @staticmethod
    def contact_cost(cnt_status_rollout, cnt_plan, weights) -> float:
        cnt_status_rollout[cnt_status_rollout > 1] = 1
        return np.sum(weights[None, ...] * np.float32(cnt_status_rollout != cnt_plan[None, ...]), axis=(-1, -2))

    @staticmethod
    def quat_dist(var, ref, weights) -> float:
        return np.sum(weights[:, 0] * (1.0 - np.square(np.sum(var * ref[None, ...], axis=-1))), axis=(-1))