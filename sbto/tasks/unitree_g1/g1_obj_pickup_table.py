import os
import numpy as np
from sbto.mj.nlp_mj import NLP_MuJoCo
import sbto.tasks.unitree_g1.g1_constants as G1
from sbto.mj.nlp_mj import ConfigNLP_Mj, dataclass
from sbto.utils.cost import quadratic_cost_nb, quaternion_dist_nb, hamming_dist_nb

@dataclass
class ConfigG1ObjPickupTable(ConfigNLP_Mj):
    # Scene
    scene_file: str = "scene_mjx_23dof_no_hands_obj_table.xml"

    # --- Joint reference ---
    keyframe_name: str = "knees_bent_wrist_yaw_90deg"

    # --- Randomize initial state ---
    scale_q: float = 0.05
    scale_v: float = 0.1
    upper_body_scale: float = 4.
    obj_x_range: tuple = (-0.01, 0.08)
    obj_y_range: tuple = (-0.04, 0.04)
    obj_w_range: tuple = (0.5, 0.5)

    # --- State costs ---
    joint_pos_weight: float = 0.1
    joint_pos_weight_terminal: float = 0.
    joint_vel_weight: float = 0.05
    joint_vel_weight_terminal: float = 0.5
    
    # --- Obj state goal ---
    obj_init_pos: tuple = (0.35, 0., 0.715)
    obj_delta_position: tuple = (0., 0., 0.1)
    obj_delta_orientation: tuple = (0., 0., 0.)
    reaching_cnt_time: float = 0.5
    delay_lift: float = 0.07

    # --- Obj state costs ---
    obj_pos_weight: float = 1.
    obj_pos_weight_terminal: float = (100., 100., 100.0)
    obj_quat_weight: float = 10.0
    obj_quat_weight_terminal: float = 20.0 
    obj_linvel_weight: float = 0.1
    obj_linvel_weight_term: float = 20.
    obj_angvel_weight: float = 0.1
    obj_angvel_weight_term: float = 10.
    
    # --- Torso position cost ---
    torso_pos_weight: float = (1., 1., 5.)
    torso_pos_weight_terminal: float = (1.0, 1.0, 50.0)

    # --- Torso linear velocity cost ---
    torso_linvel_weight: tuple = (1.0, 1.0, 1.0)
    torso_linvel_weight_terminal: tuple = (25.0, 25.0, 50.0)

    # --- Torso angular velocity cost ---
    torso_angvel_weight: float = 1.
    torso_angvel_weight_terminal: float = 1.

    # --- Torso orientation cost ---
    torso_quat_weight: float = 0.05
    torso_quat_weight_terminal: float = 5.0

    # --- Contact plan and cost ---
    contact_obj_weight: float = 15.
    contact_hands_weight: float = 10.
    contact_force_obj_weight: float = 1.0e-3
    contact_feet_weight: float = 1.
    contact_force_feet_weight: float = 1.0e-6

    # --- Control cost ---
    u_weight_default: float = 1.
    u_weight_hip_knee_scale: float = 1.
    u_weight_upperbody_scale: float = 0.1
    u_torques: float = 1.0e-5

class G1_ObjPickupTable(NLP_MuJoCo):

    def __init__(self, cfg: ConfigG1ObjPickupTable):
        xml_path = os.path.join(G1.XML_DIR_PATH, cfg.scene_file)
        super().__init__(xml_path, cfg.T, cfg.Nknots, cfg.interp_kind, cfg.Nthread)

        # --- Initial state setup ---
        self.set_initial_state_from_keyframe(cfg.keyframe_name)

        self.q_min = np.array(G1._25DoF_Obj.RESTRICTED_JOINT_RANGE)[:, 0]
        self.q_max = np.array(G1._25DoF_Obj.RESTRICTED_JOINT_RANGE)[:, 1]
        self.q_nom = self.x_0[G1._25DoF_Obj.IDX_JOINT_POS]

        # Initial state randomization
        self.keyframe_name = cfg.keyframe_name
        self.scale_q = cfg.scale_q
        self.scale_v = cfg.scale_v
        self.upper_body_scale = cfg.upper_body_scale
        self.obj_x_range = cfg.obj_x_range
        self.obj_y_range = cfg.obj_y_range
        self.obj_w_range = cfg.obj_w_range

        obj_position_0 = np.array(cfg.obj_init_pos)
        obj_position_goal = obj_position_0 + cfg.obj_delta_position
        # self.x_0[G1._25DoF_Obj.IDX_BOX_POS] = self.obj_position_0
        # self.set_initial_state(self.x_0)
        node_impact = int(cfg.reaching_cnt_time // self.dt)
        obj_position_ref = np.zeros((self.T, 3))
        obj_position_ref += obj_position_0
        t_ = np.arange(self.T - node_impact) * self.dt
        dir = obj_position_goal - obj_position_0
        obj_position_ref[node_impact:, :] += dir[None, ] * t_[:, None]

        # --- G1 costs ---
        self.add_state_cost(
            "joint_pos",
            quadratic_cost_nb,
            G1._25DoF_Obj.IDX_JOINT_POS,
            weights=cfg.joint_pos_weight,
            use_intial_as_ref=True,
            weights_terminal=cfg.joint_pos_weight_terminal,
        )
        # self.add_state_cost(
        #     "base_pos_xy",
        #     quadratic_cost_numba,
        #     [0, 1],
        #     weights=cfg.torso_pos_weight,
        #     use_intial_as_ref=True,
        #     weights_terminal=cfg.torso_pos_weight_terminal,
        # )
        self.add_state_cost(
            "joint_vel",
            quadratic_cost_nb,
            G1._25DoF_Obj.IDX_JOINT_VEL,
            weights=cfg.joint_vel_weight,
            weights_terminal=cfg.joint_vel_weight_terminal,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_POS,
            quadratic_cost_nb,
            weights=cfg.torso_pos_weight,
            weights_terminal=cfg.torso_pos_weight_terminal,
            use_intial_as_ref=True
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_LINVEL,
            quadratic_cost_nb,
            weights=cfg.torso_linvel_weight,
            weights_terminal=cfg.torso_linvel_weight_terminal,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_ANGVEL,
            quadratic_cost_nb,
            weights=cfg.torso_angvel_weight,
            weights_terminal=cfg.torso_angvel_weight_terminal,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_QUAT,
            quaternion_dist_nb,
            weights=cfg.torso_quat_weight,
            weights_terminal=cfg.torso_quat_weight_terminal,
            use_intial_as_ref=True,
        )
        # --- Obj cost ---
        self.add_state_cost(
            "obj_position",
            quadratic_cost_nb,
            G1._25DoF_Obj.IDX_BOX_POS,
            weights=cfg.obj_pos_weight,
            weights_terminal=cfg.obj_pos_weight_terminal,
            ref_values_terminal=obj_position_goal,
            use_intial_as_ref=True
        )
        self.add_state_cost(
            "obj_quat",
            quaternion_dist_nb,
            G1._25DoF_Obj.IDX_BOX_QUAT,
            weights=cfg.obj_quat_weight,
            weights_terminal=cfg.obj_quat_weight_terminal,
            use_intial_as_ref=True
        )
        self.add_state_cost(
            "obj_linvel",
            quadratic_cost_nb,
            G1._25DoF_Obj.IDX_BOX_LINVEL,
            weights=cfg.obj_linvel_weight,
            weights_terminal=cfg.obj_linvel_weight_term,
        )
        self.add_state_cost(
            "obj_angvel",
            quadratic_cost_nb,
            G1._25DoF_Obj.IDX_BOX_ANGVEL,
            weights=cfg.obj_angvel_weight,
            weights_terminal=cfg.obj_angvel_weight_term,
        )

        # --- Contact plan hands ---
        self.set_contact_sensor_id(G1.Sensors.HAND_CONTACTS, G1.Sensors.cnt_status_hand_id) # For plotting
        self.contact_plan = np.zeros((self.T, G1.N_HANDS), dtype=np.uint8)
        self.contact_plan[node_impact:] = 1.
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.cnt_status_hand_id,
            ref_values=self.contact_plan[:-1],
            ref_values_terminal=self.contact_plan[-1:],
            weights=cfg.contact_hands_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            quadratic_cost_nb,
            sub_idx_sensor=G1.Sensors.cnt_force_hand_id,
            weights=cfg.contact_force_obj_weight,
        )

        # --- Contact plan feet ---
        self.contact_plan_feet = np.full((self.T, G1.N_FEET * G1.cnt_sensor_per_foot), 1, dtype=np.uint8) # feet always in contact
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.cnt_status_feet_id,
            ref_values=self.contact_plan_feet[:-1],
            ref_values_terminal=self.contact_plan_feet[-1:],
            weights=cfg.contact_feet_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            quadratic_cost_nb,
            sub_idx_sensor=G1.Sensors.cnt_force_feet_id,
            weights=cfg.contact_force_feet_weight,
        )


        # --- Contact obj table ---
        self.contact_plan_obj = np.full((self.T, 1), 1, dtype=np.uint8) # feet always in contact
        node_lift_obj = node_impact + int(cfg.delay_lift // self.dt)
        self.contact_plan_obj[node_lift_obj:, :] = 0

        self.add_sensor_cost(
            G1.Sensors.OBJ_TABLE_CONTACT,
            hamming_dist_nb,
            sub_idx_sensor=[0],
            ref_values=self.contact_plan_obj[:-1],
            weights=cfg.contact_obj_weight,
        )

        # --- Control cost ---
        w_u_traj = np.full(self.Nu, cfg.u_weight_default)
        w_u_traj[G1._25DoF_Obj.IDX_HIP_KNEE] *= cfg.u_weight_hip_knee_scale
        w_u_traj[G1._25DoF_Obj.IDX_WAIST+1:] *= cfg.u_weight_upperbody_scale
        self.add_control_cost(
            "u_traj",
            quadratic_cost_nb,
            idx=list(range(self.Nu)),
            weights=w_u_traj,
        )

    @staticmethod
    def contact_cost(cnt_status_rollout, cnt_plan, weights) -> float:
        cnt_status_rollout[cnt_status_rollout > 1] = 1
        return np.sum(weights[None, ...] * np.float32(cnt_status_rollout != cnt_plan[None, ...]), axis=(-1, -2))

    @staticmethod
    def quat_dist(var, ref, weights) -> float:
        return np.sum(weights[:, 0] * (1.0 - np.square(np.sum(var * ref[None, ...], axis=-1))), axis=(-1))
    
    def are_initial_states_valid(self, states, obs):
        Z_MIN = 0.6
        QUAT_DIST_MAX = 0.4
        TORSO_XY_MAX_DIST = 0.07

        is_standing = states[:, 2] > Z_MIN

        torso_xyz = self.get_sensor_data(obs, G1.Sensors.TORSO_POS)
        is_centered = np.abs(torso_xyz[:, 0]) < TORSO_XY_MAX_DIST
        is_centered &= np.abs(torso_xyz[:, 1]) < TORSO_XY_MAX_DIST

        quat_ref = np.array([1., 0., 0., 0.]).reshape(1, 4)
        quat = states[:, 3:7].reshape(-1, 1, 4)
        w = np.full_like(quat_ref, 1.)
        quat_dist = quaternion_dist_nb(quat, quat_ref, w)
        is_straight = quat_dist < QUAT_DIST_MAX

        valid = is_straight & is_centered & is_standing
        return valid
    
    def randomize_initial_state(self):
        scale_q = np.full((self.Nq,), self.scale_q)
        scale_v = np.full((self.Nv,), self.scale_v)

        scale_q[:7] /= 10.
        scale_v[:6] /= 10.
        scale_q[-7:] = 0.
        scale_v[-6:] = 0.

        scale_q[G1._25DoF_Obj.IDX_WAIST+7:] *= self.upper_body_scale
        obj_qpos_id = G1._25DoF_Obj.IDX_BOX_POS + G1._25DoF_Obj.IDX_BOX_QUAT
        scale_q[obj_qpos_id] = 0.
        scale_v[-6:] = 0.

        return super().set_random_initial_state(
            self.keyframe_name,
            scale_q,
            scale_v,
            is_floating_base=True,
            obj_qpos_id=obj_qpos_id,
            N_rollout_steps=150,
            obj_x_range=self.obj_x_range,
            obj_y_range=self.obj_y_range,
            obj_w_range=self.obj_w_range,
            )
    