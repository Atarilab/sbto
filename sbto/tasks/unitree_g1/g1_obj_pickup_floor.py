import os
import numpy as np
from sbto.mj.nlp_mj import NLP_MuJoCo
import sbto.tasks.unitree_g1.g1_constants as G1
from sbto.mj.nlp_mj import ConfigTask, ConfigScene, dataclass
from sbto.utils.cost import quadratic_cost_nb, quaternion_dist_nb, hamming_dist_nb

@dataclass
class SceneG1ObjPickupFloor(ConfigScene):
    xml_path: str = "sbto/models/unitree_g1/scene_mjx_25dof_no_hands.xml"
    keyframe: str = "knees_bent_wrist_yaw_90deg"
    # --- Additional xml files ---
    sensor_file: str = "sbto/models/unitree_g1/sensors/with_obj.xml"
    contact_pair_file: str = "sbto/models/unitree_g1/contact_pairs/with_obj.xml"
    keyframes_file: str = "sbto/models/unitree_g1/keyframes/g1_25dof.xml"
    # --- Randomize initial state ---
    scale_q: float = 0.05
    scale_v: float = 0.1
    upper_body_scale: float = 5.
    obj_x_range: tuple = (-0.02, 0.03)
    obj_y_range: tuple = (-0.015, 0.015)
    obj_w_range: tuple = (-0.3, 0.3)
    body = {
        "obj": {
            "type":         "box",
            "pos":          (0.35, 0., 0.115),
            "size":         (0.1, 0.1, 0.115),
            "mass":         0.6,
            "euler":        (0., 0., 0.),
            "rgba":         (0.3, 0.3, 0.3, 1.),
            "group":        1,
            "freejoint":    True,
            "priority":     0,
            "condim":       3,
            "contype":      0,
            "conaffinity":  1,
            "solref":       (0.008, 1.),
            "friction":     (0.6, 0.003, 0.001),
        },
    }

@dataclass
class TaskG1ObjPickupFloor(ConfigTask):
    # Scene file
    xml_path: str = "sbto/models/unitree_g1/scene_mjx_25dof_no_hands.xml"

    # --- Timing ---
    squat_time: float = 1.     # seconds to reach squat pose
    pickup_time: float = 0.2   # seconds to reach the box after squatting
    standup_time: float = 0.1  # seconds to reach the box after pickup

    # -- Obj Goal ---
    obj_init_pos: tuple = (0.35, 0., 0.115)
    obj_delta_position: tuple = (0., 0., 0.7)

    obj_pos_weight: float = 0.
    obj_pos_weight_terminal: float = (10.0, 10.0, 10.0)
    obj_quat_weight: float = 0.6
    obj_quat_weight_terminal: float = 0.6

    # --- Weights ---
    joint_pos_weight: float = 0.2
    joint_pos_weight_terminal: float = 2e2
    joint_vel_weight: float = 1e-2
    joint_vel_weight_terminal: float = 0.1

    base_height_weight: float = 2.
    base_height_weight_terminal: float = 75.0

    base_quat_weight: float = 3.
    base_quat_weight_terminal: float = 50.0

    torso_pos_weight: tuple = (2., 2., 1.)
    torso_pos_weight_terminal: tuple = (50.0, 50.0, 30.0)

    obj_linvel_weight: float = 1.0e-3
    obj_angvel_weight: float = 1.0e-2

    # Hand contact to object
    contact_hands_weight: float = 3.
    contact_hands_force: float = 1.0e-5
    contact_feet_weight: float = 0.1
    contact_obj_weight: float = 2.
    collision_obj_thigh: float = 1.

    u_weight_default: float = 1e-4


class G1_ObjPickupFloor(NLP_MuJoCo):

    def __init__(
        self,
        cfg: TaskG1ObjPickupFloor,
        ):
        super().__init__(cfg)
        self.cfg_scene = SceneG1ObjPickupFloor()
        self.edit.add_keyframes_from_file(self.cfg_scene.keyframes_file)
        self.edit.add_cnt_pairs_from_file(self.cfg_scene.contact_pair_file)
        self.edit.add_sensors_from_file(self.cfg_scene.sensor_file)
        self.add_scene_body(self.cfg_scene)

        # --- Initial state setup ---
        self.set_initial_state_from_keyframe(self.cfg_scene.keyframe, actuators_only=True)

        self.q_min = np.array(G1._25DoF_ObjFloor.RESTRICTED_JOINT_RANGE)[:, 0]
        self.q_max = np.array(G1._25DoF_ObjFloor.RESTRICTED_JOINT_RANGE)[:, 1]
        self.q_nom = self.x_0[self.act_qposadr]
        self.set_scaling(cfg)

        # Squat pose reference
        stand_pose = self.mj_model.keyframe(self.cfg_scene.keyframe).qpos
        squat_pose = self.mj_model.keyframe("knees_bent_pickup").qpos
        lift_pose = self.mj_model.keyframe("home").qpos

        stand_joints = stand_pose[G1._25DoF_Obj.IDX_JOINT_POS]
        pickup_joints = squat_pose[G1._25DoF_Obj.IDX_JOINT_POS]
        lift_joints = lift_pose[G1._25DoF_Obj.IDX_JOINT_POS]

        # --- Time parameters ---
        squat_node = int(cfg.squat_time / self.dt)
        pickup_node = int((cfg.squat_time + cfg.pickup_time) / self.dt)
        standup_node = int((cfg.squat_time + cfg.pickup_time + cfg.standup_time) / self.dt)

        # Object goal
        obj_position_goal = np.asarray(cfg.obj_init_pos) + np.asarray(cfg.obj_delta_position)
        obj_position_ref = np.tile(cfg.obj_init_pos, (self.T-1, 1))
        t_ = np.linspace(0, 1, self.T - standup_node - 1)
        obj_position_ref[standup_node:, :] += obj_position_goal[None, ] * t_[:, None]

        # --- Interpolate joints ---
        joint_ref_traj = np.tile(stand_joints, (self.T-1, 1))
        joint_pos_weight = np.full((self.T - 1, len(stand_joints)), cfg.joint_pos_weight)
        # Squat down
        alphas = np.linspace(0.2, 1.0, squat_node)
        joint_ref_traj[:squat_node] = (1 - alphas)[:, None] * stand_joints[None, :] + alphas[:, None] * pickup_joints[None, :]
        joint_pos_weight[squat_node-1] = cfg.joint_pos_weight_terminal
        # Pickup in squat
        joint_ref_traj[squat_node:standup_node] = pickup_joints
        joint_pos_weight[squat_node:standup_node] = cfg.joint_pos_weight / 5.
        # Standup
        alphas = np.linspace(0., 1.0, self.T-standup_node-1)
        joint_ref_traj[standup_node:] = (1 - alphas)[:, None] * pickup_joints[None, :] + alphas[:, None] * lift_joints[None, :]
        joint_pos_weight[standup_node:] = cfg.joint_pos_weight / 10.
        # --- Joint position cost ---
        self.add_state_cost(
            "joint_pos",
            quadratic_cost_nb,
            G1._25DoF_Obj.IDX_JOINT_POS,
            weights=joint_pos_weight,
            ref_values=joint_ref_traj,
            # weights_terminal=cfg.joint_pos_weight_terminal,
        )

        # --- Joint velocity damping ---
        self.add_state_cost(
            "joint_vel",
            quadratic_cost_nb,
            G1._25DoF_Obj.IDX_JOINT_VEL,
            weights=cfg.joint_vel_weight,
            weights_terminal=cfg.joint_vel_weight_terminal,
        )

        # --- Base height (smooth drop) ---
        base_height_ref = np.tile(self.x_0[2], (self.T-1, ))
        # Squat
        base_height_ref[:squat_node] = np.linspace(self.x_0[2], squat_pose[2], squat_node)
        alpha = 1.
        base_height_ref[squat_node:standup_node] = squat_pose[2] / 2.
        # Stand up
        base_height_ref[standup_node:] = np.linspace(squat_pose[2], self.x_0[2], self.T-1-standup_node)
        base_height_weight = np.full((self.T - 1, ), cfg.base_height_weight)
        base_height_weight[squat_node:standup_node] = cfg.base_height_weight_terminal
        base_height_weight[standup_node] = cfg.base_height_weight_terminal
        self.add_state_cost(
            "base_height",
            quadratic_cost_nb,
            2,
            weights=base_height_weight,
            ref_values=base_height_ref,
            weights_terminal=cfg.base_height_weight_terminal,
        )

        # --- Torso position ---
        self.add_state_cost(
            G1.Sensors.TORSO_POS,
            quadratic_cost_nb,
            [0, 1, 2],
            weights=cfg.torso_pos_weight,
            weights_terminal=cfg.torso_pos_weight_terminal,
            use_intial_as_ref=True,
        )

        # --- Base orientation ---
        self.add_state_cost(
            "base_quat",
            quaternion_dist_nb,
            [3, 4, 5, 6],
            weights=cfg.base_quat_weight,
            weights_terminal=cfg.base_quat_weight_terminal,
            use_intial_as_ref=True
        )

        # --- Contact plan hands ---
        contact_plan_hands = np.zeros((self.T-1, G1.N_HANDS), dtype=np.uint8)
        contact_plan_hands[pickup_node:] = 1
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.cnt_status_hand_id,
            ref_values=contact_plan_hands,
            weights=cfg.contact_hands_weight,
            weights_terminal=cfg.contact_hands_weight*50.,
        )
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            quadratic_cost_nb,
            sub_idx_sensor=G1.Sensors.cnt_force_hand_id,
            weights=cfg.contact_hands_force,
        )

        # --- Contact plan obj/floor ---
        contact_plan_obj = np.full((self.T-1, 1), 1, dtype=np.uint8) # feet always in contact
        contact_plan_obj[standup_node:, :] = 0

        self.add_sensor_cost(
            G1.Sensors.OBJ_FLOOR_CONTACT,
            hamming_dist_nb,
            sub_idx_sensor=[0],
            ref_values=contact_plan_obj,
            weights=cfg.contact_obj_weight,
        )

        # --- Contact plan feet ---
        contact_plan_feet = np.full((self.T-1, G1.N_FEET * G1.cnt_sensor_per_foot), 1, dtype=np.uint8) # feet always in contact
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.cnt_status_feet_id,
            ref_values=contact_plan_feet,
            weights=cfg.contact_feet_weight,
        )

        # --- Collision obj - thigh ---
        no_contact_plan_feet = np.zeros((self.T-1, len(G1.Sensors.OBJ_THIGH_COLLISION)), dtype=np.uint8) # feet always in contact
        self.add_sensor_cost(
            G1.Sensors.OBJ_THIGH_COLLISION,
            hamming_dist_nb,
            ref_values=no_contact_plan_feet,
            weights=cfg.collision_obj_thigh,
        )

        # Setup contact_plan for plots
        cnt_sensors = G1.Sensors.HAND_CONTACTS + G1.Sensors.OBJ_FLOOR_CONTACT
        sub_id_cnt_status = G1.Sensors.cnt_status_hand_id + [G1.Sensors.cnt_status_hand_id[-1] + 3 + 1]
        self.set_contact_sensor_id(cnt_sensors, sub_id_cnt_status) # For plotting
        self.contact_plan = np.concatenate(
            (contact_plan_hands, contact_plan_obj),
            axis=-1
        )

        # --- Obj cost ---
        self.add_state_cost(
            "obj_position",
            quadratic_cost_nb,
            G1._25DoF_Obj.IDX_BOX_POS,
            weights=cfg.obj_pos_weight,
            weights_terminal=cfg.obj_pos_weight_terminal,
            ref_values=obj_position_ref,
            ref_values_terminal=obj_position_goal,
        )
        self.add_state_cost(
            "obj_quat",
            quadratic_cost_nb,
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
            weights_terminal=cfg.obj_linvel_weight*10,
        )
        self.add_state_cost(
            "obj_angvel",
            quadratic_cost_nb,
            G1._25DoF_Obj.IDX_BOX_ANGVEL,
            weights=cfg.obj_angvel_weight,
            weights_terminal=cfg.obj_angvel_weight*10,
        )

        # --- Control regularization ---
        w_u_traj = np.full(self.Nu, cfg.u_weight_default)
        self.add_control_cost(
            "u_traj",
            quadratic_cost_nb,
            idx=list(range(self.Nu)),
            weights=w_u_traj,
        )

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
        scale_q = np.full((self.Nq,), self.cfg_scene.scale_q)
        scale_v = np.full((self.Nv,), self.cfg_scene.scale_v)

        scale_q[:7] /= 10.
        scale_v[:6] /= 10.
        scale_q[-7:] = 0.
        scale_v[-6:] = 0.

        scale_q[G1._25DoF_ObjFloor.IDX_WAIST+7:] *= self.upper_body_scale
        obj_qpos_id = G1._25DoF_ObjFloor.IDX_BOX_POS + G1._25DoF_ObjFloor.IDX_BOX_QUAT
        scale_q[obj_qpos_id] = 0.
        scale_v[-6:] = 0.

        return super().set_random_initial_state(
            self.cfg_scene.keyframe,
            scale_q,
            scale_v,
            is_floating_base=True,
            obj_qpos_id=obj_qpos_id,
            N_rollout_steps=150,
            obj_x_range=self.cfg_scene.obj_x_range,
            obj_y_range=self.cfg_scene.obj_y_range,
            obj_w_range=self.cfg_scene.obj_w_range,
            )
    