import numpy as np
from dataclasses import dataclass

import sbto.tasks.g1.constants as G1
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_mj import  TaskMj
from sbto.tasks.cost import quadratic_cost_nb, quaternion_dist_nb, hamming_dist_nb


@dataclass
class ConfigG1PickupFloor():
    # --- Timing ---
    squat_time: float = 1.     # seconds to reach squat pose
    pickup_time: float = 0.2   # seconds to reach the box after squatting
    standup_time: float = 0.1  # seconds to reach the box after pickup

    # -- Obj Goal ---
    obj_init_pos: tuple = (0.35, 0., 0.115)
    obj_delta_position: tuple = (0., 0., 0.7)

    obj_pos_weight: float = 0.
    obj_pos_weight_terminal: tuple = (10.0, 10.0, 10.0)
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

    _target_:str = "sbto.tasks.g1.g1_pickup_floor.G1PickupFloor"

class G1PickupFloor(TaskMj):

    def __init__(
        self,
        sim: SimMjRollout,
        cfg: ConfigG1PickupFloor
        ):
        super().__init__(sim)
        RESTRICTED_JOINT_RANGE = (
            # Left leg.
            (-2.57, 0.1),
            (-0.5, 0.5),
            (-0.5, 0.5),
            (0, 2.57),
            (-0.5, 0.7),
            (-0.2, 0.2),
            # Right leg.
            (-2.57, 0.1),
            (-0.5, 0.5),
            (-0.5, 0.5),
            (0, 2.57),
            (-0.5, 0.7),
            (-0.2, 0.2),
            # Waist.
            (-0.5, 0.5),
            # Left shoulder.
            (-1.57, 1.57),
            (-0.2, 1.57),
            (-1, 1),
            (-1., 1.57),
            (-1., 1.),
            (-1.57, -1.57), # 0 range for the yaw wrists
            # Right shoulder.
            (-1.57, 1.57),
            (-1.57, 0.2),
            (-1, 1),
            (-1., 1.57),
            (-1., 1.),
            (1.57, 1.57), # 0 range for the yaw wrists
        )

        sim.set_act_limits(
            np.array(RESTRICTED_JOINT_RANGE)[:, 0],
            np.array(RESTRICTED_JOINT_RANGE)[:, 1],
        )

        # Squat pose reference
        stand_pose = sim.mj_scene.mj_model.keyframe(sim.cfg.keyframe_x0).qpos
        squat_pose = sim.mj_scene.mj_model.keyframe("knees_bent_pickup").qpos
        lift_pose = sim.mj_scene.mj_model.keyframe("home").qpos

        stand_joints = stand_pose[sim.mj_scene.act_pos_adr]
        pickup_joints = squat_pose[sim.mj_scene.act_pos_adr]
        lift_joints = lift_pose[sim.mj_scene.act_pos_adr]

        # --- Time parameters ---
        dt = sim.mj_scene.dt
        squat_node = int(cfg.squat_time / dt)
        pickup_node = int((cfg.squat_time + cfg.pickup_time) / dt)
        standup_node = int((cfg.squat_time + cfg.pickup_time + cfg.standup_time) / dt)

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
            sim.mj_scene.act_pos_adr,
            weights=joint_pos_weight,
            ref_values=joint_ref_traj,
            # weights_terminal=cfg.joint_pos_weight_terminal,
        )

        # --- Joint velocity damping ---
        self.add_state_cost(
            "joint_vel",
            quadratic_cost_nb,
            sim.mj_scene.act_vel_adr,
            weights=cfg.joint_vel_weight,
            weights_terminal=cfg.joint_vel_weight_terminal,
        )

        # --- Base height (smooth drop) ---
        base_height_ref = np.tile(sim.x_0[2], (sim.T-1, ))
        # Squat
        base_height_ref[:squat_node] = np.linspace(sim.x_0[2], squat_pose[2], squat_node)
        base_height_ref[squat_node:standup_node] = squat_pose[2] / 2.
        # Stand up
        base_height_ref[standup_node:] = np.linspace(squat_pose[2], sim.x_0[2], sim.T-1-standup_node)
        base_height_weight = np.full((sim.T - 1, ), cfg.base_height_weight)
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
            sub_idx_sensor=G1.Sensors.id_cnt_status_hands,
            ref_values=contact_plan_hands,
            weights=cfg.contact_hands_weight,
            weights_terminal=cfg.contact_hands_weight*50.,
        )
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            quadratic_cost_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_force_hands,
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
        contact_plan_feet = np.full((self.T-1, G1.N_FEET * G1._cnt_sens_per_foot), 1, dtype=np.uint8) # feet always in contact
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_status_feet,
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
        sub_id_cnt_status = G1.Sensors.id_cnt_status_hands + [G1.Sensors.id_cnt_status_hands[-1] + 3 + 1]
        self.set_contact_sensor_id(cnt_sensors, sub_id_cnt_status) # For plotting
        self.contact_plan = np.concatenate(
            (contact_plan_hands, contact_plan_obj),
            axis=-1
        )

        # --- Obj cost ---
        self.add_state_cost(
            "obj_position",
            quadratic_cost_nb,
            sim.mj_scene.obj_pos_adr,
            weights=cfg.obj_pos_weight,
            weights_terminal=cfg.obj_pos_weight_terminal,
            ref_values=obj_position_ref,
            ref_values_terminal=obj_position_goal,
        )
        self.add_state_cost(
            "obj_quat",
            quadratic_cost_nb,
            sim.mj_scene.obj_quat_adr,
            weights=cfg.obj_quat_weight,
            weights_terminal=cfg.obj_quat_weight_terminal,
            use_intial_as_ref=True
        )
        self.add_state_cost(
            "obj_linvel",
            quadratic_cost_nb,
            sim.mj_scene.obj_v_adr,
            weights=cfg.obj_linvel_weight,
            weights_terminal=cfg.obj_linvel_weight*10,
        )
        self.add_state_cost(
            "obj_angvel",
            quadratic_cost_nb,
            sim.mj_scene.obj_w_adr,
            weights=cfg.obj_angvel_weight,
            weights_terminal=cfg.obj_angvel_weight*10,
        )

        # --- Control regularization ---
        w_u_traj = np.full(sim.mj_scene.Nu, cfg.u_weight_default)
        self.add_control_cost(
            "u_traj",
            quadratic_cost_nb,
            idx=list(range(sim.mj_scene.Nu)),
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