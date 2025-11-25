import numpy as np
from dataclasses import dataclass

import sbto.tasks.g1.constants as G1
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_mj import  TaskMj
from sbto.tasks.cost import quadratic_cost_nb, quaternion_dist_nb, hamming_dist_nb


@dataclass
class ConfigG1PickupTable():
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

class G1PickupTable(TaskMj):

    def __init__(
        self,
        sim: SimMjRollout,
        cfg: ConfigG1PickupTable
        ):
        super().__init__(sim)
        Nu = sim.mj_scene.Nu
        dt = sim.mj_scene.dt
        T = sim.T

        pd_range = np.array(G1._25DoF_Obj.RESTRICTED_JOINT_RANGE)
        pd_range[G1._25DoF.IDX_WAIST_YAW, 0] = -0.25
        pd_range[G1._25DoF.IDX_WAIST_YAW, 1] = 0.25

        sim.set_act_limits(
            pd_range[:, 0],
            pd_range[:, 1],
        )

        obj_position_0 = np.array(cfg.obj_init_pos)
        obj_position_goal = obj_position_0 + cfg.obj_delta_position
        # self.x_0[sim.mj_scene.obj_pos_adr] = self.obj_position_0
        # self.set_initial_state(self.x_0)
        node_impact = int(cfg.reaching_cnt_time // dt)
        obj_position_ref = np.zeros((T, 3))
        obj_position_ref += obj_position_0
        t_ = np.arange(T - node_impact) * dt
        dir = obj_position_goal - obj_position_0
        obj_position_ref[node_impact:, :] += dir[None, ] * t_[:, None]

        # --- G1 costs ---
        self.add_state_cost(
            "joint_pos",
            quadratic_cost_nb,
            sim.mj_scene.act_pos_adr,
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
            sim.mj_scene.act_vel_adr,
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
            sim.mj_scene.obj_pos_adr,
            weights=cfg.obj_pos_weight,
            weights_terminal=cfg.obj_pos_weight_terminal,
            ref_values_terminal=obj_position_goal,
            use_intial_as_ref=True
        )
        self.add_state_cost(
            "obj_quat",
            quaternion_dist_nb,
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
            weights_terminal=cfg.obj_linvel_weight_term,
        )
        self.add_state_cost(
            "obj_angvel",
            quadratic_cost_nb,
            sim.mj_scene.obj_w_adr,
            weights=cfg.obj_angvel_weight,
            weights_terminal=cfg.obj_angvel_weight_term,
        )

        # --- Contact plan hands ---
        self.set_contact_sensor_id(G1.Sensors.HAND_CONTACTS, G1.Sensors.id_cnt_status_hands) # For plotting
        self.contact_plan = np.zeros((T, G1.N_HANDS), dtype=np.uint8)
        self.contact_plan[node_impact:] = 1.
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_status_hands,
            ref_values=self.contact_plan[:-1],
            ref_values_terminal=self.contact_plan[-1:],
            weights=cfg.contact_hands_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            quadratic_cost_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_force_hands,
            weights=cfg.contact_force_obj_weight,
        )

        # --- Contact plan feet ---
        self.contact_plan_feet = np.full((T, G1.N_FEET * G1._cnt_sens_per_foot), 1, dtype=np.uint8) # feet always in contact
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_status_feet,
            ref_values=self.contact_plan_feet[:-1],
            ref_values_terminal=self.contact_plan_feet[-1:],
            weights=cfg.contact_feet_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            quadratic_cost_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_force_feet,
            weights=cfg.contact_force_feet_weight,
        )


        # --- Contact obj table ---
        self.contact_plan_obj = np.full((T, 1), 1, dtype=np.uint8) # feet always in contact
        node_lift_obj = node_impact + int(cfg.delay_lift // dt)
        self.contact_plan_obj[node_lift_obj:, :] = 0

        self.add_sensor_cost(
            G1.Sensors.OBJ_STATIC_CONTACT,
            hamming_dist_nb,
            sub_idx_sensor=[0],
            ref_values=self.contact_plan_obj[:-1],
            weights=cfg.contact_obj_weight,
        )

        # --- Control cost ---
        w_u_traj = np.full(Nu, cfg.u_weight_default)
        w_u_traj[G1._25DoF_Obj.IDX_HIP_KNEE] *= cfg.u_weight_hip_knee_scale
        w_u_traj[G1._25DoF_Obj.IDX_WAIST_YAW+1:] *= cfg.u_weight_upperbody_scale
        self.add_control_cost(
            "u_traj",
            quadratic_cost_nb,
            idx=list(range(Nu)),
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