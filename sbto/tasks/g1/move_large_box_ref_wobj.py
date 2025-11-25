import numpy as np
from dataclasses import dataclass

import sbto.tasks.g1.constants as G1
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_mj_ref import TaskMjRef
from sbto.tasks.cost import quadratic_cost_nb_mod, quaternion_dist_nb_mod, hamming_dist_nb_mod

@dataclass
class ConfigG1MoveLargeBoxObjRef():
    # --- Reference motion ---
    ref_motion_path: str = "./sbto/tasks/g1/motion/sub3_largebox_005_original.pkl"
    t0: float = 0.
    speedup: float = 1.4
    z_offset: float = 0.01

    t_hand_cnt_before_lift: float = 0.1
    t_hand_cnt_after_place: float = 0.25

    # --- State costs ---
    joint_pos_weight: float = 0.1
    joint_vel_weight: float = 0.05
    base_pos_weight: float = 5.
    base_quat_weight: float = 1.

    # --- Torso pose/vel ---
    torso_pos_weight: float = 30.
    torso_quat_weight: float = 1.
    torso_quat_weight_terminal: float = 10.
    torso_linvel_weight: float = 1.
    torso_linvel_weight_terminal: float = 10.
    torso_angvel_weight: float = 1.
    torso_angvel_weight_terminal: float = 10.


    # --- Hand pose cost ---
    hand_position: float = 5.
    hand_orientation: float = 0.1

    # --- Foot pose ---
    foot_position: float = 10.
    foot_orientation: float = 0.1

    # --- Feet Contact cost ---
    contact_feet_weight: float = 0.25
    contact_force_feet_weight: float = 1.0e-6

    # --- Obj pose cost ---
    obj_pos_weight: float = 20.
    obj_quat_weight: float = 5.
    obj_v_weight: float = 1.
    obj_w_weight: float = 1.
    # --- Obj/hands Contact cost ---
    contact_obj_weight: float = 0.5
    contact_hands_weight: float = 0.5

    collision_obj_thigh: float = 0.25

class G1MoveLargeBoxObjRef(TaskMjRef):

    def __init__(
        self,
        sim: SimMjRollout,
        cfg: ConfigG1MoveLargeBoxObjRef
        ):
        super().__init__(sim)
        Nu = sim.mj_scene.Nu
        dt = sim.mj_scene.dt
        T = sim.T
        duration = dt * T
        self.init_reference(
            cfg.ref_motion_path,
            cfg.t0,
            cfg.speedup,
            cfg.z_offset,
        )

        sensor_names = [
            G1.Sensors.TORSO_POS,
            G1.Sensors.TORSO_QUAT,
            *G1.Sensors.FEET_CONTACTS,
            *G1.Sensors.HAND_CONTACTS,
            *G1.Sensors.OBJ_FLOOR_CONTACT,
            *G1.Sensors.FEET_POS,
            *G1.Sensors.FEET_QUAT,
            *G1.Sensors.HAND_POS,
            *G1.Sensors.HAND_QUAT,
            G1.Sensors.TORSO_LINVEL,
            G1.Sensors.TORSO_LINVEL,
        ]
        self.ref.add_sensor_data(sim.mj_scene.mj_model, sensor_names)
        sim.set_initial_state(self.ref.x0)
        margin = 0.2
        q_min = np.min(self.ref.act_qpos, axis=0) - margin
        q_max = np.max(self.ref.act_qpos, axis=0) + margin
        sim.set_act_limits(q_min, q_max)

        # --- G1 costs ---
        self.add_state_cost_from_ref(
            "joint_ref",
            quadratic_cost_nb_mod,
            sim.mj_scene.act_qposadr,
            weights=cfg.joint_pos_weight,
            weights_terminal=cfg.joint_pos_weight,
        )
        self.add_state_cost_from_ref(
            "base_position",
            quadratic_cost_nb_mod,
            [0, 1, 2],
            weights=cfg.base_pos_weight,
            weights_terminal=cfg.base_pos_weight,
        )
        self.add_state_cost_from_ref(
            "base_quat",
            quadratic_cost_nb_mod,
            [3, 4, 5, 6],
            weights=cfg.base_quat_weight,
            weights_terminal=cfg.base_quat_weight,
        )
        self.add_state_cost_from_ref(
            "joint_vel",
            quadratic_cost_nb_mod,
            sim.mj_scene.act_vel_adr,
            weights=cfg.joint_vel_weight,
            weights_terminal=cfg.joint_vel_weight,
        )
        self.add_sensor_cost_from_ref(
            G1.Sensors.TORSO_POS,
            quadratic_cost_nb_mod,
            weights=cfg.torso_pos_weight,
            weights_terminal=cfg.torso_pos_weight,
        )
        self.add_sensor_cost_from_ref(
            G1.Sensors.TORSO_QUAT,
            quaternion_dist_nb_mod,
            weights=cfg.torso_quat_weight,
            weights_terminal=cfg.torso_quat_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_LINVEL,
            quadratic_cost_nb_mod,
            weights=cfg.torso_linvel_weight,
            weights_terminal=cfg.torso_linvel_weight_terminal,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_ANGVEL,
            quadratic_cost_nb_mod,
            weights=cfg.torso_angvel_weight,
            weights_terminal=cfg.torso_angvel_weight_terminal,
        )
        # --- Obj cost ---
        self.add_state_cost_from_ref(
            "obj_position",
            quadratic_cost_nb_mod,
            sim.mj_scene.obj_pos_adr,
            weights=cfg.obj_pos_weight,
            weights_terminal=cfg.obj_pos_weight,
        )
        self.add_state_cost_from_ref(
            "obj_quat",
            quaternion_dist_nb_mod,
            sim.mj_scene.obj_quat_adr,
            weights=cfg.obj_quat_weight,
            weights_terminal=cfg.obj_quat_weight,
        )
        self.add_state_cost_from_ref(
            "obj_vel",
            quaternion_dist_nb_mod,
            sim.mj_scene.obj_v_adr,
            weights=cfg.obj_v_weight,
            weights_terminal=cfg.obj_v_weight,
        )
        self.add_state_cost_from_ref(
            "obj_w",
            quaternion_dist_nb_mod,
            sim.mj_scene.obj_w_adr,
            weights=cfg.obj_w_weight,
            weights_terminal=cfg.obj_w_weight,
        )
        # Hand position
        w = np.full(6, cfg.hand_position)
        w[2] *= 2.
        w[-1] *= 2.
        self.add_sensor_cost_from_ref(
            G1.Sensors.HAND_POS,
            quadratic_cost_nb_mod,
            weights=w,
        )
        # Hand orientation
        self.add_sensor_cost_from_ref(
            G1.Sensors.HAND_QUAT,
            quaternion_dist_nb_mod,
            weights=cfg.hand_orientation,
        )
        # Foot position
        self.add_sensor_cost_from_ref(
            G1.Sensors.FEET_POS,
            quadratic_cost_nb_mod,
            weights=cfg.foot_position,
        )
        # Foot orientation
        self.add_sensor_cost_from_ref(
            G1.Sensors.FEET_QUAT,
            quaternion_dist_nb_mod,
            weights=cfg.torso_quat_weight,
        )

        # --- Contact plan feet ---
        N_feet_cnt = len(G1.Sensors.FEET_CONTACTS)
        N_hands_cnt = len(G1.Sensors.HAND_CONTACTS)
        N_obj_cnt = len(G1.Sensors.OBJ_FLOOR_CONTACT)
        N_cnt = N_feet_cnt + N_hands_cnt + N_obj_cnt

        cnt_sns =  G1.Sensors.FEET_CONTACTS + G1.Sensors.OBJ_FLOOR_CONTACT + G1.Sensors.HAND_CONTACTS
        dim_sns = 4
        cnt_sns_sub_id = list(range(0, len(cnt_sns) * dim_sns, dim_sns))
        self.set_contact_sensor_id(cnt_sns, cnt_sns_sub_id)
        self.contact_plan = np.zeros((self.T, N_cnt), dtype=np.int32)

        for i, foot_cnt in enumerate(G1.Sensors.FEET_CONTACTS):
            self.contact_plan[:, i] = self.ref.data[foot_cnt][:T, 0]
        self.contact_plan[self.contact_plan > 1] = 1

        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            hamming_dist_nb_mod,
            sub_idx_sensor=G1.Sensors.id_cnt_status_feet,
            ref_values=self.contact_plan[:, :N_feet_cnt],
            weights=cfg.contact_feet_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            quadratic_cost_nb_mod,
            sub_idx_sensor=G1.Sensors.id_cnt_force_feet,
            weights=cfg.contact_force_feet_weight,
        )

        # --- Contact plan hands/obj ---
        # Contact plan of the obj from the ref
        self.contact_plan[:, N_feet_cnt] = self.ref.data[G1.Sensors.OBJ_FLOOR_CONTACT[0]][:T, 0]
        # Contact plan of the hands slightly offset
        nodes_lifted = np.where(self.contact_plan[:, N_feet_cnt:N_feet_cnt+N_obj_cnt] == 0)[0]
        node_grasp_hands = nodes_lifted[0] - int(cfg.t_hand_cnt_before_lift / dt)
        node_release_hands = nodes_lifted[-1] + int(cfg.t_hand_cnt_after_place / dt)
        self.contact_plan[node_grasp_hands:node_release_hands, N_feet_cnt+N_obj_cnt:] = 1
        self.contact_plan[self.contact_plan > 1] = 1


        self.add_sensor_cost(
            G1.Sensors.OBJ_STATIC_CONTACT,
            hamming_dist_nb_mod,
            sub_idx_sensor=[0],
            ref_values=self.contact_plan[:, N_feet_cnt],
            weights=cfg.contact_obj_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            hamming_dist_nb_mod,
            sub_idx_sensor=G1.Sensors.id_cnt_status_hands,
            ref_values=self.contact_plan[:, N_feet_cnt+N_obj_cnt:],
            weights=cfg.contact_hands_weight,
        )

        # --- Collision obj - thigh ---
        no_contact_plan_feet = np.zeros((self.T-1, len(G1.Sensors.OBJ_THIGH_COLLISION)), dtype=np.uint8) # feet always in contact
        self.add_sensor_cost(
            G1.Sensors.OBJ_THIGH_COLLISION,
            hamming_dist_nb_mod,
            ref_values=no_contact_plan_feet,
            weights=cfg.collision_obj_thigh,
        )