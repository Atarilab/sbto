import numpy as np
from dataclasses import dataclass

import sbto.tasks.g1.constants as G1
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_mj_ref import TaskMjRef
from sbto.tasks.cost import quadratic_cost_nb, quaternion_dist_nb, hamming_dist_nb

@dataclass
class ConfigG1PickPlaceTableRef():
    # --- Reference motion ---
    ref_motion_path: str = "./sbto/tasks/g1/motion/pick_and_place_table.pkl"
    t0: float = 0.5
    speedup: float = 2. 
    z_offset: float = 0.

    # --- Obj state goal ---
    time_lift: float = 0.2
    time_after_place: float = 0.2

    # --- State costs ---
    joint_pos_weight: float = 0.05
    joint_vel_weight: float = 0.05
    torso_pos_weight: float = 2.
    base_pos_weight: float = 2.
    torso_quat_weight: float = 1.
    torso_quat_weight_terminal: float = 10.

    # --- Torso pose cost ---
    obj_pos_weight: float = 30.
    obj_quat_weight: float = 10.

    # --- Hand pose cost ---
    hand_position: float = 5.
    hand_orientation: float = 0.1

    # --- Contact cost ---
    contact_hands_weight: float = 1.
    contact_obj_weight: float = 1.
    contact_obj_weight_terminal: float = 10.
    contact_force_obj_weight: float = 1.0e-3

class G1PickPlaceTableRef(TaskMjRef):

    def __init__(
        self,
        sim: SimMjRollout,
        cfg: ConfigG1PickPlaceTableRef
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
            *G1.Sensors.HAND_CONTACTS,
            *G1.Sensors.OBJ_TABLE_CONTACT,
            *G1.Sensors.HAND_POS,
            *G1.Sensors.HAND_QUAT,
        ]
        self.ref.add_sensor_data(sim.mj_scene.mj_model, sensor_names)
        sim.set_initial_state(self.ref.x0)
        q_min = sim.mj_scene.q_min
        q_max = sim.mj_scene.q_max
        sim.set_act_limits(q_min, q_max)

        # --- G1 costs ---
        self.add_state_cost_from_ref(
            "joint_ref",
            quadratic_cost_nb,
            sim.mj_scene.act_qposadr,
            weights=cfg.joint_pos_weight,
            weights_terminal=cfg.joint_pos_weight,
        )
        self.add_state_cost_from_ref(
            "base_position",
            quadratic_cost_nb,
            [0, 1, 2],
            weights=cfg.base_pos_weight,
            weights_terminal=cfg.base_pos_weight,
        )
        self.add_state_cost_from_ref(
            "joint_vel",
            quadratic_cost_nb,
            sim.mj_scene.act_vel_adr,
            weights=cfg.joint_vel_weight,
            weights_terminal=cfg.joint_vel_weight,
        )
        self.add_sensor_cost_from_ref(
            G1.Sensors.TORSO_POS,
            quadratic_cost_nb,
            weights=cfg.torso_pos_weight,
            weights_terminal=cfg.torso_pos_weight,
        )
        self.add_sensor_cost_from_ref(
            G1.Sensors.TORSO_QUAT,
            quaternion_dist_nb,
            weights=cfg.torso_quat_weight,
            weights_terminal=cfg.torso_quat_weight_terminal,
        )
        # --- Obj cost ---
        self.add_state_cost_from_ref(
            "obj_position",
            quadratic_cost_nb,
            sim.mj_scene.obj_pos_adr,
            weights=cfg.obj_pos_weight,
            weights_terminal=cfg.obj_pos_weight,
        )
        self.add_state_cost_from_ref(
            "obj_quat",
            quaternion_dist_nb,
            sim.mj_scene.obj_quat_adr,
            weights=cfg.obj_quat_weight,
            weights_terminal=cfg.obj_quat_weight,
        )
        # Hand position
        self.add_sensor_cost_from_ref(
            G1.Sensors.HAND_POS,
            quadratic_cost_nb,
            weights=cfg.hand_position,
        )
        # Hand orientation
        self.add_sensor_cost_from_ref(
            G1.Sensors.HAND_QUAT,
            quaternion_dist_nb,
            weights=cfg.hand_orientation,
        )

        # # --- Contact plan hands ---
        self.set_contact_sensor_id(G1.Sensors.HAND_CONTACTS + G1.Sensors.OBJ_TABLE_CONTACT, [0, 4, 8]) # For plotting

        self.contact_plan = np.zeros((self.T, 3))
        # Hands always in contact
        self.contact_plan[:, 0] = 1
        self.contact_plan[:, 1] = 1
        # For the object, lift and place time
        node_lift = int(cfg.time_lift / dt)
        node_place = int(duration - cfg.time_after_place / dt)
        self.contact_plan[:, 2] = 1
        self.contact_plan[node_lift:node_place, 2] = 0

        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_status_hands,
            ref_values=self.contact_plan[:-1, :2],
            weights=cfg.contact_hands_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.OBJ_TABLE_CONTACT,
            hamming_dist_nb,
            sub_idx_sensor=[0],
            ref_values=self.contact_plan[:-1, -1],
            weights=cfg.contact_obj_weight,
            weights_terminal=cfg.contact_obj_weight_terminal,
        )
        self.add_sensor_cost(
            G1.Sensors.HAND_CONTACTS,
            quadratic_cost_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_force_hands,
            weights=cfg.contact_force_obj_weight,
        )