import numpy as np
from dataclasses import dataclass

import sbto.tasks.g1.constants as G1
from sbto.sim.sim_mj_rollout import SimMjRollout
from sbto.tasks.task_mj_ref import TaskMjRef
from sbto.tasks.cost import quadratic_cost_nb_mod, quaternion_dist_nb_mod, hamming_dist_nb_mod

@dataclass
class ConfigG1MoveLargeBoxRef():
    # --- Reference motion ---
    ref_motion_path: str = "./sbto/tasks/g1/motion/move_large_box_no_obk.pkl"
    t0: float = 0.
    speedup: float = 1.1
    z_offset: float = 0.024

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

    # --- Obj pose cost ---
    obj_pos_weight: float = 30.
    obj_quat_weight: float = 10.
    
    # --- Hand pose cost ---
    hand_position: float = 3.
    hand_orientation: float = 0.1

    # --- Foot pose ---
    foot_position: float = 10.
    foot_orientation: float = 0.1

    # --- Feet Contact cost ---
    contact_feet_weight: float = 1.
    contact_force_feet_weight: float = 1.0e-6

class G1MoveLargeBoxRef(TaskMjRef):

    def __init__(
        self,
        sim: SimMjRollout,
        cfg: ConfigG1MoveLargeBoxRef
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
            *G1.Sensors.FEET_POS,
            *G1.Sensors.FEET_QUAT,
            *G1.Sensors.HAND_POS,
            *G1.Sensors.HAND_QUAT,
            G1.Sensors.TORSO_LINVEL,
            G1.Sensors.TORSO_LINVEL,
        ]
        self.ref.add_sensor_data(sim.mj_scene.mj_model, sensor_names)
        sim.set_initial_state(self.ref.x0)
        q_min = sim.mj_scene.q_min
        q_max = sim.mj_scene.q_max
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
        # self.add_state_cost_from_ref(
        #     "obj_position",
        #     quadratic_cost_nb_mod,
        #     sim.mj_scene.obj_pos_adr,
        #     weights=cfg.obj_pos_weight,
        #     weights_terminal=cfg.obj_pos_weight,
        # )
        # self.add_state_cost_from_ref(
        #     "obj_quat",
        #     quaternion_dist_nb_mod,
        #     sim.mj_scene.obj_quat_adr,
        #     weights=cfg.obj_quat_weight,
        #     weights_terminal=cfg.obj_quat_weight,
        # )

        # Hand position
        self.add_sensor_cost_from_ref(
            G1.Sensors.HAND_POS,
            quadratic_cost_nb_mod,
            weights=cfg.hand_position,
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
        self.set_contact_sensor_id(G1.Sensors.FEET_CONTACTS, G1.Sensors.id_cnt_status_feet) # For plotting

        self.contact_plan = np.zeros((self.T, len(G1.Sensors.FEET_CONTACTS)))
        for i, foot_cnt in enumerate(G1.Sensors.FEET_CONTACTS):
            self.contact_plan[:, i] = self.ref.data[foot_cnt][:T, 0]
        self.contact_plan[self.contact_plan > 1] = 1

        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            hamming_dist_nb_mod,
            sub_idx_sensor=G1.Sensors.id_cnt_status_feet,
            ref_values=self.contact_plan[:-1],
            ref_values_terminal=self.contact_plan[-1:],
            weights=cfg.contact_feet_weight,
        )
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            quadratic_cost_nb_mod,
            sub_idx_sensor=G1.Sensors.id_cnt_force_feet,
            weights=cfg.contact_force_feet_weight,
        )
