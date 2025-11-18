import numpy as np
from dataclasses import dataclass

from sbto.sim.sim_mj_rollout import SimMjRollout
import sbto.tasks.g1.constants as G1
from sbto.utils.gait import GaitConfig, generate_contact_plan
from sbto.tasks.task_mj import TaskMj
from sbto.tasks.cost import quadratic_cost_nb, quaternion_dist_nb, hamming_dist_nb

@dataclass
class ConfigG1Gait():
    # --- Desired motion parameters ---
    v_des: tuple = (0.5, 0.0, 0.0)  # Desired torso linear velocity [vx, vy, vz]

    # --- Desired gait parameters ---
    stance_ratio: tuple = (0.55, 0.55)
    phase_offset: tuple = (0.5, 0.0)
    nominal_period: float = 0.9

    # --- State costs ---
    joint_pos_weight: float = 0.
    joint_pos_weight_terminal: float = 10.

    joint_vel_weight: float = 0.01
    joint_vel_lower_mult: float = 0.1

    # --- Torso position cost ---
    torso_height_weight: float = 1.
    torso_height_weight_terminal: float = 2000.0

    # --- Torso XY tracking cost ---
    torso_xy_weight: float = 10.
    torso_xy_weight_terminal: float = 300.0

    # --- Torso linear velocity cost ---
    torso_linvel_weight: tuple = (2.0, 2.0, 1.0)
    torso_linvel_weight_terminal: tuple = (10.0, 10.0, 40.0)

    # --- Torso angular velocity cost ---
    torso_angvel_weight: float = 1.0
    torso_angvel_weight_terminal: float = 10.0

    # --- Torso orientation cost ---
    torso_quat_weight: float = 0.01
    torso_quat_weight_terminal: float = 50.0

    # --- Contact plan and cost ---
    contact_weight: float = 10.0
    contact_weight_term: float = 10.0
    contact_force_weight: float = 1.0e-5

    # --- Control cost ---
    u_weight_default: float = 1.
    u_weight_hip_knee_scale: float = 0.1
    u_weight_upperbody_scale: float = 3.0
    u_torques: float = 1.0e-5

    _target_:str = "sbto.tasks.g1.g1_gait.G1Gait"

class G1Gait(TaskMj):

    def __init__(
        self,
        sim: SimMjRollout,
        cfg: ConfigG1Gait
        ):
        super().__init__(sim)
        RESTRICTED_JOINT_RANGE = (
            # Left leg.
            (-1.57, 1.57),
            (-0.5, 0.5),
            (-0.5, 0.5),
            (0, 1.57),
            (-0.5, 0.7),
            (-0.2, 0.2),
            # Right leg.
            (-1.57, 1.57),
            (-0.5, 0.5),
            (-0.5, 0.5),
            (0, 1.57),
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

        # --- Add costs ---
        self.add_state_cost(
            "joint_pos",
            quadratic_cost_nb,
            sim.mj_scene.act_pos_adr,
            weights=cfg.joint_pos_weight,
            use_intial_as_ref=True,
            weights_terminal=cfg.joint_pos_weight_terminal,
        )
        self.add_state_cost(
            "joint_vel_upper",
            quadratic_cost_nb,
            sim.mj_scene.act_vel_adr[G1._25DoF.IDX_WAIST_YAW:],
            weights=cfg.joint_vel_weight,
        )
        self.add_state_cost(
            "joint_vel_lower",
            quadratic_cost_nb,
            sim.mj_scene.act_vel_adr[:G1._25DoF.IDX_WAIST_YAW],
            weights=cfg.joint_vel_weight * cfg.joint_vel_lower_mult,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_POS,
            quadratic_cost_nb,
            2,
            weights=cfg.torso_height_weight,
            weights_terminal=cfg.torso_height_weight_terminal,
            use_intial_as_ref=True,
        )
        v_des = np.array(cfg.v_des)
        self.add_sensor_cost(
            G1.Sensors.TORSO_POS,
            quadratic_cost_nb,
            [0, 1],
            ref_values=v_des[None, :2] * np.linspace(0., sim.duration, num=sim.T)[:self.T-1, None],
            weights=cfg.torso_xy_weight,
            ref_values_terminal=v_des[:2] * sim.duration,
            weights_terminal=cfg.torso_xy_weight_terminal,
        )
        self.add_sensor_cost(
            G1.Sensors.TORSO_LINVEL,
            quadratic_cost_nb,
            ref_values=v_des,
            weights=cfg.torso_linvel_weight,
            ref_values_terminal=0.,
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

        # --- Contact plan ---
        gait = GaitConfig(
            G1.N_FEET,
            cfg.stance_ratio,
            cfg.phase_offset,
            cfg.nominal_period
            )
        self.set_contact_sensor_id(G1.Sensors.FEET_CONTACTS, G1.Sensors.id_cnt_status_feet)
        self.contact_plan = generate_contact_plan(sim.T, sim.mj_scene.dt, gait)
        self.contact_plan = self.contact_plan.repeat(G1._cnt_sens_per_foot, axis=-1)
        
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            hamming_dist_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_status_feet,
            ref_values=self.contact_plan[:-1],
            ref_values_terminal=self.contact_plan[-1:],
            weights=cfg.contact_weight,
            weights_terminal=cfg.contact_weight_term,
        )
        self.add_sensor_cost(
            G1.Sensors.FEET_CONTACTS,
            quadratic_cost_nb,
            sub_idx_sensor=G1.Sensors.id_cnt_force_feet,
            weights=cfg.contact_force_weight,
        )

        # --- Control cost ---
        Nu = self.mj_scene.Nu
        w_u_traj = np.full(Nu, cfg.u_weight_default)
        w_u_traj[list(G1._25DoF.IDX_HIP_KNEE)] *= cfg.u_weight_hip_knee_scale
        w_u_traj[13:] *= cfg.u_weight_upperbody_scale
        self.add_control_cost(
            "u_traj",
            quadratic_cost_nb,
            idx=list(range(Nu)),
            weights=w_u_traj,
        )
        w_u_torque = np.full(Nu-2, cfg.u_torques)
        w_u_torque[13:] *= cfg.u_weight_upperbody_scale
        self.add_sensor_cost(
            G1.Sensors.TORQUES,
            quadratic_cost_nb,
            weights=w_u_torque
            )