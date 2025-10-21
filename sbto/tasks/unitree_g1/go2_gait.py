# sbto/tasks/unitree_g1/g1_gait_go2.py

import os
import numpy as np
import mujoco

from sbto.mj.nlp_mj import NLP_MuJoCo
import sbto.tasks.unitree_g1.g1_constants as const


class Go2_Gait(NLP_MuJoCo):
    
    SCENE = "scene_position.xml"

    def __init__(
        self,
        T,
        Nknots=0,
        interp_kind="linear",
        Nthread=-1,
    ):
        xml_path = os.path.join(const.XML_DIR_PATH, G1_Gait.SCENE)
        super().__init__(xml_path, T, Nknots, interp_kind, Nthread)

        # -------------------- initial state --------------------
        self.set_initial_state_from_keyframe("home")

        # -------------------- model sizes & indices --------------------
        Nq = self.mj_model.nq
        Nv = self.mj_model.nv
        base_q = 7 if self.mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0
        base_v = 6 if base_q == 7 else 0

        # joint qpos (after base) and qvel indices
        idx_joint_pos = np.arange(base_q, Nq)
        Nj = len(idx_joint_pos)
        idx_joint_vel = np.arange(Nq + base_v, Nq + base_v + Nj)

        # -------------------- joint limits--------------------
        qmin, qmax = [], []
        for j in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            lo, hi = self.mj_model.jnt_range[j]
            if lo == 0.0 and hi == 0.0:  
                lo, hi = -1.0, 1.0
            qmin.append(lo); qmax.append(hi)
        self.q_min = np.array(qmin, dtype=float)
        self.q_max = np.array(qmax, dtype=float)

        # nominal joint pose = initial
        self.q_nom = self.x_0[idx_joint_pos]
        self.a_min = self.q_nom - self.q_min
        self.a_max = self.q_max - self.q_nom

        # -------------------- gait targets --------------------
        # forward COM velocity target (m/s)
        self.v_des = np.array([0.5, 0.0, 0.0], dtype=float)

        # time grid for position ramp
        t_grid = np.linspace(0.0, self.duration, num=T)[: self.T - 1]
        xy_ref = t_grid[:, None] * self.v_des[None, :2]  # (T-1, 2)

        idx_base_z = 2
        z0 = self.x_0[idx_base_z]
        z_ref = np.full(self.T - 1, z0, dtype=float)

        # base orientation 
        idx_base_quat = np.arange(3, 7)
        quat_ref = np.tile(self.x_0[idx_base_quat], (self.T - 1, 1))  # (T-1, 4)

        # base linear velocity indices 
        idx_base_linvel = np.arange(Nq, Nq + 3)

        # -------------------- costs --------------------
        # joint pos near nominal 
        self.add_state_cost(
            "joint_pos",
            self.quadratic_cost,
            idx_joint_pos,
            weights=5.0,
            use_intial_as_ref=True,
            weights_terminal=30.0,
        )

        #  joint vel damping
        self.add_state_cost(
            "joint_vel",
            self.quadratic_cost,
            idx_joint_vel,
            weights=1e-3,
        )

        # base height tracking
        self.add_state_cost(
            "base_z",
            self.quadratic_cost,
            idx_base_z,
            ref_values=z_ref,
            weights=30.0,
            weights_terminal=80.0,
        )

        #  base orientation near initial (simple quadratic on quat OK here)
        self.add_state_cost(
            "base_quat",
            self.quadratic_cost,
            idx_base_quat,
            ref_values=quat_ref,
            weights=5.0,
            weights_terminal=50.0,
        )

        #  base linear velocity tracking (vx ~ 0.5 m/s)
        self.add_state_cost(
            "base_linvel",
            self.quadratic_cost,
            idx_base_linvel,
            ref_values=self.v_des,            # broadcasted
            weights=[5.0, 1.0, 1.0],
            weights_terminal=[0.0, 0.0, 50.0],
        )

        #  base XY position ramp (consistent with v_des)
        self.add_state_cost(
            "base_xy",
            self.quadratic_cost,
            [0, 1],
            ref_values=xy_ref,                # (T-1, 2)
            weights=[1.0, 1.0],
            weights_terminal=[200.0, 200.0],
        )

        #  small control effort
        self.add_control_cost(
            "u_traj",
            self.quadratic_cost,
            idx=list(range(self.Nu)),
            weights=0.01,
        )

    @staticmethod
    def contact_cost(cnt_status_rollout, cnt_plan, weights) -> float:
        cnt_status_rollout[cnt_status_rollout > 1] = 1
        return np.sum(
            weights[None, ...] * np.float32(cnt_status_rollout != cnt_plan[None, ...]),
            axis=(-1, -2),
        )

    @staticmethod
    def quat_dist(var, ref, weights) -> float:
        return np.sum(
            weights[:, 0] * (1.0 - np.square(np.sum(var * ref[None, ...], axis=-1))),
            axis=(-1),
        )

    def get_q_des_from_u_traj(self, act):
        act = np.clip(act, -1.0, 1.0)
        q_des = np.where(act < 0, act * self.a_min, act * self.a_max) + self.q_nom
        return q_des