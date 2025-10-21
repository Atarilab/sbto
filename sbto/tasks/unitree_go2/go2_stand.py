import os
import numpy as np
import mujoco

from sbto.mj.nlp_mj import NLP_MuJoCo
import sbto.tasks.unitree_g1.g1_constants as const


class Go2_Stand(NLP_MuJoCo):
    SCENE = "scene_position.xml"

    def __init__(
        self,
        T,
        Nknots=0,
        interp_kind="linear",
        Nthread=-1,
    ):
        xml_path = os.path.join(const.XML_DIR_PATH, Go2_Stand.SCENE)
        super().__init__(xml_path, T, Nknots, interp_kind, Nthread)
        

        # Initial pose from Go2 XML
        keyframe_name = "home"
        self.set_initial_state_from_keyframe(keyframe_name)
        # --- sizes & indices ---
        Nq = self.mj_model.nq
        Nv = self.mj_model.nv
        base_q = 7 if self.mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0
        base_v = 6 if base_q == 7 else 0

        idx_joint_pos = np.arange(base_q, Nq)                 # joint qpos after base
        Nj = len(idx_joint_pos)
        idx_joint_vel = np.arange(Nq + base_v, Nq + base_v + Nj)

        
        # ---------- Sizes & indices (free base aware) ----------
        Nq = self.mj_model.nq
        Nv = self.mj_model.nv
        base_q = 7 if self.mj_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0
        base_v = 6 if base_q == 7 else 0
        idx_joint_pos = np.arange(base_q, Nq)                    # joint qpos after base
        Nj = len(idx_joint_pos)
        idx_joint_vel = np.arange(Nq + base_v, Nq + base_v + Nj) # joint qvel after base v

        # ---------- Joint limits from model (no manual limits) ----------
        jmin, jmax = [], []
        for j in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            lo, hi = self.mj_model.jnt_range[j]
            if lo == 0.0 and hi == 0.0:  # unlimited → fallback
                lo, hi = -1.0, 1.0
            jmin.append(lo); jmax.append(hi)
        self.q_min = np.array(jmin)
        self.q_max = np.array(jmax)

        # Nominal pose from initial state
        self.q_nom = self.x_0[idx_joint_pos]
        self.a_min = self.q_nom - self.q_min
        self.a_max = self.q_max - self.q_nom

        # ---------- Standing costs (state-based only) ----------
        # Keep joints near initial
        self.add_state_cost(
            "joint_pos",
            self.quadratic_cost,
            idx_joint_pos,
            weights=30.0,
            use_intial_as_ref=True,
            weights_terminal=50.0,
        )

        # Dampen joint velocities
        self.add_state_cost(
            "joint_vel",
            self.quadratic_cost,
            idx_joint_vel,
            weights=0.001,
        )

        # Keep base height near initial (z at qpos index 2)
        idx_base_z = 2
        z_ref = np.full(self.T - 1, self.x_0[idx_base_z])
        self.add_state_cost(
            "base_z",
            self.quadratic_cost,
            idx_base_z,
            ref_values=z_ref,
            weights=30.0,
            weights_terminal=60.0,
        )

        # Keep base orientation near initial (quat qpos 3:7)
        idx_base_quat = np.arange(3, 7)
        quat_ref = np.tile(self.x_0[idx_base_quat], (self.T - 1, 1))
        self.add_state_cost(
            "base_quat",
            self.quadratic_cost,
            idx_base_quat,
            ref_values=quat_ref,
            weights=5.0,
            weights_terminal=50.0,
        )

        # Dampen base angular velocity (ω = 3:6 in qvel; offset by Nq in state vector)
        idx_base_angvel = np.arange(Nq + 3, Nq + 6)
        self.add_state_cost(
            "base_angvel",
            self.quadratic_cost,
            idx_base_angvel,
            weights=1.0,
            weights_terminal=10.0,
        )

        # Small control effort
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
        # Kept for API compatibility (unused here)
        return np.sum(
            weights[:, 0] * (1.0 - np.square(np.sum(var * ref[None, ...], axis=-1))),
            axis=(-1),
        )

    def get_q_des_from_u_traj(self, act):
        action_scale = 1.0
        act = np.clip(act * action_scale, -1.0, 1.0)
        q_des = np.where(act < 0, act * self.a_min, act * self.a_max) + self.q_nom
        return q_des