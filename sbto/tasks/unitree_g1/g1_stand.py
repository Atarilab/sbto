import os
import numpy as np
from sbto.mj.nlp_mj import NLP_MuJoCo
import sbto.tasks.unitree_g1.g1_constants as const

class G1_Stand(NLP_MuJoCo):
    SCENE = "scene_mjx_23dof_custom_collisions.xml"
    def __init__(self,
                 T,
                 Nknots = 0,
                 interp_kind="linear",
                 Nthread = -1
                 ):
        xml_path = os.path.join(const.XML_DIR_PATH, G1_Stand.SCENE)
        super().__init__(xml_path, T, Nknots, interp_kind, Nthread)

        keyframe_name = "knees_bent"
        self.set_initial_state_from_keyframe(keyframe_name)

        self.q_min = np.array(const.RESTRICTED_JOINT_RANGE)[:, 0]
        self.q_max = np.array(const.RESTRICTED_JOINT_RANGE)[:, 1]
        self.a = 0.5 * (self.q_min + self.q_max)
        self.b = 0.5 * (self.q_max - self.q_min)

        idx_joint_pos = np.arange(7, 7 + 23)
        self.add_state_cost(
            "joint_pos",
            self.quadratic_cost,
            idx_joint_pos,
            self.x_0[idx_joint_pos],
            1.,
            )
        
        self.add_state_cost(
            "height",
            self.quadratic_cost,
            2,
            self.x_0[2],
            100.,
            terminal=True
            )
        
        