import os
import mujoco
import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass, field

from sbto.sim.sim_base import Array
from sbto.sim.model_editor import ModelEditor

IntArray = npt.NDArray[np.int64]

MJ_JNT_FREE = 0
FREE_JOINT_SIZE = 7

@dataclass
class ConfigMjScene():
    """
    Configuration class for defining a MuJoCo scene.

    Attributes
    ----------
    xml_scene_path : str
        Path to the scene or MJCF file describing the simulation environment.
    
    xml_sensors_path : str
        Sensors from this MJCF file will be added to the model.
    
    xml_contact_pairs_path : str
        Contact pairs from this MJCF file will be added to the model.
    
    xml_keyrfames_path : str
        Keyframes from this MJCF file will be added to the model.
    
    add_body : dict
        Dictionary specifying additional bodies to add to the scene.
        Each key is a body name, and each value is a dictionary describing its properties:
            {
                "type": str,              # Geometry type ("box", "cylinder", "sphere", etc.)
                "size": tuple,            # Size parameters (depends on geom type)
                "pos": tuple,             # Position in world coordinates (x, y, z)
                "euler": tuple,           # Orientation in Euler angles (rx, ry, rz)
                "rgba": tuple,            # Color and transparency (r, g, b, a)
                "freejoint": bool,        # True for dynamic/free body, False if static
                "mass": float,            # Body mass (kg)
                "priority": int,          # Rendering priority
                "contype": int,           # Contact type bitmask
                "conaffinity": int,       # Contact affinity bitmask
                "solref": tuple,          # Solver reference parameters
                "friction": tuple,        # Contact friction coefficients (slide, spin, roll)
            }
        Additional MuJoCo geom parameters can also be included.

    scale_q : float | tuple
        Scale factor or range for randomizing initial positions (qpos).

    scale_v : float | tuple
        Scale factor or range for randomizing initial velocities (qvel).

    is_floating_base : bool
        If True, treat the main object as having a free-floating base (6-DOF).

    obj_qpos_id : tuple
        Indices in the qpos vector corresponding to the target object.

    N_rollout_steps : int
        Number of simulation steps to perform per rollout.

    obj_x_range : Tuple[float, float]
        Range of x-axis randomization for object initialization.

    obj_y_range : Tuple[float, float]
        Range of y-axis randomization for object initialization.

    obj_z_range : Tuple[float, float]
        Range of z-axis randomization for object initialization.

    obj_w_range : Tuple[float, float]
        Range of object orientation (yaw or quaternion w-component) randomization.
    """
    # base scene path
    xml_scene_path: str
    xml_sensors_path: str = ""
    xml_contact_pairs_path: str = ""
    xml_keyframes_path: str = ""
    add_body: Dict[str, Any] = field(default_factory=lambda: {})
    # random state initialization
    scale_q : tuple = (0.1, )
    scale_v : tuple = (0.1, )
    obj_x_range: Tuple[float, float] = (0.0, 0.0)
    obj_y_range: Tuple[float, float] = (0.0, 0.0)
    obj_z_range: Tuple[float, float] = (0.0, 0.0)
    obj_w_range: Tuple[float, float] = (0.0, 0.0)
    _target_:str = "sbto.sim.scene_mj.MjScene"

    # def __post_init__(self):
    #     self._filename = "config_scene.yaml"
    #     self.class_path = "sbto.sim.scene_mj.MjScene"

class MjScene():
    def __init__(self, cfg: ConfigMjScene):
        self.cfg = cfg
        # Model editing for downstream tasks
        self.mj_model = None
        self.mj_data = None
        self.edit = ModelEditor(cfg.xml_scene_path, self._update_model_data)
        self._init_scene()

    @property
    def Nq(self) -> int:
        return self.mj_model.nq
    
    @property
    def Nv(self) -> int:
        return self.mj_model.nv
    
    @property
    def Nx(self) -> int:
        return self.mj_model.nv + self.mj_model.nq
    
    @property
    def Nu(self) -> int:
        return self.mj_model.nu

    @property
    def Nobs(self) -> int:
        return self.mj_model.nsensordata
    
    @property
    def dt(self) -> float:
        return self.mj_model.opt.timestep
    
    @property
    def act_joint_ids(self) -> Array:
        return self.mj_model.actuator_trnid[:, 0]
    
    @property
    def act_qposadr(self) -> Array:
        return self.mj_model.jnt_qposadr[self.act_joint_ids]

    @property
    def act_dofadr(self) -> Array:
        return self.mj_model.jnt_dofadr[self.act_joint_ids]
    
    @property
    def act_pos_adr(self) -> Array:
        return self.mj_model.jnt_qposadr[self.act_joint_ids]

    @property
    def act_vel_adr(self) -> Array:
        return self.Nq + self.mj_model.jnt_dofadr[self.act_joint_ids]

    @property
    def base_joint_id(self) -> int:
        """Return joint id of object free joint (if any)."""
        MJ_JNT_FREE = 0
        free_joints_id = np.where(self.mj_model.jnt_type == MJ_JNT_FREE)[0]
        if len(free_joints_id) > 0:
            return free_joints_id[0]
        return None

    @property
    def base_pos_adr(self) -> np.ndarray:
        """3D position of object."""
        if self.base_joint_id is None:
            return np.array([], dtype=int)
        base = self.mj_model.jnt_qposadr[self.base_joint_id]
        return np.arange(base, base + 3)

    @property
    def base_quat_adr(self) -> np.ndarray:
        """Quaternion of object."""
        if self.base_joint_id is None:
            return np.array([], dtype=int)
        base = self.mj_model.jnt_qposadr[self.base_joint_id]
        return np.arange(base + 3, base + 7)

    @property
    def base_v_adr(self) -> np.ndarray:
        """Linear velocity address of object in qvel."""
        if self.base_joint_id is None:
            return np.array([], dtype=int)
        adr = self.Nq + self.mj_model.jnt_dofadr[self.base_joint_id]
        return np.arange(adr, adr + 3)

    @property
    def base_w_adr(self) -> np.ndarray:
        """Angular velocity address of object in qvel."""
        if self.base_joint_id is None:
            return np.array([], dtype=int)
        adr = self.Nq + self.mj_model.jnt_dofadr[self.base_joint_id]
        return np.arange(adr + 3, adr + 6)

    @property
    def obj_joint_id(self) -> int:
        """Return joint id of object free joint (if any)."""
        MJ_JNT_FREE = 0
        free_joints_id = np.where(self.mj_model.jnt_type == MJ_JNT_FREE)[0]
        if len(free_joints_id) > 1:
            return free_joints_id[1]
        return None
    
    @property
    def obj_qpos_adr(self) -> List[int]:
        """Return qpos address range for the object's free joint (7D: pos + quat)."""
        if self.obj_joint_id is None:
            return []
        adr = self.mj_model.jnt_qposadr[self.obj_joint_id]
        return list(range(adr, adr + 7))
    
    @property
    def obj_pos_adr(self) -> np.ndarray:
        """3D position of object."""
        if self.obj_joint_id is None:
            return np.array([], dtype=int)
        base = self.mj_model.jnt_qposadr[self.obj_joint_id]
        return np.arange(base, base + 3)

    @property
    def obj_quat_adr(self) -> np.ndarray:
        """Quaternion of object."""
        if self.obj_joint_id is None:
            return np.array([], dtype=int)
        base = self.mj_model.jnt_qposadr[self.obj_joint_id]
        return np.arange(base + 3, base + 7)

    @property
    def obj_v_adr(self) -> np.ndarray:
        """Linear velocity address of object in qvel."""
        if self.obj_joint_id is None:
            return np.array([], dtype=int)
        adr = self.Nq + self.mj_model.jnt_dofadr[self.obj_joint_id]
        return np.arange(adr, adr + 3)

    @property
    def obj_w_adr(self) -> np.ndarray:
        """Angular velocity address of object in qvel."""
        if self.obj_joint_id is None:
            return np.array([], dtype=int)
        adr = self.Nq + self.mj_model.jnt_dofadr[self.obj_joint_id]
        return np.arange(adr + 3, adr + 6)
    
    @property
    def is_floating_base(self) -> bool:
        return self.act_qposadr[0] > 0
    
    @property
    def is_obj(self) -> bool:
        MJ_JNT_FREE = 0
        free_joints_id = np.argwhere(self.mj_model.jnt_type == MJ_JNT_FREE)
        for free_joint_i in free_joints_id:
            if free_joint_i > 0:
                return True
        return False

    @property
    def obj_qpos_adr(self) -> List[int]:
        MJ_JNT_FREE = 0
        FREE_JOINT_SIZE = 7
        obj_qpos_adr = []
        free_joints_id = np.argwhere(self.mj_model.jnt_type == MJ_JNT_FREE)
        for free_joint_i in free_joints_id:
            if free_joint_i > 0:
                obj_qpos_adr = list(free_joint_i + np.arange(FREE_JOINT_SIZE))
        return obj_qpos_adr
    
    @property
    def q_min(self) -> Array:
        return np.array(self.mj_model.jnt_range)[self.act_joint_ids, 0]
    
    @property
    def q_max(self) -> Array:
        return np.array(self.mj_model.jnt_range)[self.act_joint_ids, 1]
    
    def update_data(self, qpos: Array, qvel: Array) -> None:
        self.mj_data.qpos = np.copy(qpos)
        self.mj_data.qvel = np.copy(qvel)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _update_model_data(self, mj_model):
        self.mj_model = mj_model
        self.mj_data = mujoco.MjData(self.mj_model)

    def _init_scene(self):
        self.add_body()
        self.add_keyframes()
        self.add_contact_pairs()    
        self.add_sensors()

    def add_body(self):
        INVALID_KWARGS = [
            "type"
        ]

        for body_name, kwargs in self.cfg.add_body.items():
            # get geom type
            geom_type = kwargs["type"]
            if not geom_type in self.edit.AVAILABLE_GEOM:
                geom_type = "box"

            geom_kwargs = {k:v for k, v in kwargs.items() if not k in INVALID_KWARGS}

            match geom_type:
                case "box":
                    self.edit.add_box(name=body_name, **geom_kwargs)
                case "cylinder":
                    self.edit.add_cylinder(name=body_name, **geom_kwargs)
                case "sphere":
                    self.edit.add_sphere(name=body_name, **geom_kwargs)
    
    def add_keyframes(self):
        path = self.cfg.xml_keyframes_path
        if path:
            if os.path.exists(path):
                self.edit.add_keyframes_from_file(path)
            else:
                print(f"MJCF file {path} not found")

    def add_sensors(self):
        path = self.cfg.xml_sensors_path
        if path:
            if os.path.exists(path):
                self.edit.add_sensors_from_file(path)
            else:
                print(f"MJCF file {path} not found")

    def add_contact_pairs(self):
        path = self.cfg.xml_contact_pairs_path
        if path:
            if os.path.exists(path):
                self.edit.add_cnt_pairs_from_file(path)
            else:
                print(f"MJCF file {path} not found")