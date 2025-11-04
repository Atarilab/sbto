import numpy as np
import numpy.typing as npt
from typing import Dict
import mujoco

Array = npt.NDArray[np.float64]

MJ_JNT_FREE = 0

def split_x_traj(xml_path, x_traj : Array) -> Dict[str, Array]:
    """
    Split x_traj data into subarrays:
    - base_xyz_quat, base_linvel_angvel
    - actuator_pos, actuator_vel
    - obj_i_xyz_quat, obj_i_linvel_angvel
    """
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    name2id = {}
    nq = mj_model.nq

    # Actuated joints
    act_joint_ids = mj_model.actuator_trnid[:, 0]
    act_qposadr = mj_model.jnt_qposadr[act_joint_ids]
    act_dofadr = mj_model.jnt_dofadr[act_joint_ids]
    
    name2id["actuator_pos"] = act_qposadr
    name2id["actuator_vel"] = act_dofadr + nq
    
    # Free joints
    SIZE_POS, SIZE_VEL = 7, 6
    POS_POSTFIX = "_xyz_quat"
    VEL_POSTFIX = "_linvel_angvel"
    free_joints_id = np.argwhere(mj_model.jnt_type == MJ_JNT_FREE)
    for i, free_joint_i in enumerate(free_joints_id):
        if i == 0:
            name = "base"
        else:
            name = f"obj_{i-1}"
        start_qpos = mj_model.jnt_qposadr[free_joint_i][0]
        start_qvel = mj_model.jnt_dofadr[free_joint_i][0]
        name2id[name + POS_POSTFIX] = np.arange(start_qpos, start_qpos + SIZE_POS)
        name2id[name + VEL_POSTFIX] = np.arange(start_qvel, start_qvel + SIZE_VEL) + nq

    n_extracted_joints = 0
    for name, id in name2id.items():
        n_extracted_joints += np.sum(np.shape(id))

    if n_extracted_joints != x_traj.shape[-1]:
        raise ValueError(f"Missing extracted joints (got {n_extracted_joints}, should be {x_traj.shape[-1]})")

    # Extract data
    extracted_data = {}
    n_dim_traj = x_traj.ndim
    for name, id in name2id.items():
        if n_dim_traj == 2:
            id_ = np.atleast_2d(id)
        elif n_dim_traj == 3:
            id_ = np.atleast_3d(id)
        extracted_data[name] = np.take_along_axis(x_traj, id_, axis=-1)

    return extracted_data

def expand_traj_actuated_and_objects(
    x: np.ndarray,
    xml_src: mujoco.MjModel,
    xml_dst: mujoco.MjModel,
    use_model_defaults=True,
):
    """
    Expand x from model_src → model_dst, assuming:
      - free base joint exists & matches
      - additional free joints = objects (count may differ)
      - only actuated joints need mapping (name-based)
      - object states padded/truncated as needed
    """
    model_src = mujoco.MjModel.from_xml_path(xml_src)
    model_dst = mujoco.MjModel.from_xml_path(xml_dst)

    T, Nx = x.shape

    # === Locate free joints ===
    free_src = np.where(model_src.jnt_type == MJ_JNT_FREE)[0]
    free_dst = np.where(model_dst.jnt_type == MJ_JNT_FREE)[0]

    is_base_src = min(len(free_src), 1)
    is_base_dst = min(len(free_dst), 1)

    n_obj_src = len(free_src) - 1
    n_obj_dst = len(free_dst) - 1

    FREE_QPOS = 7
    FREE_QVEL = 6

    base_qpos_src_dim = is_base_src * FREE_QPOS
    base_qvel_src_dim = is_base_src * FREE_QVEL
    base_qpos_dst_dim = is_base_dst * FREE_QPOS
    base_qvel_dst_dim = is_base_dst * FREE_QVEL
    obj_qpos_src_dim = n_obj_src * FREE_QPOS
    obj_qvel_src_dim = n_obj_src * FREE_QVEL
    obj_qpos_dst_dim = n_obj_dst * FREE_QPOS
    obj_qvel_dst_dim = n_obj_dst * FREE_QVEL

    # === Robot joint DOFs ===
    joint_dofs_src = model_src.nq - (base_qpos_src_dim + obj_qpos_src_dim)
    joint_dofs_dst = model_dst.nq - (base_qpos_dst_dim + obj_qpos_dst_dim)

    # === Split source x into components ===
    qpos_src = x[:, :base_qpos_src_dim + joint_dofs_src + obj_qpos_src_dim]
    qvel_src = x[:, base_qpos_src_dim + joint_dofs_src + obj_qpos_src_dim :
                    base_qpos_src_dim + joint_dofs_src + obj_qpos_src_dim +
                    base_qvel_src_dim + joint_dofs_src + obj_qvel_src_dim]

    qpos_base = qpos_src[:, :base_qpos_src_dim]
    qvel_base = qvel_src[:, :base_qvel_src_dim]

    qpos_joint_src = qpos_src[:, base_qpos_src_dim : base_qpos_src_dim + joint_dofs_src]
    qvel_joint_src = qvel_src[:, base_qvel_src_dim : base_qvel_src_dim + joint_dofs_src]

    qpos_obj_src = qpos_src[:, base_qpos_src_dim + joint_dofs_src :]
    qvel_obj_src = qvel_src[:, base_qvel_src_dim + joint_dofs_src :]

    # === Build joint name lists ===
    def joint_names(model):
        return [model.joint(j).name for j in range(model.njnt)]

    names_src = joint_names(model_src)
    names_dst = joint_names(model_dst)

    # Remove free joints from name lists (we skip them)
    actuated_src = names_src[is_base_src:len(names_src)-n_obj_src]
    actuated_dst = names_dst[is_base_dst:len(names_dst)-n_obj_dst]

    assert len(actuated_src) == joint_dofs_src
    assert len(actuated_dst) == joint_dofs_dst

    # === Allocate dst actuated storage ===
    qpos_joint_dst = np.zeros((T, joint_dofs_dst))
    qvel_joint_dst = np.zeros((T, joint_dofs_dst))

    # === Copy mapped actuated joints ===
    for j_dst, name in enumerate(actuated_dst):
        if name in actuated_src:
            j_src = actuated_src.index(name)
            qpos_joint_dst[:, j_dst] = qpos_joint_src[:, j_src]
            qvel_joint_dst[:, j_dst] = qvel_joint_src[:, j_src]
        else:
            if use_model_defaults:
                qpos_joint_dst[:, j_dst] = model_dst.qpos0[base_qpos_src_dim + j_dst]
            else:
                qpos_joint_dst[:, j_dst] = 0.0
            qvel_joint_dst[:, j_dst] = 0.0

    # === Handle objects (pad/truncate) ===
    qpos_obj_dst = np.zeros((T, obj_qpos_dst_dim))
    qvel_obj_dst = np.zeros((T, obj_qvel_dst_dim))

    # Copy existing object states (min source/dest)
    copy_n = min(n_obj_src, n_obj_dst)
    if copy_n > 0:
        qpos_obj_dst[:, :copy_n * FREE_QPOS] = \
            qpos_obj_src[:, :copy_n * FREE_QPOS]
        qvel_obj_dst[:, :copy_n * FREE_QVEL] = \
            qvel_obj_src[:, :copy_n * FREE_QVEL]

    # If dest has *more* objects, they stay zeros / defaults
    if n_obj_dst > n_obj_src and use_model_defaults:
        for k in range(n_obj_src, n_obj_dst):
            base_idx = base_qpos_dst_dim + joint_dofs_dst + k * FREE_QPOS
            qpos_obj_dst[:, k * FREE_QPOS : (k + 1) * FREE_QPOS] = \
                model_dst.qpos0[base_idx : base_idx + FREE_QPOS]

    # === Recombine state ===
    qpos_dst = np.hstack([qpos_base, qpos_joint_dst, qpos_obj_dst])
    qvel_dst = np.hstack([qvel_base, qvel_joint_dst, qvel_obj_dst])

    x_new = np.hstack([qpos_dst, qvel_dst])
    return x_new

def post_process_for_rl(
    traj_file_path: str,
    src_model_path: str,
    dst_model_path: str,
    ) -> None:
    """
    Save trajectory data for RL downstream tasks.
    """
    file = np.load(traj_file_path)
    time = file["time"]
    x_traj = file["x"]
    x_new_traj = expand_traj_actuated_and_objects(x_traj, src_model_path, dst_model_path)
    data_dict = split_x_traj(dst_model_path, x_new_traj)
    
    new_file_path = traj_file_path.replace(".npz", "_rl_format.npz")
    np.savez_compressed(
        new_file_path,
        time = time,
        **data_dict
    )
