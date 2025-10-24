XML_DIR_PATH = "sbto/models/unitree_g1/"

N_FEET = 2
N_HANDS = 2

cnt_sensor_per_foot = 3
# found force
cnt_sensor_dim_per_foot = 1 + 3
cnt_sensor_per_hand = 1
# found force torque
cnt_sensor_dim_per_hand = 1 + 3 + 3

class Sensors:
    FEET_CONTACTS = [
        "left_foot1",
        "left_foot2",
        "left_foot3",
        "right_foot1",
        "right_foot2",
        "right_foot3",
    ]
    HAND_CONTACTS = [
        "left_hand_cnt",
        "right_hand_cnt",
    ]
    OBJ_TABLE_CONTACT = ["obj_table_cnt"]
    OBJ_FLOOR_CONTACT = ["obj_floor_cnt"]
    FEET_POS = [
        "left_foot_pos",
        "right_foot_pos",
    ]
    FEET_VEL = [
        "left_foot_vel",
        "right_foot_vel",
    ]
    OBJ_POS = "global_pos_obj"
    OBJ_QUAT = "global_quat_obj"
    BASE_POS = "global_pos_pelvis"
    BASE_QUAT = "orientation_pelvis"
    BASE_UPRIGHT = "upvector_pelvis"
    BASE_LINVEL = "global_linvel_pelvis"
    BASE_ANGVEL = "global_angvel_pelvis"
    TORSO_POS = "global_pos_torso"
    TORSO_QUAT = "orientation_torso"
    TORSO_UPRIGHT = "upvector_torso"
    TORSO_LINVEL = "global_linvel_torso"
    TORSO_ANGVEL = "global_angvel_torso"
    TORQUES = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ]
    
    cnt_status_hand_id = [
        i for i in
        range(
            0,
            N_HANDS * cnt_sensor_per_hand * cnt_sensor_dim_per_hand, 
            cnt_sensor_dim_per_hand
            )
        ]
    cnt_force_hand_id = [
        i for i in
        range(N_HANDS * cnt_sensor_per_hand * cnt_sensor_dim_per_hand)
        if not i % cnt_sensor_dim_per_hand in [0, 4, 5, 6]
        ]
    cnt_torque_hand_id = [
        i for i in
        range(N_HANDS * cnt_sensor_per_hand * cnt_sensor_dim_per_hand)
        if not i % cnt_sensor_dim_per_hand in [0, 1, 2, 3]
        ]
    cnt_status_feet_id = [
        i for i in
        range(
            0,
            N_FEET * cnt_sensor_per_foot * cnt_sensor_dim_per_foot, 
            cnt_sensor_dim_per_foot
            )
        ]
    cnt_force_feet_id = [
        i for i in
        range(N_FEET * cnt_sensor_per_foot * cnt_sensor_dim_per_foot)
        if i % cnt_sensor_dim_per_foot != 0
        ]
    
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
    # Right shoulder.
    (-1.57, 1.57),
    (-1.57, 0.2),
    (-1, 1),
    (-1., 1.57),
    (-1., 1.),
)

# --- System DOFs ---
NDOF_G1 = 23  # robot: 23 + 2 wrists
NQ_G1 = 7 + NDOF_G1        # 7 for base pose (3 pos + 4 quat)
NV_G1 = 6 + NDOF_G1        # 6 for base velocity (3 lin + 3 ang)
iNV_G1 = NQ_G1 + 7        # NQ + 7 for obj

# --- Robot joint indices ---
IDX_JOINT_POS = list(range(7, 7 + NDOF_G1))              # qpos indices for joints
IDX_JOINT_VEL = list(range(NQ_G1, NQ_G1 + NDOF_G1))    # qvel indices for joints

# Example subgroups (these depend on your robot’s joint ordering)
IDX_HIP_KNEE = [0, 3, 6, 9]       # example leg joint indices
IDX_SHOULDER_PITCH = [13, 19]     # example shoulder pitch joints
IDX_WAIST = 12                    # example waist joint index

# --- Obj state indices ---
# Obj qpos starts right after the robot qpos (7 + NDOF_G1)
IDX_BOX_POS = list(range(NQ_G1, NQ_G1 + 3))
IDX_BOX_QUAT = list(range(NQ_G1 + 3, NQ_G1 + 7))

# Obj qvel starts right after robot qvel (6 + NDOF_G1)
IDX_BOX_LINVEL = list(range(iNV_G1 + NV_G1, iNV_G1 + NV_G1 + 3))
IDX_BOX_ANGVEL = list(range(iNV_G1 + NV_G1 + 3, iNV_G1 + NV_G1 + 6))