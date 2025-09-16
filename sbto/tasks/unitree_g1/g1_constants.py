XML_DIR_PATH = "sbto/models/unitree_g1/"

N_FEET = 2

cnt_sensor_per_foot = 3
cnt_sensor_dim_per_foot = 1 + 3

class G1Sensors:
    FEET_CONTACTS = [
        "left_foot1",
        "left_foot2",
        "left_foot3",
        "right_foot1",
        "right_foot2",
        "right_foot3",
    ]
    FEET_POS = [
        "left_foot_pos",
        "right_foot_pos",
    ]
    FEET_VEL = [
        "left_foot_vel",
        "right_foot_vel",
    ]
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
    
    cnt_status_id = [
        i for i in
        range(
            0,
            N_FEET * cnt_sensor_per_foot * cnt_sensor_dim_per_foot, 
            cnt_sensor_dim_per_foot
            )
        ]
    cnt_force_id = [
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
    (-0.4, 0.4),
    (-0.2, 0.2),
    # Right leg.
    (-1.57, 1.57),
    (-0.5, 0.5),
    (-0.5, 0.5),
    (0, 1.57),
    (-0.4, 0.4),
    (-0.2, 0.2),
    # Waist.
    (-0.5, 0.5),
    # Left shoulder.
    (-1.57, 1.57),
    (0., 1.57),
    (-1, 1),
    (-0.1, 2.0944),
    (-1.97222, 1.97222),
    # Right shoulder.
    (-1.57, 1.57),
    (-1.57, 0.),
    (-1, 1),
    (-0.1, 2.0944),
    (-1.97222, 1.97222),
)