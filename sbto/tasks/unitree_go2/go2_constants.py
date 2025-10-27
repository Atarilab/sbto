# sbto/tasks/unitree_go2/go2_constants.py


XML_DIR_PATH = "sbto/models/go2/"


N_FEET = 4
NDOFS = 12  
cnt_sensor_per_foot = 1

class Sensors:
    FEET_CONTACTS = [
        "FR_foot",
        "FL_foot",
        "RR_foot",
        "RL_foot",
    ]
   
    BASE_POS   = "global_position"
    BASE_QUAT  = "orientation"
    BASE_LINVEL = "global_linvel"
    BASE_ANGVEL = "global_angvel"

    JOINT_POS = [
        "abduction_front_left_pos",
        "hip_front_left_pos",
        "knee_front_left_pos",
        "abduction_hind_left_pos",
        "hip_hind_left_pos",
        "knee_hind_left_pos",
        "abduction_front_right_pos",
        "hip_front_right_pos",
        "knee_front_right_pos",
        "abduction_hind_right_pos",
        "hip_hind_right_pos",
        "knee_hind_right_pos",
    ]
    JOINT_VEL = [
        "abduction_front_left_vel",
        "hip_front_left_vel",
        "knee_front_left_vel",
        "abduction_hind_left_vel",
        "hip_hind_left_vel",
        "knee_hind_left_vel",
        "abduction_front_right_vel",
        "hip_front_right_vel",
        "knee_front_right_vel",
        "abduction_hind_right_vel",
        "hip_hind_right_vel",
        "knee_hind_right_vel",
    ]


    FEET_SITES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    
    FEET_CONTACTS = [
            "FR_foot",
            "FL_foot",
            "RR_foot",
            "RL_foot",
    ]
    cnt_status_id = [0]
    ee_labels = ["FR", "FL", "RR", "RL"] 
    cnt_status_id = [0, 1, 2, 3]
    cnt_force_id = []

