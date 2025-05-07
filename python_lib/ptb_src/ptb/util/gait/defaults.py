from enum import Enum


class OsimIKLabels(Enum):
    """
    Gait2392 Joint labels
    """
    left_cols = ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
                 "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
                 "subtalar_angle_l", "mtp_angle_l"]
    right_cols = ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
                  "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r",
                  "subtalar_angle_r", "mtp_angle_r"]
    cols = ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz",
            "hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle",
            "subtalar_angle", "mtp_angle"]