from .arena_dataset import ArenaDataset
import numpy as np
from tqdm import tqdm

MAPPING = ArenaDataset.BODY_PART_2_INDEX

def get_kinematics(keypoints, mean=True):
    """
    Collect speed, acceleration, mean, head and tail angular velocities angular accelerations, from the keypoints.

    Args:
        keypoints (dict): dictionary with keypoints of all runs
    Returns:
        dict: dictionary with the collected kinematics for each run
    """
    kinematics = {}
    for run, keypoints in tqdm(keypoints.items(), desc="Processing runs"):
        kinematics[run] = {
            'speed': get_speed(keypoints),
            'acceleration': get_acceleration(keypoints),
            'nt_angular_velocity': get_nt_angular_velocity(keypoints),
            'nt_angular_acceleration': get_nt_angular_acceleration(keypoints),
            'h_angular_velocity': get_h_angular_velocity(keypoints),
            'h_angular_acceleration': get_h_angular_acceleration(keypoints),
            't_angular_velocity': get_t_angular_velocity(keypoints),
            't_angular_acceleration': get_t_angular_acceleration(keypoints)
        }
    return kinematics


def get_speed(keypoints):
    """
    Collect center speed from the keypoints.

    Args:
        keypoints (array): array with keypoints of a run (num_frames, num_keypoints, 2)
    Returns:
        array: array with the speed for each frame (num_frames-1,)
    """
    deltas = np.diff(keypoints[:, MAPPING['mouse_center'], :], axis=0)
    speed = np.linalg.norm(deltas, axis=1)
    return speed

def get_acceleration(keypoints):
    """
    Collect acceleration from the keypoints.

    Args:
        keypoints (array): array with keypoints of a run (num_frames, num_keypoints, 2)
    Returns:
        array: array with the acceleration for each frame (num_frames-2,)
    """
    speed = get_speed(keypoints)
    acceleration = np.diff(speed)
    return acceleration

def get_nt_angular_velocity(keypoints):
    """
    Collect neck to start of tail angular velocity from the keypoints.

    Args:
        keypoints (array): array with keypoints of a run (num_frames, num_keypoints, 2)
    Returns:
        array: array with the angular velocity for each frame (num_frames-1,)
    """
    neck_vector = keypoints[:, MAPPING['neck'], :] - keypoints[:, MAPPING['mouse_center'], :]
    tail_vector = keypoints[:, MAPPING['tail_base'], :] - keypoints[:, MAPPING['mouse_center'], :]
    angles = np.arctan2(neck_vector[:, 1], neck_vector[:, 0]) - np.arctan2(tail_vector[:, 1], tail_vector[:, 0])
    angular_velocity = np.diff(angles)
    return angular_velocity

def get_nt_angular_acceleration(keypoints):
    """
    Collect neck to start of tail angular acceleration from the keypoints.

    Args:
        keypoints (array): array with keypoints of a run (num_frames, num_keypoints, 2)
    Returns:
        array: array with the angular acceleration for each frame (num_frames-2,)
    """
    angular_velocity = get_nt_angular_velocity(keypoints)
    angular_acceleration = np.diff(angular_velocity)
    return angular_acceleration
    

def get_h_angular_velocity(keypoints):
    """
    Collect head angular velocity from the keypoints.

    Args:
        keypoints (array): array with keypoints of a run (num_frames, num_keypoints, 2)
    Returns:
        array: array with the head angular velocity for each frame (num_frames-1,)
    """
    head_vector = keypoints[:, MAPPING['nose'], :] - keypoints[:, MAPPING['neck'], :]
    angles = np.arctan2(head_vector[:, 1], head_vector[:, 0])
    angular_velocity = np.diff(angles)
    return angular_velocity

def get_h_angular_acceleration(keypoints):
    """
    Collect head angular acceleration from the keypoints.

    Args:
        keypoints (array): array with keypoints of a run (num_frames, num_keypoints, 2)
    Returns:
        array: array with the head angular acceleration for each frame (num_frames-2,)
    """
    angular_velocity = get_h_angular_velocity(keypoints)
    angular_acceleration = np.diff(angular_velocity)
    return angular_acceleration

def get_t_angular_velocity(keypoints):
    """
    Collect tail angular velocity from the keypoints.

    Args:
        keypoints (array): array with keypoints of a run (num_frames, num_keypoints, 2)
    Returns:
        array: array with the tail angular velocity for each frame (num_frames-1,)
    """
    tail_vector = keypoints[:, MAPPING['tail_base'], :] - keypoints[:, MAPPING['tail_end'], :]
    angles = np.arctan2(tail_vector[:, 1], tail_vector[:, 0])
    angular_velocity = np.diff(angles)
    return angular_velocity

def get_t_angular_acceleration(keypoints):
    """
    Collect tail angular acceleration from the keypoints.

    Args:
        keypoints (array): array with keypoints of a run (num_frames, num_keypoints, 2)
    Returns:
        array: array with the tail angular acceleration for each frame (num_frames-2,)
    """
    angular_velocity = get_t_angular_velocity(keypoints)
    angular_acceleration = np.diff(angular_velocity)
    return angular_acceleration

