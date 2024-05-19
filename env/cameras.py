"""Camera configs."""

import numpy as np
import pybullet as p


class RealSenseD435:
    """Default configuration with 3 RealSense RGB-D cameras.
    https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html
    camera_image_size = (480, 640)
    camera_fov_w = 69.4  # horizontal field of view, width of image
    camera_focal_length = (float(camera_image_size[1]) / 2) / np.tan((np.pi * camera_fov_w / 180) / 2)
    camera_focal_length = 462.14
    """

    # Mimic RealSense D435 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = np.array([[462.14, 0, 320], [0, 462.14, 240], [0, 0, 1]])

    # Set default camera poses. 
    # relative to the manipulator
    front_position = (1.0, 0, 0.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    # back_position = (0.2, 0, 0.75)
    # back_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    # back_rotation = p.getQuaternionFromEuler(back_rotation)
    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)

    # Default camera configs.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": left_position,
            "rotation": left_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": right_position,
            "rotation": right_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
    ]

class RealSenseD455:
    """Default configuration with 3 RealSense RGB-D cameras.
    https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html
    camera_image_size = (480, 640)
    camera_fov_w = 69.4  # horizontal field of view, width of image
    camera_focal_length = (float(camera_image_size[1]) / 2) / np.tan((np.pi * camera_fov_w / 180) / 2)
    camera_focal_length = 462.14
    """

    # Mimic RealSense D455 RGB-D camera parameters.
    image_size = (720, 1280)
    intrinsics = np.array([[634.72, 0, 644.216], [0, 634.118, 368.458], [0, 0, 1]])

    # Set default camera poses.
    front_position = (0.5, 0, 0.7)
    front_rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))
    # Default camera configs.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            "zrange": (0.01, 2.0),
            "noise": True,
        },
    ]


class Oracle:
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = np.array([[63e4, 0, 320], [0, 63e4, 240], [0, 0, 1]])
    position = (0.5, 0, 1000.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

    # Camera config.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": position,
            "rotation": rotation,
            "zrange": (999.7, 1001.0),
            "noise": False,
        }
    ]
