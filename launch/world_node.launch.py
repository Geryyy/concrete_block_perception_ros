import os
import subprocess
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from ament_index_python import get_package_share_directory
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

from launch import LaunchDescription


def generate_launch_description():
    pkg_dir = FindPackageShare("concrete_block_perception")

    return LaunchDescription(
        [
            Node(
                package="concrete_block_perception",
                executable="world_model_node",
                name="block_world_model_node",
                output="screen",
                parameters=[
                    {
                        # "calib_yaml": "",  # optional override
                        "world_frame": "world",
                        "assoc_dist": 0.6,
                        "min_points": 30,
                    }
                ],
                remappings=[
                    ("detections", "/yolos_segmentor/detections"),
                    ("points", "/seyond_points"),
                    ("image", "/yolos_segmentor/mask"),
                ],
            ),
        ]
    )
