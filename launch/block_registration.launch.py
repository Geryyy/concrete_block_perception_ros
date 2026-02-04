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
    return LaunchDescription(
        [
            Node(
                package="concrete_block_perception",
                executable="block_registration_node",
                name="block_registration_node",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [
                            FindPackageShare("concrete_block_perception"),
                            "config",
                            "block_registration.yaml",
                        ]
                    )
                ],
            ),
        ]
    )
