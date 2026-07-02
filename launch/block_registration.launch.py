from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "calib_yaml",
                default_value="calib_zed2i_to_seyond_new_sensor_head.yaml",
                description="Calibration YAML in concrete_block_perception/config.",
            ),
            Node(
                package="concrete_block_perception",
                executable="block_registration_node",
                name="registration_node",
                output="screen",
                parameters=[
                    PathJoinSubstitution(
                        [
                            FindPackageShare("concrete_block_perception"),
                            "config",
                            "block_registration.yaml",
                        ]
                    ),
                    {"calib_yaml": LaunchConfiguration("calib_yaml")},
                ],
            ),
        ]
    )
