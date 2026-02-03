from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_perception"),
            "config",
            "block_detection_tracking.yaml",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
            ),
            Node(
                package="concrete_block_perception",
                executable="block_detection_tracking_node",
                name="block_detection_tracking_node",
                parameters=[
                    params_file,
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    },
                ],
                remappings=[
                    ("detections", "/yolos_segmentor/detections"),
                    ("masks", "/yolos_segmentor/mask"),
                ],
            ),
        ]
    )
