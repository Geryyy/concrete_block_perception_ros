from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_perception"),
            "config",
            "block_detection_tracking.yaml",
        ]
    )

    container = ComposableNodeContainer(
        name="detection_tracking_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        emulate_tty=True,
        composable_node_descriptions=[
            ComposableNode(
                package="concrete_block_perception",
                plugin="concrete_block_perception::BlockDetectionTrackingNode",
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
                    ("tracked_detections", "/cbp/tracked_detections"),
                ],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
            ),
            container,
        ]
    )
