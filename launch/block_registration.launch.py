from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    registration_params = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_perception"),
            "config",
            "block_registration.yaml",
        ]
    )

    container = ComposableNodeContainer(
        name="registration_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",  # multithreaded
        output="screen",
        emulate_tty=True,
        composable_node_descriptions=[
            ComposableNode(
                package="concrete_block_perception",
                plugin="concrete_block_perception::BlockRegistrationNode",
                name="block_registration_node",
                parameters=[registration_params],
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
    )

    return LaunchDescription([container])
