from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    calib_yaml = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_perception"),
            "config",
            "calib_zed2i_to_seyond.yaml",
        ]
    )

    world_model_params = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_perception"),
            "config",
            "world_model.yaml",
        ]
    )

    container = ComposableNodeContainer(
        name="world_model_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        emulate_tty=True,
        composable_node_descriptions=[
            ComposableNode(
                package="concrete_block_perception",
                plugin="concrete_block_perception::WorldModelNode",
                name="world_model_node",
                parameters=[
                    world_model_params,
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "calib_yaml": calib_yaml,
                    },
                ],
                remappings=[
                    ("tracked_detections", "/cbp/tracked_detections"),
                    ("points", "/seyond_points"),
                    ("block_world_model", "/cbp/block_world_model"),
                    ("block_world_model_markers", "/cbp/block_world_model_markers"),
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
