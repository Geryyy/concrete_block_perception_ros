from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
            ),
            Node(
                package="concrete_block_perception",
                executable="world_model_node",
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
                output="screen",
            ),
        ]
    )
