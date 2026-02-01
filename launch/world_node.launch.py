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

    params_file = PathJoinSubstitution(
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
                name="block_world_model_node",
                parameters=[
                    params_file,
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "calib_yaml": calib_yaml,
                    },
                ],
                remappings=[
                    ("detections", "/yolos_segmentor/detections"),
                    ("points", "/seyond_points"),
                    ("image", "/yolos_segmentor/mask"),
                ],
                output="screen",
            ),
        ]
    )
