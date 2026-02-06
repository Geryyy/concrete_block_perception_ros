from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


# usage:
# ros2 launch concrete_block_perception block_world_model_bag.launch.py \
#   bag:=/path/to/your/bag


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    bag_path = LaunchConfiguration("bag")
    world_model_params = LaunchConfiguration("world_model_params")
    block_detection_tracking_params = LaunchConfiguration(
        "block_detection_tracking_params"
    )

    # -----------------------
    # Launch arguments
    # -----------------------
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (ROS) time",
    )

    declare_bag_path = DeclareLaunchArgument(
        "bag",
        default_value="/home/vscode/Documents/2025-12-17/pzs_crane_1_pickup",
        description="Path to rosbag to replay",
    )

    declare_block_detection_tracking_params = DeclareLaunchArgument(
        "block_detection_tracking_params",
        default_value=PathSubstitution(FindPackageShare("concrete_block_perception"))
        / "config"
        / "block_detection_tracking.yaml",
        description="YAML parameter file for block detection tracking node",
    )

    declare_world_model_params = DeclareLaunchArgument(
        "world_model_params",
        default_value=PathSubstitution(FindPackageShare("concrete_block_perception"))
        / "config"
        / "world_model.yaml",
        description="YAML parameter file for world model node",
    )

    # -----------------------
    # Perception pipeline
    # -----------------------
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathSubstitution(FindPackageShare("concrete_block_perception"))
            / "launch"
            / "perception.launch.py"
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # -----------------------
    # Rosbag replay helper nodes
    # -----------------------
    rosbag_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathSubstitution(FindPackageShare("epsilon_crane_bringup_hmi"))
            / "launch"
            / "rosbag_replay_nodes.launch.py"
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
        }.items(),
    )

    # -----------------------
    # block detection tracking node (WITH YAML)
    # -----------------------
    block_detection_tracker = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathSubstitution(FindPackageShare("concrete_block_perception"))
            / "launch"
            / "detection_tracking.launch.py"
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": block_detection_tracking_params,
        }.items(),
    )

    # -----------------------
    # World model (WITH YAML)
    # -----------------------
    world_node_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathSubstitution(FindPackageShare("concrete_block_perception"))
            / "launch"
            / "world_node.launch.py"
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": world_model_params,
        }.items(),
    )

    # -----------------------
    # Rosbag play (delayed!)
    # -----------------------
    rosbag_play = TimerAction(
        period=3.0,  # ensure TF + nodes are ready
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "play",
                    bag_path,
                    "--clock",
                ],
                output="screen",
            )
        ],
    )

    # -----------------------
    # Launch description
    # -----------------------
    return LaunchDescription(
        [
            declare_use_sim_time,
            declare_bag_path,
            declare_block_detection_tracking_params,
            declare_world_model_params,
            perception_launch,
            rosbag_nodes_launch,
            block_detection_tracker,
            world_node_launch,
            rosbag_play,
        ]
    )
