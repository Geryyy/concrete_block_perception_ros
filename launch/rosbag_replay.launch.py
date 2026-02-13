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
            perception_launch,
            rosbag_nodes_launch,
            rosbag_play,
        ]
    )
