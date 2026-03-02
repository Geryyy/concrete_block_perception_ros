from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bag_path = LaunchConfiguration("bag")
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_bag = DeclareLaunchArgument(
        "bag",
        default_value="/home/vscode/Documents/2025-12-17/pzs_crane_1_pickup",
        description="Path to rosbag to replay",
    )
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time",
    )
    pipeline_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathSubstitution(FindPackageShare("concrete_block_perception"))
            / "launch"
            / "rosbag_block_world_model.launch.py"
        ),
        launch_arguments={
            "bag": bag_path,
            "use_sim_time": use_sim_time,
            "perception_mode": "IDLE",
        }.items(),
    )

    set_mode_call = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    (
                        "ros2 service call /world_model_node/set_mode "
                        "concrete_block_perception/srv/SetPerceptionMode "
                        "\"{mode: SCENE_SCAN, target_block_id: '', enable_debug: true}\""
                    ),
                ],
                output="screen",
            )
        ],
    )

    get_coarse_call = TimerAction(
        period=9.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    (
                        "ros2 service call /world_model_node/get_coarse_blocks "
                        "concrete_block_perception/srv/GetCoarseBlocks "
                        '"{force_refresh: false, timeout_s: 0.0, '
                        'query_stamp: {sec: 0, nanosec: 0}}"'
                    ),
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            declare_bag,
            declare_use_sim_time,
            pipeline_launch,
            set_mode_call,
            get_coarse_call,
        ]
    )
