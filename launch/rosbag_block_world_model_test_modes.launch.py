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

    scene_discovery_call = TimerAction(
        period=8.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    (
                        "ros2 service call /world_model_node/run_pose_estimation "
                        "concrete_block_perception/srv/RunPoseEstimation "
                        "\"{mode: SCENE_DISCOVERY, target_block_id: '', "
                        'enable_debug: true, timeout_s: 8.0}"'
                    ),
                ],
                output="screen",
            )
        ],
    )

    refine_block_call = TimerAction(
        period=12.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    (
                        "ros2 service call /world_model_node/run_pose_estimation "
                        "concrete_block_perception/srv/RunPoseEstimation "
                        "\"{mode: REFINE_BLOCK, target_block_id: 'block_1', "
                        'enable_debug: true, timeout_s: 8.0}"'
                    ),
                ],
                output="screen",
            )
        ],
    )

    refine_grasped_call = TimerAction(
        period=18.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    (
                        "ros2 service call /world_model_node/run_pose_estimation "
                        "concrete_block_perception/srv/RunPoseEstimation "
                        "\"{mode: REFINE_GRASPED, target_block_id: '', "
                        'enable_debug: true, timeout_s: 8.0}"'
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
            scene_discovery_call,
            # refine_block_call,
            # refine_grasped_call,
        ]
    )
