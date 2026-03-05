from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def _one_shot_pose_estimation_call(
    period_s: float,
    mode: str,
    target_block_id,
    enable_when: LaunchConfiguration,
) -> TimerAction:
    return TimerAction(
        period=period_s,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    (
                        "ros2 service call /world_model_node/run_pose_estimation "
                        "concrete_block_perception/srv/RunPoseEstimation "
                        f"\"{{mode: {mode}, target_block_id: '$CBP_TARGET_BLOCK_ID', "
                        'enable_debug: true, timeout_s: 8.0}"'
                    ),
                ],
                additional_env={
                    "CBP_TARGET_BLOCK_ID": target_block_id,
                },
                output="screen",
                condition=IfCondition(enable_when),
            )
        ],
    )


def _set_block_task_status_call(
    period_s: float,
    block_id,
    task_status: int,
    enable_when: LaunchConfiguration,
) -> TimerAction:
    return TimerAction(
        period=period_s,
        actions=[
            ExecuteProcess(
                cmd=[
                    "bash",
                    "-lc",
                    (
                        "ros2 service call /world_model_node/set_block_task_status "
                        "concrete_block_perception/srv/SetBlockTaskStatus "
                        "\"{block_id: '$CBP_TASK_BLOCK_ID', task_status: $CBP_TASK_STATUS}\""
                    ),
                ],
                additional_env={
                    "CBP_TASK_BLOCK_ID": block_id,
                    "CBP_TASK_STATUS": str(task_status),
                },
                output="screen",
                condition=IfCondition(enable_when),
            )
        ],
    )


def generate_launch_description():
    bag_path = LaunchConfiguration("bag")
    use_sim_time = LaunchConfiguration("use_sim_time")
    rviz = LaunchConfiguration("rviz")
    perception_mode = LaunchConfiguration("perception_mode")
    run_scene_discovery = LaunchConfiguration("run_scene_discovery")
    run_refine_block = LaunchConfiguration("run_refine_block")
    run_set_task_move = LaunchConfiguration("run_set_task_move")
    run_refine_grasped = LaunchConfiguration("run_refine_grasped")
    refine_block_id = LaunchConfiguration("refine_block_id")
    task_move_block_id = LaunchConfiguration("task_move_block_id")

    declare_bag = DeclareLaunchArgument(
        "bag",
        # default_value="/home/vscode/Documents/2025-12-17/pzs_crane_1_pickup",
        default_value="/home/vscode/Documents/2025-12-17/pzs_crane_2_placement_rotated",
        # default_value="/home/vscode/Documents/2025-12-17/pzs_crane_3_pickup_rotated",
        # default_value="/home/vscode/Documents/2025-12-17/pzs_crane_4_placement_far",
        # default_value="/home/vscode/Documents/2025-12-17/pzs_crane_5_stacking",
        # default_value="/home/vscode/Documents/2025-12-17/pzs_crane_6_stacking_rotated",
        # default_value="/home/vscode/Documents/2025-12-17/pzs_crane_7_palettes",
        # default_value="/home/vscode/Documents/2025-12-17/pzs_crane_8_palettes",
        description="Path to rosbag to replay",
    )
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Start RViz in rosbag replay helper launch",
    )
    declare_perception_mode = DeclareLaunchArgument(
        "perception_mode",
        default_value="IDLE",
        description="Startup world-model perception mode",
    )
    declare_run_scene_discovery = DeclareLaunchArgument(
        "run_scene_discovery",
        default_value="true",
        description="Trigger one-shot SCENE_DISCOVERY call",
    )
    declare_run_refine_block = DeclareLaunchArgument(
        "run_refine_block",
        default_value="false",
        description="Trigger one-shot REFINE_BLOCK call",
    )
    declare_run_set_task_move = DeclareLaunchArgument(
        "run_set_task_move",
        default_value="false",
        description="Set a world-model block to TASK_MOVE at 13s",
    )
    declare_run_refine_grasped = DeclareLaunchArgument(
        "run_refine_grasped",
        default_value="false",
        description="Trigger one-shot REFINE_GRASPED call",
    )
    declare_refine_block_id = DeclareLaunchArgument(
        "refine_block_id",
        default_value="wm_block_1",
        description="Block id used by REFINE_BLOCK one-shot call",
    )
    declare_task_move_block_id = DeclareLaunchArgument(
        "task_move_block_id",
        default_value="wm_block_1",
        description="Block id set to TASK_MOVE before REFINE_GRASPED",
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
            "rviz": rviz,
            "perception_mode": perception_mode,
        }.items(),
    )

    scene_discovery_call = _one_shot_pose_estimation_call(
        period_s=8.0,
        mode="SCENE_DISCOVERY",
        target_block_id="",
        enable_when=run_scene_discovery,
    )
    refine_block_call = _one_shot_pose_estimation_call(
        period_s=11.0,
        mode="REFINE_BLOCK",
        target_block_id=refine_block_id,
        enable_when=run_refine_block,
    )
    set_task_move_call = _set_block_task_status_call(
        period_s=16.0,
        block_id=task_move_block_id,
        task_status=2,  # Block.TASK_MOVE
        enable_when=run_set_task_move,
    )
    refine_grasped_call = _one_shot_pose_estimation_call(
        period_s=18.0,
        mode="REFINE_GRASPED",
        target_block_id="",
        enable_when=run_refine_grasped,
    )

    return LaunchDescription(
        [
            # Avoid Fast DDS XML parser noise when this env var is present but empty.
            UnsetEnvironmentVariable("FASTRTPS_DEFAULT_PROFILES_FILE"),
            declare_bag,
            declare_use_sim_time,
            declare_rviz,
            declare_perception_mode,
            declare_run_scene_discovery,
            declare_run_refine_block,
            declare_run_set_task_move,
            declare_run_refine_grasped,
            declare_refine_block_id,
            declare_task_move_block_id,
            pipeline_launch,
            scene_discovery_call,
            refine_block_call,
            set_task_move_call,
            refine_grasped_call,
        ]
    )
