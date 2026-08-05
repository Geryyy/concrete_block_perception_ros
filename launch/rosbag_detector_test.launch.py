"""Local rosbag test bench for the concrete_block_detector.

Everything runs on *this* machine; the only external input is a rosbag
supplying /clock, TF, /joint_states, the Seyond cloud and the Blackfly image
(e.g. ``./play_bag.sh`` at the workspace root).

**Start the bag first, and keep it playing.** Every node here runs with
``use_sim_time``, so without /clock their time is frozen: RViz renders nothing
and looks hung, and scene discovery just burns its timeout because the detector
has no cloud younger than ``cached_cloud_max_age_s``. play_bag.sh plays once and
stops, so a long RViz session outlives it -- replay before each discovery.

Composed of:

  * ``remote_replay_client.launch.py`` with ``rviz:=false`` -- robot model
    (robot_state_publisher x3), the world -> elastic/world static TF, the crane
    IMU broadcasters and the gripper grasp detector.
  * ``world_node.launch.py`` -- the world model in IDLE, i.e. on-demand
    single-shot ``run_pose_estimation`` driven from the RViz panel.
  * ``concrete_block_detector.launch.py`` -- the detector under test.
  * the wall-plan server, so RViz's Plan Control panel is usable.
  * RViz with ``rviz/detector_test.rviz`` -- cbs.rviz plus a "Detector debug"
    group carrying the detector's own output. Neither shared config shows it:
    ``crane_imu.rviz`` has no detector display at all, and ``cbs.rviz`` points
    its one at ``/concrete_block_detector/markers``, which nothing publishes --
    the node creates that publisher with the *relative* name ``markers`` and
    runs in no namespace, so it resolves to plain ``/markers``.

Startup order matters. The world model must be discoverable before RViz's
Perception Control panel creates its service clients -- the panel builds them
once in onInitialize() and only ever recreates them if the service *name*
changes, so a client created too early stays unmatched for the whole session.
The world model therefore starts first and everything else follows a short
discovery delay.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Seconds between the world model coming up and RViz creating its panel
# service clients.  Same rationale as remote_replay_world_model.launch.py.
DISCOVERY_DELAY_S = 3.0


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    tool = LaunchConfiguration("tool")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    points_topic = LaunchConfiguration("points_topic")
    transport = LaunchConfiguration("transport")
    world_model_params = LaunchConfiguration("world_model_params")
    detector_params = LaunchConfiguration("detector_params")
    world_model_scene_discovery_params = LaunchConfiguration(
        "world_model_scene_discovery_params"
    )
    detector_scene_discovery_params = LaunchConfiguration(
        "detector_scene_discovery_params"
    )
    grasp_detector_config = LaunchConfiguration("grasp_detector_config")
    start_wall_plan_server = LaunchConfiguration("start_wall_plan_server")

    cbs_bt_share = PathSubstitution(FindPackageShare("concrete_block_behavior_tree"))
    assembly_planning_share = PathSubstitution(
        FindPackageShare("concrete_block_assembly_planning")
    )

    # ── World model: first, so the RViz panel finds it ──────────────────
    world_model_launch = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathSubstitution(FindPackageShare("concrete_block_world_model"))
                    / "launch"
                    / "world_node.launch.py"
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "perception_mode": "IDLE",
                    "params_file": world_model_params,
                    "scene_discovery_params": world_model_scene_discovery_params,
                }.items(),
            )
        ],
        scoped=True,
        forwarding=True,
    )

    wall_plan_server = Node(
        package="concrete_block_assembly_planning",
        executable="wall_plan_server",
        name="concrete_block_wall_plan_server",
        output="screen",
        parameters=[
            assembly_planning_share / "config" / "wall_plan_server.yaml",
            {
                "use_sim_time": use_sim_time,
                "world_model_service": "/world_model_node/get_coarse_blocks",
                "world_model_timeout_s": 2.0,
                "output_frame": "world",
                "wall_plans_file": assembly_planning_share
                / "config"
                / "wall_plans.yaml",
            },
        ],
        condition=IfCondition(start_wall_plan_server),
    )

    # ── Robot model + TF + IMU + grasp detector (no RViz; we start our own) ──
    # Scoped: IncludeLaunchDescription does not confine launch_arguments to the
    # included file, and remote_replay_client also takes an argument named
    # "rviz". Unscoped, its rviz:=false overwrites ours and our RViz below is
    # silently skipped.
    replay_client_launch = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathSubstitution(FindPackageShare("concrete_block_perception"))
                    / "launch"
                    / "remote_replay_client.launch.py"
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "rviz": "false",
                    "tool": tool,
                    "grasp_detector_config": grasp_detector_config,
                }.items(),
            )
        ],
        scoped=True,
        forwarding=True,
    )

    # ── The detector under test ─────────────────────────────────────────
    # Scoped for the same reason: params_file / scene_discovery_params are
    # generic names shared with world_node.launch.py.
    detector_launch = GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathSubstitution(FindPackageShare("concrete_block_detector"))
                    / "launch"
                    / "concrete_block_detector.launch.py"
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "params_file": detector_params,
                    "scene_discovery_params": detector_scene_discovery_params,
                    "points_topic": points_topic,
                    "transport": transport,
                }.items(),
            )
        ],
        scoped=True,
        forwarding=True,
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz_rosbag_detector_test",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        arguments=["-d", rviz_config],
        condition=IfCondition(rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use the /clock published by 'ros2 bag play --clock'.",
            ),
            DeclareLaunchArgument(
                "tool",
                default_value="pzs100_description",
                description="Package defining the tool; must match the bag.",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathSubstitution(
                    FindPackageShare("concrete_block_perception")
                )
                / "rviz"
                / "detector_test.rviz",
                description=(
                    "Detector test bench config: cbs.rviz plus the detector's own "
                    "output and pipeline-stage debug clouds. Pass cbs.rviz to fall "
                    "back to the assembly view."
                ),
            ),
            DeclareLaunchArgument("points_topic", default_value="/seyond/points"),
            DeclareLaunchArgument("transport", default_value="cloudini"),
            DeclareLaunchArgument(
                "world_model_params",
                default_value=PathSubstitution(
                    FindPackageShare("concrete_block_world_model")
                )
                / "config"
                / "world_model.yaml",
            ),
            DeclareLaunchArgument(
                "detector_params",
                default_value=PathSubstitution(
                    FindPackageShare("concrete_block_detector")
                )
                / "config"
                / "detector.yaml",
            ),
            DeclareLaunchArgument(
                "world_model_scene_discovery_params",
                default_value=PathSubstitution(
                    FindPackageShare("concrete_block_perception")
                )
                / "config"
                / "grip_at_top_world_model_scene_discovery.yaml",
            ),
            DeclareLaunchArgument(
                "detector_scene_discovery_params",
                default_value=PathSubstitution(
                    FindPackageShare("concrete_block_perception")
                )
                / "config"
                / "grip_at_top_detector_scene_discovery.yaml",
            ),
            DeclareLaunchArgument(
                "grasp_detector_config",
                default_value=cbs_bt_share
                / "config"
                / "gripper_grasp_detector_real.yaml",
            ),
            DeclareLaunchArgument(
                "start_wall_plan_server",
                default_value="true",
                description="Needed by RViz's Plan Control panel.",
            ),
            # cbs.rviz embeds the BehaviorTree panel; point it at the CBS trees
            # so it is usable rather than erroring on an unset package.
            SetEnvironmentVariable(
                name="BEHAVIOR_TREE_PANEL_BT_PACKAGE",
                value="concrete_block_behavior_tree",
            ),
            SetEnvironmentVariable(
                name="BEHAVIOR_TREE_PANEL_BT_CATALOG",
                value=cbs_bt_share / "config" / "bt_panel_catalog.yaml",
            ),
            world_model_launch,
            wall_plan_server,
            TimerAction(
                period=DISCOVERY_DELAY_S,
                actions=[replay_client_launch, detector_launch, rviz_node],
            ),
        ]
    )
