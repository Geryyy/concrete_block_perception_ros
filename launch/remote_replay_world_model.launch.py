"""Remote replay client with a local world model.

The world model must be discoverable before RViz's Perception Control panel
creates its service clients.  Start it first, then give DDS a short discovery
window before starting the existing remote replay client (robot model, TF,
IMU broadcasters, gripper detector, and RViz).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    rviz = LaunchConfiguration("rviz")
    tool = LaunchConfiguration("tool")
    points_topic = LaunchConfiguration("points_topic")
    world_model_params = LaunchConfiguration("world_model_params")
    detector_params = LaunchConfiguration("detector_params")
    grasp_detector_config = LaunchConfiguration("grasp_detector_config")

    world_model_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathSubstitution(FindPackageShare("concrete_block_world_model"))
            / "launch"
            / "world_node.launch.py"
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "perception_mode": "IDLE",
            "params_file": world_model_params,
        }.items(),
    )

    detector_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathSubstitution(FindPackageShare("concrete_block_detector"))
            / "launch"
            / "concrete_block_detector.launch.py"
        ),
        launch_arguments={
            "params_file": detector_params,
            "points_topic": points_topic,
            "transport": "cloudini",
            "use_sim_time": use_sim_time,
        }.items(),
    )

    remote_client_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathSubstitution(FindPackageShare("concrete_block_perception"))
            / "launch"
            / "remote_replay_client.launch.py"
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "rviz": rviz,
            "tool": tool,
            "grasp_detector_config": grasp_detector_config,
        }.items(),
    )

    default_world_model_params = (
        PathSubstitution(FindPackageShare("concrete_block_world_model"))
        / "config"
        / "world_model.yaml"
    )
    default_grasp_detector_config = (
        PathSubstitution(FindPackageShare("concrete_block_behavior_tree"))
        / "config"
        / "gripper_grasp_detector_real.yaml"
    )
    default_detector_params = (
        PathSubstitution(FindPackageShare("concrete_block_detector"))
        / "config"
        / "detector.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument(
                "tool", default_value="epsilon_7040_description"),
            DeclareLaunchArgument("points_topic", default_value="/seyond/points"),
            DeclareLaunchArgument(
                "world_model_params", default_value=default_world_model_params),
            DeclareLaunchArgument("detector_params", default_value=default_detector_params),
            DeclareLaunchArgument(
                "grasp_detector_config", default_value=default_grasp_detector_config),
            world_model_launch,
            TimerAction(period=3.0, actions=[remote_client_launch, detector_launch]),
        ]
    )
