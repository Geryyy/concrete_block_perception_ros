"""Local client bringup for a distributed rosbag-replay session.

This is a trimmed copy of concrete_block_perception's
``rosbag_block_world_model.launch.py`` (the known-good pipeline that shows the
robot model in RViz). The heavy half of that pipeline runs on the *remote* peer
over Cyclone DDS (see cyclonedds.xml P2P config), so the parts that live there
are removed here:

  * perception.launch.py        -> remote
  * world_node.launch.py        -> remote
  * ros2 bag play --clock       -> remote

What stays on *this* machine is exactly the rosbag-replay helper nodes:

  * robot model publishing (robot_state_publisher x3: rigid / elastic / full)
  * the world -> elastic/world static TF
  * the crane IMU sensor/TF broadcasters
  * RViz (crane_imu.rviz)

Everything published here is shared with the remote peer over Cyclone DDS, and
the joint_states / /clock arrive from the remote 'ros2 bag play --clock'.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    rviz = LaunchConfiguration("rviz")
    tool = LaunchConfiguration("tool")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use the /clock published by the remote 'ros2 bag play --clock'",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Start a local RViz (crane_imu.rviz)",
    )
    declare_tool = DeclareLaunchArgument(
        "tool",
        default_value="epsilon_7040_description",
        description="Name of the package where the tool is defined",
    )

    # -----------------------
    # Rosbag replay helper nodes (robot model + TF + RViz)
    # -----------------------
    rosbag_nodes_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathSubstitution(FindPackageShare("epsilon_crane_bringup_hmi"))
            / "launch"
            / "rosbag_replay_nodes.launch.py"
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "rviz": rviz,
            "tool": tool,
        }.items(),
    )

    return LaunchDescription(
        [
            declare_use_sim_time,
            declare_rviz,
            declare_tool,
            rosbag_nodes_launch,
        ]
    )
