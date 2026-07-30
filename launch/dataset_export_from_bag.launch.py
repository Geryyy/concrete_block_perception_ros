"""Play one rosbag and export a periodic, unlabeled dataset snapshot.

Starts the Cloudini decoder for the point cloud, the scene_dataset_exporter_node,
and (by default) `ros2 bag play` for the given bag. The launch shuts itself
down when bag playback finishes, so this is safe to loop over many bags.

    ros2 launch concrete_block_perception dataset_export_from_bag.launch.py \
        bag_path:=/home/vscode/Documents/2026-07-07-grip_at_top \
        output_dir:=dataset
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    bag_path = LaunchConfiguration("bag_path").perform(context)
    source_bag = LaunchConfiguration("source_bag").perform(context)
    if not source_bag:
        source_bag = os.path.basename(os.path.normpath(bag_path))

    cloudini_topic = LaunchConfiguration("cloudini_topic").perform(context)
    decoded_points_topic = LaunchConfiguration("decoded_points_topic").perform(context)

    cloudini_decoder = Node(
        package="cloudini_ros",
        executable="cloudini_topic_converter",
        name="dataset_export_cloudini_decoder",
        parameters=[{
            "compressing": False,
            "topic_input": cloudini_topic,
            "topic_output": decoded_points_topic,
            "use_sim_time": True,
        }],
        output="screen",
    )

    exporter = Node(
        package="concrete_block_perception",
        executable="scene_dataset_exporter_node",
        name="scene_dataset_exporter_node",
        parameters=[{
            "export_interval_s": LaunchConfiguration("export_interval_s"),
            "world_frame": LaunchConfiguration("world_frame"),
            "output_dir": LaunchConfiguration("output_dir"),
            "source_bag": source_bag,
            "use_sim_time": True,
        }],
        remappings=[
            ("image", LaunchConfiguration("image_topic")),
            ("camera_info", LaunchConfiguration("camera_info_topic")),
            ("points", decoded_points_topic),
        ],
        output="screen",
    )

    actions = [cloudini_decoder, exporter]

    if LaunchConfiguration("play_bag").perform(context) == "true":
        bag_play = ExecuteProcess(
            cmd=["ros2", "bag", "play", bag_path, "--clock", "--rate", LaunchConfiguration("rate")],
            output="screen",
        )
        # Exit the whole launch once the bag finishes, so this is loopable.
        shutdown_on_bag_exit = RegisterEventHandler(
            OnProcessExit(target_action=bag_play, on_exit=[EmitEvent(event=Shutdown())])
        )
        # /tf_static is normally published once. If bag playback starts before
        # the exporter's TransformListener has finished subscription matching,
        # that one-shot message is missed for good -- and if it lands right
        # before tf2 detects the playback clock jump, the resulting buffer
        # clear discards it a second way. Give discovery a moment first.
        startup_delay_s = LaunchConfiguration("startup_delay_s").perform(context)
        actions += [
            TimerAction(period=float(startup_delay_s), actions=[bag_play]),
            shutdown_on_bag_exit,
        ]

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("bag_path", description="Path to the rosbag directory to play."),
        DeclareLaunchArgument(
            "source_bag", default_value="",
            description="Recorded in metadata.yaml; defaults to bag_path's directory name."),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("play_bag", default_value="true"),
        DeclareLaunchArgument(
            "startup_delay_s", default_value="2.0",
            description="Delay before bag playback starts, so TF/topic subscriptions "
            "(especially the one-shot /tf_static) are matched before publishing begins."),
        DeclareLaunchArgument(
            "export_interval_s", default_value="1.0",
            description="Minimum gap, in the cloud's own message time, between exported samples."),
        DeclareLaunchArgument("output_dir", default_value="dataset"),
        DeclareLaunchArgument("world_frame", default_value="world"),
        DeclareLaunchArgument("image_topic", default_value="/blackfly_rotated/image_rect"),
        DeclareLaunchArgument("camera_info_topic", default_value="/blackfly_rotated/camera_info"),
        DeclareLaunchArgument("cloudini_topic", default_value="/seyond/points/cloudini"),
        DeclareLaunchArgument("decoded_points_topic", default_value="/seyond/points"),
        OpaqueFunction(function=_launch_setup),
    ])
