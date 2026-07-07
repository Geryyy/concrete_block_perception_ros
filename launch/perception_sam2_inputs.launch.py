from launch import LaunchDescription
import glob
import os

from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def cuda_library_path():
    paths = glob.glob("/usr/local/lib/python3.10/dist-packages/nvidia/*/lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        paths.append(existing)
    return ":".join(paths)


def generate_launch_description():
    pkg_dir = FindPackageShare("concrete_block_perception")
    world_model_pkg_dir = FindPackageShare("concrete_block_world_model")

    image_topic = "/blackfly_rotated/image_rect"

    model_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution([pkg_dir, "config", "yolo26n_seg_best.onnx"]),
        description="Path to YOLO segmentation model.",
    )
    labels_arg = DeclareLaunchArgument(
        "labels_path",
        default_value=PathJoinSubstitution([pkg_dir, "config", "block.names"]),
        description="Path to YOLO class labels.",
    )
    use_gpu_arg = DeclareLaunchArgument(
        "use_gpu",
        default_value="false",
        description="Use GPU for YOLO inference.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock.",
    )
    perception_mode_arg = DeclareLaunchArgument(
        "perception_mode",
        default_value="IDLE",
        description="World-model perception mode: IDLE or CONTINUOUS.",
    )
    world_model_overlay_arg = DeclareLaunchArgument(
        "world_model_overlay_params_file",
        default_value=PathJoinSubstitution(
            [world_model_pkg_dir, "config", "world_model_seed_none.yaml"]
        ),
        description="Optional overlay params file for world_model_node startup seeding.",
    )
    yolo_conf_arg = DeclareLaunchArgument(
        "yolo_conf_threshold",
        default_value="0.4",
        description="YOLO confidence threshold for the topic segmentor feeding SAM2.",
    )
    yolo_timing_arg = DeclareLaunchArgument(
        "yolo_timing_log_every_n_frames",
        default_value="10",
        description="Log topic YOLO timing every N frames.",
    )

    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dir, "launch", "perception.launch.py"])
        ),
        launch_arguments={
            "model_path": LaunchConfiguration("model_path"),
            "labels_path": LaunchConfiguration("labels_path"),
            "use_gpu": LaunchConfiguration("use_gpu"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "perception_mode": LaunchConfiguration("perception_mode"),
            "world_model_overlay_params_file": LaunchConfiguration("world_model_overlay_params_file"),
        }.items(),
    )

    yolo_segmentor_topics = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros2_yolos_cpp"), "launch", "segmentor.launch.py"]
            )
        ),
        launch_arguments={
            "model_path": LaunchConfiguration("model_path"),
            "labels_path": LaunchConfiguration("labels_path"),
            "use_gpu": LaunchConfiguration("use_gpu"),
            "conf_threshold": LaunchConfiguration("yolo_conf_threshold"),
            "image_topic": image_topic,
            "timing_log_every_n_frames": LaunchConfiguration("yolo_timing_log_every_n_frames"),
        }.items(),
    )

    configure_yolo_topics = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "lifecycle", "set", "/yolos_segmentor", "configure"],
                output="screen",
            )
        ],
    )
    activate_yolo_topics = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=["ros2", "lifecycle", "set", "/yolos_segmentor", "activate"],
                output="screen",
            )
        ],
    )

    return LaunchDescription(
        [
            model_arg,
            labels_arg,
            use_gpu_arg,
            use_sim_time_arg,
            perception_mode_arg,
            world_model_overlay_arg,
            yolo_conf_arg,
            yolo_timing_arg,
            SetEnvironmentVariable("LD_LIBRARY_PATH", cuda_library_path()),
            perception,
            yolo_segmentor_topics,
            configure_yolo_topics,
            activate_yolo_topics,
        ]
    )
