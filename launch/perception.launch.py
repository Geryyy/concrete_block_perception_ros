import os
import subprocess
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from ament_index_python import get_package_share_directory
from launch.substitutions import LaunchConfiguration, PathSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

from launch import LaunchDescription


def generate_launch_description():
    pkg_dir = FindPackageShare("concrete_block_perception")

    # Declare launch arguments
    model_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution(
            [pkg_dir, "config", "yolo26n_seg_best.onnx"]
        ),
        description="Path to YOLO segmentation model",
    )

    labels_arg = DeclareLaunchArgument(
        "labels_path",
        default_value=PathJoinSubstitution([pkg_dir, "config", "block.names"]),
        description="Path to class labels file",
    )

    gpu_arg = DeclareLaunchArgument(
        "use_gpu",
        default_value="false",
        description="Whether to use GPU for inference",
    )

    sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock",
    )

    mode_arg = DeclareLaunchArgument(
        "pipeline_mode",
        default_value="full",
        description="segment | track | register | full",
    )

    start_world_model_arg = DeclareLaunchArgument(
        "start_world_model",
        default_value="true",
        description="Whether to start world_model_node",
    )

    registration_backend_arg = DeclareLaunchArgument(
        "registration_backend",
        default_value="legacy",
        description="Registration backend: legacy | teaser",
    )

    block_detection_tracking_params = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_perception"),
            "config",
            "block_detection_tracking.yaml",
        ]
    )

    block_registration_params = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_perception"),
            "config",
            "block_registration.yaml",
        ]
    )

    calib_yaml = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_perception"),
            "config",
            "calib_zed2i_to_seyond.yaml",
        ]
    )

    world_model_params = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_perception"),
            "config",
            "world_model.yaml",
        ]
    )

    return LaunchDescription(
        [
            model_arg,
            labels_arg,
            gpu_arg,
            sim_time_arg,
            mode_arg,
            start_world_model_arg,
            registration_backend_arg,
            Node(
                package="cloudini_ros",
                executable="cloudini_topic_converter",
                name="cloudini_topic_converter",
                parameters=[
                    {
                        "compressing": False,
                        "topic_input": "/seyond_points/compressed",
                        "topic_output": "/seyond_points",
                    }
                ],
                arguments=["--ros-args", "--log-level", "WARN"],
            ),
            # IncludeLaunchDescription(
            #     PathSubstitution(FindPackageShare("foxglove_bridge"))
            #     / "launch"
            #     / "foxglove_bridge_launch.xml",
            #     launch_arguments={
            #         "port": "8765",
            #         "log_level": "warn",
            #         "best_effort_qos_topic_whitelist": (
            #             "^/(seyond_points.*|zed2i/.*/image.*|"
            #             "yolos_segmentor/(debug_image|mask).*)"
            #         ),
            #     }.items(),
            # ),
            Node(
                package="image_transport",
                executable="republish",
                arguments=[
                    "compressed",
                    "raw",
                    "--ros-args",
                    "--remap",
                    "in/compressed:=/zed2i/warped/left/image_rect_color/compressed",
                    "--remap",
                    "out:=/zed2i/warped/left/image_rect_color/image_raw",
                ],
                output="screen",
            ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("ros2_yolos_cpp"))
                / "launch"
                / "segmentor_service.launch.py",
                launch_arguments={
                    "model_path": LaunchConfiguration("model_path"),
                    "labels_path": LaunchConfiguration("labels_path"),
                    "use_gpu": LaunchConfiguration("use_gpu"),
                }.items(),
            ),
            Node(
                package="concrete_block_perception",
                executable="block_detection_tracking_node",
                name="block_detection_tracking_node",
                parameters=[
                    block_detection_tracking_params,
                ],
                output="screen",
                emulate_tty=True,
            ),
            Node(
                package="concrete_block_perception",
                executable="block_registration_node",
                name="registration_node",
                parameters=[block_registration_params],
                remappings=[
                    ("debug/cutout_cloud", "/cbp/debug/registration_cutout"),
                    ("debug/template_cloud", "/cbp/debug/registration_template"),
                    ("debug/segmentation_mask", "/cbp/debug/registration_mask"),
                ],
                output="screen",
                emulate_tty=True,
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'", LaunchConfiguration("registration_backend"), "' == 'legacy'"
                        ]
                    )
                ),
            ),
            Node(
                package="concrete_block_registration_teaser",
                executable="block_registration_teaser_node",
                name="registration_node",
                parameters=[block_registration_params],
                remappings=[
                    ("debug/cutout_cloud", "/cbp/debug/registration_cutout"),
                    ("debug/template_cloud", "/cbp/debug/registration_template"),
                    ("debug/segmentation_mask", "/cbp/debug/registration_mask"),
                ],
                output="screen",
                emulate_tty=True,
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'", LaunchConfiguration("registration_backend"), "' == 'teaser'"
                        ]
                    )
                ),
            ),
            Node(
                package="concrete_block_perception",
                executable="world_model_node",
                name="world_model_node",
                condition=IfCondition(LaunchConfiguration("start_world_model")),
                parameters=[
                    world_model_params,
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "calib_yaml": calib_yaml,
                        "pipeline_mode": LaunchConfiguration("pipeline_mode"),
                    },
                ],
                output="screen",
                emulate_tty=True,
                remappings=[
                    # =========================
                    # Inputs
                    # =========================
                    # Image input (synced with cloud)
                    ("image", "/zed2i/warped/left/image_rect_color/image_raw"),
                    # Point cloud input (10 Hz)
                    ("points", "/seyond_points"),
                    ("tracked_detections", "/cbp/tracked_detections"),
                    # =========================
                    # Outputs
                    # =========================
                    ("block_world_model", "/cbp/block_world_model"),
                    ("block_world_model_markers", "/cbp/block_world_model_markers"),
                    # =========================
                    # Debug topics
                    # =========================
                    ("debug/detection_overlay", "/cbp/debug/detection_overlay"),
                    ("debug/tracking_overlay", "/cbp/debug/tracking_overlay"),
                    ("debug/registration_cutout", "/cbp/debug/registration_cutout"),
                    ("debug/registration_template", "/cbp/debug/registration_template"),
                    ("debug/refine_grasped_roi_input", "/cbp/debug/refine_grasped_roi_input"),
                ],
            ),
        ]
    )
