from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

from launch import LaunchDescription
import glob
import os


def cuda_library_path():
    paths = glob.glob("/usr/local/lib/python3.10/dist-packages/nvidia/*/lib")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        paths.append(existing)
    return ":".join(paths)


def generate_launch_description():
    pkg_dir = FindPackageShare("concrete_block_perception")
    world_model_pkg_dir = FindPackageShare("concrete_block_world_model")

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
        description="Deprecated; ignored. Processing is triggered by run_pose_estimation.",
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
        description="Optional overlay params file for world_model_node startup seeding",
    )
    start_processing_stack_arg = DeclareLaunchArgument(
        "start_processing_stack",
        default_value="true",
        description="Whether to start segmentation/tracking/registration nodes",
    )

    start_world_model_arg = DeclareLaunchArgument(
        "start_world_model",
        default_value="true",
        description="Whether to start world_model_node",
    )
    calib_yaml_arg = DeclareLaunchArgument(
        "calib_yaml",
        default_value="calib_blackfly_to_seyond.yaml",
        description="Calibration YAML in concrete_block_perception/config.",
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

    world_model_params = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_world_model"),
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
            perception_mode_arg,
            world_model_overlay_arg,
            start_world_model_arg,
            start_processing_stack_arg,
            calib_yaml_arg,
            SetEnvironmentVariable("LD_LIBRARY_PATH", cuda_library_path()),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("ros2_yolos_cpp"))
                / "launch"
                / "segmentor_service.launch.py",
                launch_arguments={
                    "model_path": LaunchConfiguration("model_path"),
                    "labels_path": LaunchConfiguration("labels_path"),
                    "use_gpu": LaunchConfiguration("use_gpu"),
                }.items(),
                condition=IfCondition(LaunchConfiguration("start_processing_stack")),
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
                condition=IfCondition(LaunchConfiguration("start_processing_stack")),
            ),
            Node(
                package="concrete_block_perception",
                executable="block_registration_node",
                name="registration_node",
                parameters=[
                    block_registration_params,
                    {"calib_yaml": LaunchConfiguration("calib_yaml")},
                ],
                remappings=[
                    (
                        "debug/registration/mask_cutout",
                        "/cbp/debug/registration/mask_cutout",
                    ),
                    (
                        "debug/registration/cleaned_cutout",
                        "/cbp/debug/registration/cleaned_cutout",
                    ),
                    (
                        "debug/registration/plane_cloud",
                        "/cbp/debug/registration/plane_cloud",
                    ),
                    ("debug/registration/template", "/cbp/debug/registration/template"),
                    ("debug/registration/mask", "/cbp/debug/registration/mask"),
                    (
                        "debug/registration/diagnostics",
                        "/cbp/debug/registration/diagnostics",
                    ),
                    (
                        "debug/registration/gripper_boxes",
                        "/cbp/debug/registration/gripper_boxes",
                    ),
                ],
                output="screen",
                emulate_tty=True,
                condition=IfCondition(LaunchConfiguration("start_processing_stack")),
            ),
            # Node(
            #     package="cloudini_ros",
            #     executable="cloudini_topic_converter",
            #     name="seyond_points_cloudini_decoder",
            #     parameters=[
            #         {
            #             "compressing": False,
            #             "topic_input": "/seyond/points/cloudini",
            #             "topic_output": "/seyond/points",
            #         }
            #     ],
            #     output="screen",
            #     emulate_tty=True,
            #     condition=IfCondition(LaunchConfiguration("start_world_model")),
            # ),
            Node(
                package="concrete_block_world_model",
                executable="world_model_node",
                name="world_model_node",
                condition=IfCondition(LaunchConfiguration("start_world_model")),
                parameters=[
                    world_model_params,
                    LaunchConfiguration("world_model_overlay_params_file"),
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "perception_mode": LaunchConfiguration("perception_mode"),
                    },
                ],
                output="screen",
                emulate_tty=True,
                remappings=[
                    # =========================
                    # Inputs
                    # =========================
                    # Image input (synced with cloud)
                    ("image", "/blackfly_rotated/image_rect"),
                    # Point cloud input (10 Hz)
                    ("points", "/seyond/points"),
                    # =========================
                    # Outputs
                    # =========================
                    ("block_world_model", "/cbp/block_world_model"),
                    ("block_world_model_markers", "/cbp/block_world_model_markers"),
                    ("block_goal_markers", "/cbp/block_goal_markers"),
                    # =========================
                    # Debug topics
                    # =========================
                    ("debug/detection_overlay", "/cbp/debug/detection_overlay"),
                    (
                        "debug/yolo_service_debug_image",
                        "/cbp/debug/yolo_service_debug_image",
                    ),
                    (
                        "debug/continuous_merged_mask",
                        "/cbp/debug/continuous_merged_mask",
                    ),
                    ("debug/tracking_overlay", "/cbp/debug/tracking_overlay"),
                    (
                        "debug/refine_grasped_roi_input",
                        "/cbp/debug/refine_grasped_roi_input",
                    ),
                    ("timing/continuous_seg_ms", "/cbp/timing/continuous_seg_ms"),
                    ("timing/continuous_cutout_ms", "/cbp/timing/continuous_cutout_ms"),
                    ("timing/continuous_coarse_ms", "/cbp/timing/continuous_coarse_ms"),
                    (
                        "timing/continuous_registration_ms",
                        "/cbp/timing/continuous_registration_ms",
                    ),
                    ("timing/continuous_upsert_ms", "/cbp/timing/continuous_upsert_ms"),
                    ("timing/continuous_total_ms", "/cbp/timing/continuous_total_ms"),
                    (
                        "timing/continuous_detections",
                        "/cbp/timing/continuous_detections",
                    ),
                    ("timing/continuous_accepted", "/cbp/timing/continuous_accepted"),
                    ("timing/continuous_rejected", "/cbp/timing/continuous_rejected"),
                ],
            ),
        ]
    )
