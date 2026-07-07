from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    pkg_dir = FindPackageShare("concrete_block_perception")

    stage_arg = DeclareLaunchArgument(
        "stage",
        default_value="segment",
        description="Commissioning nodes to start: segment | track | register | full",
    )

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

    needs_track = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("stage"),
                "' in ['track', 'register', 'full']",
            ]
        )
    )

    needs_registration = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("stage"),
                "' in ['register', 'full']",
            ]
        )
    )

    return LaunchDescription(
        [
            stage_arg,
            model_arg,
            labels_arg,
            gpu_arg,
            sim_time_arg,
            calib_yaml_arg,
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
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
                output="screen",
                emulate_tty=True,
                condition=needs_track,
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
                    ("debug/registration/mask_cutout", "/cbp/debug/registration/mask_cutout"),
                    ("debug/registration/cleaned_cutout", "/cbp/debug/registration/cleaned_cutout"),
                    ("debug/registration/plane_cloud", "/cbp/debug/registration/plane_cloud"),
                    ("debug/registration/template", "/cbp/debug/registration/template"),
                    ("debug/registration/mask", "/cbp/debug/registration/mask"),
                    ("debug/registration/diagnostics", "/cbp/debug/registration/diagnostics"),
                    ("debug/registration/gripper_boxes", "/cbp/debug/registration/gripper_boxes"),
                ],
                output="screen",
                emulate_tty=True,
                condition=needs_registration,
            ),
            Node(
                package="concrete_block_world_model",
                executable="world_model_node",
                name="world_model_node",
                parameters=[
                    world_model_params,
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    },
                ],
                output="screen",
                emulate_tty=True,
                remappings=[
                    ("image", "/blackfly_rotated/image_rect"),
                    ("points", "/seyond/points"),
                    ("block_world_model", "/cbp/block_world_model"),
                    ("block_world_model_markers", "/cbp/block_world_model_markers"),
                    ("debug/detection_overlay", "/cbp/debug/detection_overlay"),
                    ("debug/continuous_merged_mask", "/cbp/debug/continuous_merged_mask"),
                    ("debug/tracking_overlay", "/cbp/debug/tracking_overlay"),
                    ("timing/continuous_seg_ms", "/cbp/timing/continuous_seg_ms"),
                    ("timing/continuous_cutout_ms", "/cbp/timing/continuous_cutout_ms"),
                    ("timing/continuous_coarse_ms", "/cbp/timing/continuous_coarse_ms"),
                    ("timing/continuous_registration_ms", "/cbp/timing/continuous_registration_ms"),
                    ("timing/continuous_upsert_ms", "/cbp/timing/continuous_upsert_ms"),
                    ("timing/continuous_total_ms", "/cbp/timing/continuous_total_ms"),
                    ("timing/continuous_detections", "/cbp/timing/continuous_detections"),
                    ("timing/continuous_accepted", "/cbp/timing/continuous_accepted"),
                    ("timing/continuous_rejected", "/cbp/timing/continuous_rejected"),
                ],
            ),
        ]
    )
