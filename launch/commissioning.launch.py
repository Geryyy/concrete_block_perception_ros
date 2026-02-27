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
        description="Commissioning stage: segment | track | register | full",
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
            ),
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
                parameters=[block_registration_params],
                output="screen",
                emulate_tty=True,
                condition=needs_registration,
            ),
            Node(
                package="concrete_block_perception",
                executable="world_model_node",
                name="world_model_node",
                parameters=[
                    world_model_params,
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "calib_yaml": calib_yaml,
                        "pipeline_mode": LaunchConfiguration("stage"),
                    },
                ],
                output="screen",
                emulate_tty=True,
                remappings=[
                    ("image", "/zed2i/warped/left/image_rect_color/image_raw"),
                    ("points", "/seyond_points"),
                    ("tracked_detections", "/cbp/tracked_detections"),
                    ("block_world_model", "/cbp/block_world_model"),
                    ("block_world_model_markers", "/cbp/block_world_model_markers"),
                    ("debug/detection_overlay", "/cbp/debug/detection_overlay"),
                    ("debug/tracking_overlay", "/cbp/debug/tracking_overlay"),
                    ("debug/registration_cutout", "/cbp/debug/registration_cutout"),
                    ("debug/registration_template", "/cbp/debug/registration_template"),
                ],
            ),
        ]
    )
