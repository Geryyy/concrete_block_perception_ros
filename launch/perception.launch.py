from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathSubstitution,
    PathJoinSubstitution,
)
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_dir = FindPackageShare("concrete_block_perception")

    # ---------------------------------------------------------
    # Launch arguments
    # ---------------------------------------------------------
    model_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution(
            [pkg_dir, "config", "yolo26n_seg_best.onnx"]
        ),
    )

    labels_arg = DeclareLaunchArgument(
        "labels_path",
        default_value=PathJoinSubstitution([pkg_dir, "config", "block.names"]),
    )

    gpu_arg = DeclareLaunchArgument(
        "use_gpu",
        default_value="false",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
    )

    # ---------------------------------------------------------
    # Parameter files
    # ---------------------------------------------------------
    block_detection_tracking_params = PathJoinSubstitution(
        [pkg_dir, "config", "block_detection_tracking.yaml"]
    )

    block_registration_params = PathJoinSubstitution(
        [pkg_dir, "config", "block_registration.yaml"]
    )

    world_model_params = PathJoinSubstitution([pkg_dir, "config", "world_model.yaml"])

    calib_yaml = PathJoinSubstitution([pkg_dir, "config", "calib_zed2i_to_seyond.yaml"])

    # ---------------------------------------------------------
    # Unified Perception Container
    # ---------------------------------------------------------
    perception_container = ComposableNodeContainer(
        name="perception_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        emulate_tty=True,
        composable_node_descriptions=[
            # -------------------------
            # YOLO Segmentor
            # -------------------------
            ComposableNode(
                package="ros2_yolos_cpp",
                plugin="ros2_yolos_cpp::YolosSegmentorServiceNode",
                name="yolos_segmentor_service",
                parameters=[
                    {
                        "model_path": LaunchConfiguration("model_path"),
                        "labels_path": LaunchConfiguration("labels_path"),
                        "use_gpu": LaunchConfiguration("use_gpu"),
                    }
                ],
                # extra_arguments=[{"use_intra_process_comms": True}],
            ),
            # -------------------------
            # Detection Tracking
            # -------------------------
            ComposableNode(
                package="concrete_block_perception",
                plugin="concrete_block_perception::BlockDetectionTrackingNode",
                name="block_detection_tracking_node",
                parameters=[block_detection_tracking_params],
                # extra_arguments=[{"use_intra_process_comms": True}],
            ),
            # -------------------------
            # Registration
            # -------------------------
            ComposableNode(
                package="concrete_block_perception",
                plugin="concrete_block_perception::BlockRegistrationNode",
                name="block_registration_node",
                parameters=[block_registration_params],
                # extra_arguments=[{"use_intra_process_comms": True}],
            ),
            # -------------------------
            # World Model
            # -------------------------
            ComposableNode(
                package="concrete_block_perception",
                plugin="concrete_block_perception::WorldModelNode",
                name="world_model_node",
                parameters=[
                    world_model_params,
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "calib_yaml": calib_yaml,
                    },
                ],
                remappings=[
                    ("image", "/zed2i/warped/left/image_rect_color/image_raw"),
                    ("points", "/seyond_points"),
                    ("block_world_model", "/cbp/block_world_model"),
                    ("block_world_model_markers", "/cbp/block_world_model_markers"),
                    ("debug/detection_overlay", "/cbp/debug/detection_overlay"),
                    ("debug/tracking_overlay", "/cbp/debug/tracking_overlay"),
                    ("debug/registration_cutout", "/cbp/debug/registration_cutout"),
                    ("debug/registration_template", "/cbp/debug/registration_template"),
                ],
                # extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
    )

    # ---------------------------------------------------------
    # Infrastructure Nodes (standalone)
    # ---------------------------------------------------------
    cloud_converter = Node(
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
    )

    image_republish = Node(
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
    )

    # ---------------------------------------------------------
    # Launch description
    # ---------------------------------------------------------
    return LaunchDescription(
        [
            model_arg,
            labels_arg,
            gpu_arg,
            use_sim_time_arg,
            cloud_converter,
            image_republish,
            perception_container,
        ]
    )
