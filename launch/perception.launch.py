import os
import subprocess
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from ament_index_python import get_package_share_directory
from launch.substitutions import LaunchConfiguration, PathSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

from launch import LaunchDescription


def generate_launch_description():
    pkg_dir = FindPackageShare("concrete_block_perception")

    lifecycle_manager_yaml = os.path.join(
        get_package_share_directory("concrete_block_perception"),
        "config",
        "lifecycle_manager.yaml",
    )

    yolo_model_path = PathSubstitution(pkg_dir) / "config/yolo26n_seg_best.onnx"

    yolo_labels_path = PathSubstitution(pkg_dir) / "config/block.names"

    return LaunchDescription(
        [
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
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("foxglove_bridge"))
                / "launch"
                / "foxglove_bridge_launch.xml",
                launch_arguments={
                    "port": "8765",
                    "best_effort_qos_topic_whitelist": (
                        "^/(seyond_points.*|zed2i/.*/image.*|"
                        "yolos_segmentor/(debug_image|mask).*)"
                    ),
                }.items(),
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager",
                output="screen",
                parameters=[lifecycle_manager_yaml],
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
            # Node(
            #     package="concrete_block_perception",
            #     executable="detection_node.py",
            #     parameters=[
            #         {
            #             "model_path": PathJoinSubstitution(
            #                 [
            #                     FindPackageShare("concrete_block_perception"),
            #                     "config",
            #                     "yolo_model.pt",
            #                 ]
            #             ),
            #             "image_topic": "/zed2i/warped/left/image_rect_color/image_raw",
            #             "confidence": 0.5,
            #             "imgsz": 1280,
            #             "device": "0",
            #             "show_debug_window": True,
            #         }
            #     ],
            #     output="screen",
            # ),
            # Node(
            #     package="concrete_block_perception",
            #     executable="segmentation_node.py",
            #     parameters=[
            #         {
            #             "model_path": PathJoinSubstitution(
            #                 [
            #                     FindPackageShare("concrete_block_perception"),
            #                     "config",
            #                     "FastSAM-x.pt",
            #                 ]
            #             ),
            #             "device": "cpu",  # or "cpu"
            #             "imgsz": 1280,
            #             "conf": 0.1,
            #             "iou": 0.9,
            #             "enable_debug": True,  # overlay image in response
            #             "select_smallest_mask": True,
            #         }
            #     ],
            #     output="screen",
            # ),
            IncludeLaunchDescription(
                PathSubstitution(FindPackageShare("ros2_yolos_cpp"))
                / "launch"
                / "segmentor.launch.py",
                launch_arguments={
                    "model_path": yolo_model_path,
                    "labels_path": yolo_labels_path,
                    "use_gpu": "false",
                    "image_topic": "/zed2i/warped/left/image_rect_color/image_raw",
                    "camera_info_topic": "/zed2i/warped/left/camera_info",
                }.items(),
            ),
        ]
    )
