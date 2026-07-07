from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/blackfly_rotated/image_rect",
        description="Raw RGB image topic consumed by SAM2.",
    )
    detections_topic_arg = DeclareLaunchArgument(
        "detections_topic",
        default_value="/yolos_segmentor/detections",
        description="YOLO Detection2DArray topic used as SAM2 box prompts.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock.",
    )
    sam2_device_arg = DeclareLaunchArgument(
        "sam2_device",
        default_value="cuda",
        description="SAM2 torch device.",
    )
    sam2_checkpoint_arg = DeclareLaunchArgument(
        "sam2_checkpoint",
        default_value="",
        description="SAM2 checkpoint path. Empty uses SAM2_CHECKPOINT_DIR/default.",
    )
    log_every_arg = DeclareLaunchArgument(
        "log_every_n_frames",
        default_value="1",
        description="SAM2 sync/progress log interval.",
    )

    sam2_node = Node(
        package="sam2_segmentation",
        executable="sam2_segmentation_node",
        name="sam2_segmentation_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("sam2_segmentation"), "config", "sam2_segmentation.yaml"]
            ),
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "device": LaunchConfiguration("sam2_device"),
                "checkpoint": LaunchConfiguration("sam2_checkpoint"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "log_every_n_frames": LaunchConfiguration("log_every_n_frames"),
            },
        ],
    )

    return LaunchDescription(
        [
            image_topic_arg,
            detections_topic_arg,
            use_sim_time_arg,
            sam2_device_arg,
            sam2_checkpoint_arg,
            log_every_arg,
            sam2_node,
        ]
    )
