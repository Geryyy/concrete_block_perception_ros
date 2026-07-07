from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/blackfly_rotated/image_rect",
        description="Raw RGB image topic consumed by SAM2 video.",
    )
    detections_topic_arg = DeclareLaunchArgument(
        "detections_topic",
        default_value="/yolos_segmentor/detections",
        description="YOLO Detection2DArray topic used as SAM2 video box prompts.",
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
    clip_length_arg = DeclareLaunchArgument(
        "clip_length",
        default_value="4",
        description="Number of synced frames per video-propagation clip.",
    )
    clip_stride_arg = DeclareLaunchArgument(
        "clip_stride",
        default_value="1",
        description="Minimum number of new frames before starting another SAM2 video window.",
    )
    offload_video_arg = DeclareLaunchArgument(
        "offload_video_to_cpu",
        default_value="false",
        description="Offload SAM2 video frames to CPU to save GPU memory at lower speed.",
    )

    sam2_node = Node(
        package="sam2_segmentation",
        executable="sam2_video_segmentation_node",
        name="sam2_video_segmentation_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("sam2_segmentation"), "config", "sam2_video_segmentation.yaml"]
            ),
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "detections_topic": LaunchConfiguration("detections_topic"),
                "device": LaunchConfiguration("sam2_device"),
                "checkpoint": LaunchConfiguration("sam2_checkpoint"),
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "clip_length": LaunchConfiguration("clip_length"),
                "clip_stride": LaunchConfiguration("clip_stride"),
                "offload_video_to_cpu": LaunchConfiguration("offload_video_to_cpu"),
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
            clip_length_arg,
            clip_stride_arg,
            offload_video_arg,
            sam2_node,
        ]
    )
