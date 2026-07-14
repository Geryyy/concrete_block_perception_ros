# Standalone YOLO segmentation verification.
#
# Runs the topic-based YolosSegmentorNode directly on a live image topic and
# republishes its annotated debug image on the cbp yolo debug topic. This
# bypasses the segmentor service / world-model orchestrator entirely, so you
# can see exactly what YOLO reports (detections + masks) with no downstream
# filtering. Intended for manual inspection, not to run alongside the full
# perception pipeline (both would publish the same debug topic).
#
# YolosSegmentorNode is a lifecycle node, so this launch also drives the
# transitions (configure -> activate) automatically; without that the node
# stays unconfigured and creates no subscription/publishers.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    pkg_dir = FindPackageShare("concrete_block_perception")

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=PathJoinSubstitution([pkg_dir, "config", "yolo26n_seg_best.onnx"]),
        description="Path to YOLO segmentation ONNX model",
    )
    labels_path_arg = DeclareLaunchArgument(
        "labels_path",
        default_value=PathJoinSubstitution([pkg_dir, "config", "block.names"]),
        description="Path to class labels file",
    )
    use_gpu_arg = DeclareLaunchArgument(
        "use_gpu",
        default_value="true",
        description="Enable GPU inference",
    )
    conf_threshold_arg = DeclareLaunchArgument(
        "conf_threshold",
        default_value="0.4",
        description="Confidence threshold",
    )
    nms_threshold_arg = DeclareLaunchArgument(
        "nms_threshold",
        default_value="0.45",
        description="NMS IoU threshold",
    )
    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/blackfly_rotated/image_rect",
        description="Input RGB image topic (Blackfly)",
    )
    debug_topic_arg = DeclareLaunchArgument(
        "debug_topic",
        default_value="/cbp/debug/yolo_service_debug_image",
        description="Output topic for the annotated YOLO debug image",
    )

    seg_node = LifecycleNode(
        package="ros2_yolos_cpp",
        executable="yolos_segmentor_node",
        name="yolos_segmentor_verify",
        namespace="",
        output="screen",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "labels_path": LaunchConfiguration("labels_path"),
            "use_gpu": LaunchConfiguration("use_gpu"),
            "conf_threshold": LaunchConfiguration("conf_threshold"),
            "nms_threshold": LaunchConfiguration("nms_threshold"),
            "publish_debug_image": True,
            "publish_timing": True,
        }],
        remappings=[
            ("~/image_raw", LaunchConfiguration("image_topic")),
            ("~/debug_image", LaunchConfiguration("debug_topic")),
        ],
    )

    # Lifecycle management: configure on startup, then activate once configured.
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(seg_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )
    activate_on_inactive = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=seg_node,
            goal_state="inactive",
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(seg_node),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        )
    )

    return LaunchDescription([
        model_path_arg,
        labels_path_arg,
        use_gpu_arg,
        conf_threshold_arg,
        nms_threshold_arg,
        image_topic_arg,
        debug_topic_arg,
        activate_on_inactive,
        seg_node,
        configure_event,
    ])
