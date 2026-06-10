from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            parameters=[{"port": 8765}],
        ),
        Node(
            package="cloudini_ros",
            executable="cloudini_topic_converter",
            name="cloudini_topic_converter",
            parameters=[{
                "compressing": False,
                "topic_input": "/seyond_points/compressed",
                "topic_output": "/seyond_points",
            }],
            arguments=["--ros-args", "--log-level", "WARN"],
        ),
    ])
