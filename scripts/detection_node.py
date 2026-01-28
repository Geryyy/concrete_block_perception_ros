#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError(
        "Ultralytics is not installed. Run: pip install ultralytics"
    ) from e

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__("yolo_detector_node")

        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter("model_path", "best.pt")
        self.declare_parameter("image_topic", "/image")
        self.declare_parameter("confidence", 0.5)
        self.declare_parameter("imgsz", 1280)
        self.declare_parameter("device", "0")
        self.declare_parameter("show_debug_window", False)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf = self.get_parameter("confidence").value
        self.imgsz = self.get_parameter("imgsz").value
        self.device = self.get_parameter("device").value
        self.show_debug = self.get_parameter("show_debug_window").value

        # ----------------------------
        # YOLO model
        # ----------------------------
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        # ----------------------------
        # ROS interfaces
        # ----------------------------
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, self.get_parameter("image_topic").value, self.image_callback, 10
        )

        self.centers_pub = self.create_publisher(PoseArray, "detections/centers", 10)

        self.debug_img_pub = self.create_publisher(Image, "detections/debug_image", 10)

        self.get_logger().info("YOLO detector node ready.")

    # ------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------
    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model(
            source=img,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        r = results[0]
        img_vis = img.copy()

        pose_array = PoseArray()
        pose_array.header = msg.header

        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                # ----------------------------
                # PoseArray (pixel coordinates)
                # ----------------------------
                p = Pose()
                p.position.x = cx
                p.position.y = cy
                p.position.z = 0.0
                pose_array.poses.append(p)

                # ----------------------------
                # Debug drawing
                # ----------------------------
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img_vis, (int(cx), int(cy)), 5, (255, 0, 0), -1)

                label = f"{self.model.names[cls]} {conf:.2f}"
                cv2.putText(
                    img_vis,
                    label,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # ----------------------------
        # Publish outputs
        # ----------------------------
        self.centers_pub.publish(pose_array)

        debug_msg = self.bridge.cv2_to_imgmsg(img_vis, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_img_pub.publish(debug_msg)

        # ----------------------------
        # Optional OpenCV window
        # ----------------------------
        if self.show_debug:
            cv2.imshow("YOLO detections", img_vis)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
