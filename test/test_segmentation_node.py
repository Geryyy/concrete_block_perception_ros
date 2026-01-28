#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from concrete_block_perception.srv import SegmentAtPoint


class FastSAMServiceTester(Node):
    def __init__(self, image_path, x, y):
        super().__init__("fastsam_service_tester")

        self.image_path = image_path
        self.x = x
        self.y = y

        self.bridge = CvBridge()

        self.cli = self.create_client(SegmentAtPoint, "segment_at_point")

        self.get_logger().info("Waiting for FastSAM service...")
        self.cli.wait_for_service()

        self.send_request()

    def send_request(self):
        img_bgr = cv2.imread(self.image_path)
        if img_bgr is None:
            raise RuntimeError(f"Failed to load image: {self.image_path}")

        req = SegmentAtPoint.Request()
        req.image = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
        req.x = int(self.x)
        req.y = int(self.y)

        self.future = self.cli.call_async(req)
        self.future.add_done_callback(self.handle_response)

    def handle_response(self, future):
        res = future.result()

        mask = self.bridge.imgmsg_to_cv2(res.mask, desired_encoding="mono8")

        cv2.imshow("Returned mask", mask)
        cv2.waitKey(0)

        # Debug image may be empty if disabled
        if res.debug_image.data:
            dbg = self.bridge.imgmsg_to_cv2(res.debug_image, desired_encoding="bgr8")
            cv2.imshow("Debug image", dbg)
            cv2.waitKey(0)

        cv2.destroyAllWindows()
        self.get_logger().info("Test completed.")
        rclpy.shutdown()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", required=True, help="Path to test image", default="./test.png"
    )
    parser.add_argument(
        "--x", type=int, required=True, help="Click x coordinate", default=721
    )
    parser.add_argument(
        "--y", type=int, required=True, help="Click y coordinate", default=388
    )
    args = parser.parse_args()

    rclpy.init()
    FastSAMServiceTester(args.image, args.x, args.y)
    rclpy.spin(rclpy.get_default_context().nodes[0])


if __name__ == "__main__":
    main()
