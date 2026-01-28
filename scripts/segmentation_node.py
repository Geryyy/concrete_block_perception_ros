#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from ultralytics import FastSAM

from sensor_msgs.msg import Image
from concrete_block_perception.srv import SegmentAtPoint


class FastSAMSegmentationService(Node):
    def __init__(self):
        super().__init__("fastsam_segmentation_service")

        # ------------------------------------------------
        # Parameters
        # ------------------------------------------------
        self.declare_parameter("model_path", "FastSAM-x.pt")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("imgsz", 1024)
        self.declare_parameter("conf", 0.1)
        self.declare_parameter("iou", 0.9)
        self.declare_parameter("enable_debug", False)
        self.declare_parameter("select_smallest_mask", False)

        model_path = self.get_parameter("model_path").value
        self.device = self.get_parameter("device").value
        self.imgsz = self.get_parameter("imgsz").value
        self.conf = self.get_parameter("conf").value
        self.iou = self.get_parameter("iou").value
        self.enable_debug = self.get_parameter("enable_debug").value
        self.select_smallest = self.get_parameter("select_smallest_mask").value

        self.get_logger().info(f"Loading FastSAM model: {model_path}")
        self.model = FastSAM(model_path)

        self.bridge = CvBridge()

        self.srv = self.create_service(
            SegmentAtPoint, "segment_at_point", self.handle_request
        )

        self.get_logger().info(
            f"FastSAM service ready (select_smallest_mask={self.select_smallest})"
        )

    # ------------------------------------------------
    # Service callback
    # ------------------------------------------------
    def handle_request(self, req, res):
        img_bgr = self.bridge.imgmsg_to_cv2(req.image, desired_encoding="bgr8")
        point = [req.x, req.y]

        results = self.model(
            img_bgr,
            points=[point],
            labels=[1],
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
        )

        masks = results[0].masks
        if masks is None or masks.data.shape[0] == 0:
            self.get_logger().warn("FastSAM returned no masks.")
            return res

        # ------------------------------------------------
        # Select mask
        # ------------------------------------------------
        mask_tensor = masks.data  # (N, H, W)

        if self.select_smallest and mask_tensor.shape[0] > 1:
            areas = [
                float(mask_tensor[i].sum().item()) for i in range(mask_tensor.shape[0])
            ]
            idx = int(np.argmin(areas))

            self.get_logger().debug(
                f"Selected smallest mask {idx} (area={areas[idx]:.1f})"
            )
        else:
            idx = 0

        mask = mask_tensor[idx]
        mask_np = (mask.cpu().numpy() > 0).astype(np.uint8) * 255

        # ------------------------------------------------
        # Response mask
        # ------------------------------------------------
        res.mask = self.bridge.cv2_to_imgmsg(mask_np, encoding="mono8")
        res.mask.header = req.image.header

        # ------------------------------------------------
        # Debug visualization
        # ------------------------------------------------
        if self.enable_debug:
            vis = img_bgr.copy()
            color = np.array([0, 255, 0], dtype=np.uint8)

            vis[mask_np > 0] = 0.5 * vis[mask_np > 0] + 0.5 * color

            cv2.circle(vis, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"mask={idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            res.debug_image = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            res.debug_image.header = req.image.header

        return res


def main(args=None):
    rclpy.init(args=args)
    node = FastSAMSegmentationService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
