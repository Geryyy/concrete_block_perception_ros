#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node

from concrete_block_perception.srv import RunPoseEstimation


class BenchmarkClient(Node):
    def __init__(self, service_name: str):
        super().__init__("run_pose_estimation_benchmark")
        self._client = self.create_client(RunPoseEstimation, service_name)

    def wait_ready(self, timeout_s: float) -> bool:
        return self._client.wait_for_service(timeout_sec=timeout_s)

    def call_once(self, mode: str, target_block_id: str, enable_debug: bool, timeout_s: float):
        req = RunPoseEstimation.Request()
        req.mode = mode
        req.target_block_id = target_block_id
        req.enable_debug = enable_debug
        req.timeout_s = float(timeout_s)

        t0 = time.perf_counter()
        future = self._client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s + 2.0)
        t1 = time.perf_counter()

        if not future.done() or future.result() is None:
            return {
                "success": False,
                "message": "service call timed out or failed",
                "latency_ms": (t1 - t0) * 1000.0,
                "num_blocks": 0,
            }

        res = future.result()
        return {
            "success": bool(res.success),
            "message": str(res.message),
            "latency_ms": (t1 - t0) * 1000.0,
            "num_blocks": len(res.blocks.blocks),
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark RunPoseEstimation service for backend A/B runs.")
    parser.add_argument("--service", default="/world_model_node/run_pose_estimation")
    parser.add_argument("--mode", default="SCENE_DISCOVERY")
    parser.add_argument("--target-block-id", default="")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--enable-debug", action="store_true")
    parser.add_argument("--backend-label", default="legacy")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rclpy.init()
    node = BenchmarkClient(args.service)

    if not node.wait_ready(5.0):
        raise RuntimeError(f"Service not available: {args.service}")

    runs = []
    for i in range(args.iterations):
        run = node.call_once(
            args.mode,
            args.target_block_id,
            args.enable_debug,
            args.timeout,
        )
        run["index"] = i
        runs.append(run)
        if args.delay > 0.0 and i < args.iterations - 1:
            time.sleep(args.delay)

    latencies = [r["latency_ms"] for r in runs]
    successes = [1 if r["success"] else 0 for r in runs]

    result = {
        "backend": args.backend_label,
        "service": args.service,
        "mode": args.mode,
        "target_block_id": args.target_block_id,
        "iterations": args.iterations,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "success_rate": sum(successes) / max(len(successes), 1),
            "latency_ms_mean": statistics.fmean(latencies) if latencies else 0.0,
            "latency_ms_p50": statistics.median(latencies) if latencies else 0.0,
            "latency_ms_max": max(latencies) if latencies else 0.0,
            "avg_num_blocks": statistics.fmean([r["num_blocks"] for r in runs]) if runs else 0.0,
        },
        "runs": runs,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
