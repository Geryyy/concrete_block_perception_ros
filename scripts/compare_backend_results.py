#!/usr/bin/env python3

import argparse
import csv
import json


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Compare legacy vs TEASER benchmark JSON outputs.")
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--teaser", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    legacy = load_json(args.legacy)
    teaser = load_json(args.teaser)

    lsum = legacy.get("summary", {})
    tsum = teaser.get("summary", {})

    summary = {
        "legacy_backend": legacy.get("backend", "legacy"),
        "teaser_backend": teaser.get("backend", "teaser"),
        "mode_legacy": legacy.get("mode", ""),
        "mode_teaser": teaser.get("mode", ""),
        "success_rate_legacy": lsum.get("success_rate", 0.0),
        "success_rate_teaser": tsum.get("success_rate", 0.0),
        "success_rate_delta_teaser_minus_legacy": tsum.get("success_rate", 0.0) - lsum.get("success_rate", 0.0),
        "latency_ms_mean_legacy": lsum.get("latency_ms_mean", 0.0),
        "latency_ms_mean_teaser": tsum.get("latency_ms_mean", 0.0),
        "latency_ms_mean_delta_teaser_minus_legacy": tsum.get("latency_ms_mean", 0.0) - lsum.get("latency_ms_mean", 0.0),
        "avg_num_blocks_legacy": lsum.get("avg_num_blocks", 0.0),
        "avg_num_blocks_teaser": tsum.get("avg_num_blocks", 0.0),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
