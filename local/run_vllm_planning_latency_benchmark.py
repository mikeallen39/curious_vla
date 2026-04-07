#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests

import run_vllm_semantic_validation as semantic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a planning-style latency benchmark through a running vLLM-Ascend OpenAI-compatible API."
    )
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:18002/v1")
    parser.add_argument("--model-name", required=True, help="Model name/path exposed by the server.")
    parser.add_argument("--warmup-root", required=True, help="Warmup dataset root.")
    parser.add_argument("--scene-dir", default=None, help="Directory containing scene pickle files.")
    parser.add_argument("--scene-path", action="append", default=None, help="Optional explicit scene pickle path. Can be repeated.")
    parser.add_argument("--scene-limit", type=int, default=4, help="Number of unique scenes to use when scene-path is not provided.")
    parser.add_argument("--scene-offset", type=int, default=0, help="Start offset into the sorted scene list.")
    parser.add_argument("--selection-mode", choices=["sorted", "diverse-by-command"], default="diverse-by-command")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs.")
    parser.add_argument("--benchmark-runs", type=int, default=4, help="Number of measured runs.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max completion tokens per request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--timeout-sec", type=float, default=300.0, help="HTTP timeout per request.")
    parser.add_argument("--width", type=int, default=1280, help="Resize width for VL image upload.")
    parser.add_argument("--height", type=int, default=704, help="Resize height for VL image upload.")
    parser.add_argument("--use-raw-resolution", action="store_true", help="Upload raw image resolution instead of resized images.")
    parser.add_argument("--run-text-planning-control", action="store_true", help="Run the text-only planning control before benchmarking.")
    parser.add_argument("--min-pass-rate", type=float, default=1.0, help="Minimum overall valid rate required for recommendation.")
    parser.add_argument("--max-step-distance-m", type=float, default=12.0)
    parser.add_argument("--max-final-displacement-m", type=float, default=40.0)
    parser.add_argument("--max-abs-yaw-rad", type=float, default=3.2)
    parser.add_argument("--max-yaw-step-rad", type=float, default=1.2)
    parser.add_argument("--max-first-step-error-m", type=float, default=2.5)
    parser.add_argument("--output-json", default=None, help="Optional report path.")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    return semantic.percentile(values, q)


def summarize_reports(reports: list[dict], key: str) -> dict:
    values = [float(report[key]) for report in reports if report.get(key) is not None]
    return {
        f"mean_{key}": round(statistics.mean(values), 6) if values else None,
        f"p50_{key}": round(percentile(values, 50), 6) if values else None,
        f"p95_{key}": round(percentile(values, 95), 6) if values else None,
        f"min_{key}": round(min(values), 6) if values else None,
        f"max_{key}": round(max(values), 6) if values else None,
    }


def prefix_keys(data: dict, prefix: str) -> dict:
    return {f"{prefix}{key}": value for key, value in data.items()}


def run_scene_once(
    session: requests.Session,
    scene_path: Path,
    sensor_blobs_root: Path,
    args: argparse.Namespace,
    means,
    stds,
) -> dict:
    start = time.perf_counter()
    scene_report = semantic.run_scene_validation(session, scene_path, sensor_blobs_root, args, means, stds)
    total_scene_sec = time.perf_counter() - start
    request_latency_sec = scene_report.get("latency_sec")
    scene_report["total_scene_sec"] = round(total_scene_sec, 6)
    if request_latency_sec is not None:
        scene_report["client_overhead_sec"] = round(max(0.0, total_scene_sec - float(request_latency_sec)), 6)
    else:
        scene_report["client_overhead_sec"] = None
    return scene_report


def main() -> None:
    args = parse_args()
    warmup_root = Path(args.warmup_root).expanduser().resolve()
    sensor_blobs_root = warmup_root / "sensor_blobs"
    if not sensor_blobs_root.exists():
        raise FileNotFoundError(f"Sensor blobs path does not exist: {sensor_blobs_root}")

    scene_paths = semantic.pick_scene_paths(args)
    means, stds = semantic.load_stats()
    session = requests.Session()

    report: dict = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "base_url": args.base_url,
        "model_name": args.model_name,
        "warmup_root": str(warmup_root),
        "scene_paths": [str(p) for p in scene_paths],
        "scene_limit": args.scene_limit,
        "scene_offset": args.scene_offset,
        "selection_mode": args.selection_mode,
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "timeout_sec": args.timeout_sec,
        "resize": {
            "mode": "raw" if args.use_raw_resolution else "fixed",
            "width": None if args.use_raw_resolution else args.width,
            "height": None if args.use_raw_resolution else args.height,
        },
        "validation_thresholds": {
            "max_step_distance_m": args.max_step_distance_m,
            "max_final_displacement_m": args.max_final_displacement_m,
            "max_abs_yaw_rad": args.max_abs_yaw_rad,
            "max_yaw_step_rad": args.max_yaw_step_rad,
            "max_first_step_error_m": args.max_first_step_error_m,
            "min_pass_rate": args.min_pass_rate,
        },
    }

    if args.run_text_planning_control:
        print("Running text-only planning gate...")
        text_planning_control = semantic.run_text_planning_control(session, args, means, stds)
        report["text_planning_control"] = text_planning_control
        if text_planning_control["request_ok"]:
            print(
                "  overall_valid={overall} latency={latency:.3f}s".format(
                    overall=text_planning_control["overall_valid"],
                    latency=text_planning_control["latency_sec"],
                )
            )
        else:
            print(f"  text planning gate failed: {text_planning_control['error']}")

    total_runs = args.warmup_runs + args.benchmark_runs
    warmup_reports: list[dict] = []
    benchmark_reports: list[dict] = []

    for idx in range(total_runs):
        scene_path = scene_paths[idx % len(scene_paths)]
        phase = "warmup" if idx < args.warmup_runs else "benchmark"
        phase_idx = idx + 1 if phase == "warmup" else idx - args.warmup_runs + 1
        phase_total = args.warmup_runs if phase == "warmup" else args.benchmark_runs
        print(f"{phase.capitalize()} {phase_idx}/{phase_total}: {scene_path.name}")
        scene_report = run_scene_once(session, scene_path, sensor_blobs_root, args, means, stds)
        latency_text = "n/a" if scene_report["latency_sec"] is None else f"{scene_report['latency_sec']:.3f}s"
        print(
            "  total={total:.3f}s request={request} overhead={overhead} overall_valid={overall}".format(
                total=scene_report["total_scene_sec"],
                request=latency_text,
                overhead="n/a" if scene_report["client_overhead_sec"] is None else f"{scene_report['client_overhead_sec']:.3f}s",
                overall=scene_report["overall_valid"],
            )
        )
        if phase == "warmup":
            warmup_reports.append(scene_report)
        else:
            benchmark_reports.append(scene_report)

    request_ok_count = sum(1 for report_item in benchmark_reports if report_item["request_ok"])
    contract_valid_count = sum(1 for report_item in benchmark_reports if report_item["contract"]["contract_valid"])
    intent_valid_count = sum(1 for report_item in benchmark_reports if report_item["intent_alignment"]["intent_alignment_valid"])
    trajectory_valid_count = sum(1 for report_item in benchmark_reports if report_item["trajectory"]["trajectory_valid"])
    overall_valid_count = sum(1 for report_item in benchmark_reports if report_item["overall_valid"])
    validated_reports = [report_item for report_item in benchmark_reports if report_item["overall_valid"]]

    benchmark_summary = {
        "count": len(benchmark_reports),
        "request_ok_count": request_ok_count,
        "contract_valid_count": contract_valid_count,
        "intent_alignment_valid_count": intent_valid_count,
        "trajectory_valid_count": trajectory_valid_count,
        "overall_valid_count": overall_valid_count,
        "request_ok_rate": round(request_ok_count / len(benchmark_reports), 6) if benchmark_reports else None,
        "contract_valid_rate": round(contract_valid_count / len(benchmark_reports), 6) if benchmark_reports else None,
        "intent_alignment_valid_rate": round(intent_valid_count / len(benchmark_reports), 6) if benchmark_reports else None,
        "trajectory_valid_rate": round(trajectory_valid_count / len(benchmark_reports), 6) if benchmark_reports else None,
        "overall_valid_rate": round(overall_valid_count / len(benchmark_reports), 6) if benchmark_reports else None,
        "validated_count": len(validated_reports),
    }
    benchmark_summary.update(summarize_reports(benchmark_reports, "latency_sec"))
    benchmark_summary.update(summarize_reports(benchmark_reports, "total_scene_sec"))
    benchmark_summary.update(summarize_reports(benchmark_reports, "client_overhead_sec"))
    benchmark_summary.update(prefix_keys(summarize_reports(validated_reports, "latency_sec"), "validated_"))
    benchmark_summary.update(prefix_keys(summarize_reports(validated_reports, "total_scene_sec"), "validated_"))
    benchmark_summary["recommended_for_latency_benchmark"] = bool(
        benchmark_summary["overall_valid_rate"] is not None and benchmark_summary["overall_valid_rate"] >= args.min_pass_rate
    )

    report["warmup_reports"] = warmup_reports
    report["benchmark_reports"] = benchmark_reports
    report["benchmark_summary"] = benchmark_summary

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
