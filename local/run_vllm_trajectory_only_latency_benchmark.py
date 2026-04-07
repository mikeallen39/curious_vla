#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

import run_vllm_semantic_validation as semantic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trajectory-only latency benchmark through a running vLLM-Ascend OpenAI-compatible API."
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
    parser.add_argument("--max-tokens", type=int, default=256, help="Max completion tokens per request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--timeout-sec", type=float, default=300.0, help="HTTP timeout per request.")
    parser.add_argument("--width", type=int, default=1280, help="Resize width for VL image upload.")
    parser.add_argument("--height", type=int, default=704, help="Resize height for VL image upload.")
    parser.add_argument("--use-raw-resolution", action="store_true", help="Upload raw image resolution instead of resized images.")
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


def build_trajectory_only_prompt(scene_dict_list: list[dict[str, Any]]) -> str:
    command_str = semantic.current_command_str(scene_dict_list)
    history_trajectory = semantic.build_history_trajectory(scene_dict_list)
    return f"""Suppose you are driving.
Input:
- 1 frame of front-view image collected from the ego-vehicle at the present timestep
Picture 1: <image> the front view of the ego-vehicle
- Current high-level intent (string): {command_str}
- 1.5-second past trajectory(3 steps at 2 Hz): {history_trajectory}
Each trajectory point format: (x:float, y:float, heading:float)

Task: Future Trajectory Prediction
Given the input, directly predict the optimal 4-second normalized future trajectory (8 steps at 2 Hz) of the ego vehicle.
Return only strict JSON with no markdown, no explanation, and no extra keys.
Output format:
{{
  "future_trajectory": [[x, y, heading], [x, y, heading], [x, y, heading], [x, y, heading], [x, y, heading], [x, y, heading], [x, y, heading], [x, y, heading]]
}}
"""


def validate_trajectory_only_contract(obj: dict[str, Any] | None, raw_text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "json_parse_ok": obj is not None,
        "future_trajectory_key_ok": False,
        "no_extra_keys": False,
        "trajectory_parsed_ok": False,
        "contract_valid": False,
    }
    if not raw_text.strip():
        return result
    if obj is None:
        return result

    keys = set(obj.keys())
    result["future_trajectory_key_ok"] = "future_trajectory" in obj
    result["no_extra_keys"] = keys == {"future_trajectory"}
    result["trajectory_parsed_ok"] = semantic.parse_triplets(obj.get("future_trajectory")) is not None
    result["contract_valid"] = bool(
        result["json_parse_ok"]
        and result["future_trajectory_key_ok"]
        and result["no_extra_keys"]
        and result["trajectory_parsed_ok"]
    )
    return result


def run_scene_once(
    session: requests.Session,
    scene_path: Path,
    sensor_blobs_root: Path,
    args: argparse.Namespace,
    means,
    stds,
) -> dict:
    scene_dict_list = semantic.load_scene_pickle(scene_path)
    image_path = semantic.scene_image_path(scene_dict_list, sensor_blobs_root)
    prompt = build_trajectory_only_prompt(scene_dict_list)
    messages, image_meta = semantic.build_openai_messages(image_path, prompt, args)
    reference_future = semantic.extract_reference_future(scene_dict_list)
    expected_intent = semantic.current_command_str(scene_dict_list)

    base_report = {
        "scene_path": str(scene_path),
        "scene_name": scene_dict_list[3]["scene_name"],
        "scene_token": scene_dict_list[3]["scene_token"],
        "log_name": scene_dict_list[3]["log_name"],
        "image_path": str(image_path),
        "image_original_size": image_meta["original_size"],
        "image_processed_size": image_meta["processed_size"],
        "command": expected_intent,
    }

    start = time.perf_counter()
    try:
        response_json, latency_sec = semantic.post_chat_completion(
            session=session,
            base_url=args.base_url,
            model_name=args.model_name,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
        )
        total_scene_sec = time.perf_counter() - start
        raw_text = semantic.extract_message_content(response_json)
        obj = semantic.extract_first_json_object(raw_text)
        contract = validate_trajectory_only_contract(obj, raw_text)
        norm_traj = semantic.extract_trajectory(obj, raw_text)
        denorm_traj = semantic.denormalize_trajectory(norm_traj, means, stds) if norm_traj is not None else None
        trajectory = semantic.validate_denormalized_trajectory(denorm_traj, reference_future, args)
        overall_valid = bool(contract["contract_valid"] and trajectory["trajectory_valid"])
        return {
            **base_report,
            "request_ok": True,
            "latency_sec": round(latency_sec, 6),
            "total_scene_sec": round(total_scene_sec, 6),
            "client_overhead_sec": round(max(0.0, total_scene_sec - float(latency_sec)), 6),
            "raw_text_preview": raw_text[:2000],
            "contract": contract,
            "trajectory": trajectory,
            "overall_valid": overall_valid,
        }
    except Exception as exc:
        total_scene_sec = time.perf_counter() - start
        return {
            **base_report,
            "request_ok": False,
            "error": str(exc),
            "latency_sec": None,
            "total_scene_sec": round(total_scene_sec, 6),
            "client_overhead_sec": None,
            "raw_text_preview": "",
            "contract": {"contract_valid": False},
            "trajectory": {"trajectory_valid": False},
            "overall_valid": False,
        }


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
        "prompt_mode": "trajectory_only",
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
            "max-first-step-error-m": args.max_first_step_error_m,
        },
    }

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

    request_ok_count = sum(1 for item in benchmark_reports if item["request_ok"])
    contract_valid_count = sum(1 for item in benchmark_reports if item["contract"]["contract_valid"])
    trajectory_valid_count = sum(1 for item in benchmark_reports if item["trajectory"]["trajectory_valid"])
    overall_valid_count = sum(1 for item in benchmark_reports if item["overall_valid"])
    validated_reports = [item for item in benchmark_reports if item["overall_valid"]]

    benchmark_summary = {
        "count": len(benchmark_reports),
        "request_ok_count": request_ok_count,
        "contract_valid_count": contract_valid_count,
        "trajectory_valid_count": trajectory_valid_count,
        "overall_valid_count": overall_valid_count,
        "request_ok_rate": round(request_ok_count / len(benchmark_reports), 6) if benchmark_reports else None,
        "contract_valid_rate": round(contract_valid_count / len(benchmark_reports), 6) if benchmark_reports else None,
        "trajectory_valid_rate": round(trajectory_valid_count / len(benchmark_reports), 6) if benchmark_reports else None,
        "overall_valid_rate": round(overall_valid_count / len(benchmark_reports), 6) if benchmark_reports else None,
        "validated_count": len(validated_reports),
    }
    benchmark_summary.update(summarize_reports(benchmark_reports, "latency_sec"))
    benchmark_summary.update(summarize_reports(benchmark_reports, "total_scene_sec"))
    benchmark_summary.update(summarize_reports(benchmark_reports, "client_overhead_sec"))
    benchmark_summary.update({f"validated_{k}": v for k, v in summarize_reports(validated_reports, "latency_sec").items()})
    benchmark_summary.update({f"validated_{k}": v for k, v in summarize_reports(validated_reports, "total_scene_sec").items()})

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
