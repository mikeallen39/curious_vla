#!/usr/bin/env python3

import argparse
import json
import math
import os
import pickle
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import torch


def patch_torch_compiler() -> None:
    if not hasattr(torch, "compiler"):
        torch.compiler = SimpleNamespace(is_compiling=lambda: False)
    elif not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False


patch_torch_compiler()

import torch_npu  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAVSIM_ROOT = PROJECT_ROOT / "navsim_eval"
if str(NAVSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(NAVSIM_ROOT))

os.environ.setdefault("STATS_PATH", str(PROJECT_ROOT / "stats" / "trajectory_stats_train.json"))

from navsim.agents.curious_vla.curious_vla_client import parse_trajectory_string_with_yaw  # noqa: E402
from navsim.agents.curious_vla.curious_vla_config import CuriousVlaConfig  # noqa: E402
from navsim.agents.curious_vla.navsim_qwen_norm_agent_cot import NavsimCoTQwenAgent  # noqa: E402
from navsim.common.dataclasses import AgentInput, SceneMetadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a planning latency benchmark through NavsimCoTQwenAgent.compute_trajectory() on Ascend NPU."
    )
    parser.add_argument("--model-dir", required=True, help="Local model directory.")
    parser.add_argument("--warmup-root", required=True, help="Warmup dataset root.")
    parser.add_argument(
        "--scene-dir",
        default=None,
        help="Directory containing scene pickle files. Defaults to <warmup-root>/openscene_meta_datas.",
    )
    parser.add_argument("--scene-path", default=None, help="Optional single scene pickle path.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs.")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Number of measured runs.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length per request.")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype on NPU.",
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device index.")
    parser.add_argument("--width", type=int, default=1280, help="Optional resize width.")
    parser.add_argument("--height", type=int, default=704, help="Optional resize height.")
    parser.add_argument("--use-raw-resolution", action="store_true", help="Skip resizing and use raw image resolution.")
    parser.add_argument("--output-json", default=None, help="Optional JSON report path.")
    parser.add_argument("--agent-log-dir", default=None, help="Optional directory for agent JSONL logs.")
    parser.add_argument("--max-step-distance-m", type=float, default=12.0, help="Max allowed distance between consecutive predicted trajectory points.")
    parser.add_argument("--max-final-displacement-m", type=float, default=40.0, help="Max allowed final displacement magnitude.")
    parser.add_argument("--max-abs-yaw-rad", type=float, default=3.2, help="Max allowed absolute yaw in prediction.")
    parser.add_argument("--max-yaw-step-rad", type=float, default=1.2, help="Max allowed yaw change between consecutive predicted points.")
    parser.add_argument("--max-first-step-error-m", type=float, default=2.5, help="Max allowed 0.5s XY error against available future ground truth.")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def pick_scene_paths(args: argparse.Namespace) -> list[Path]:
    if args.scene_path:
        return [Path(args.scene_path).expanduser().resolve()]

    scene_dir = Path(args.scene_dir).expanduser().resolve() if args.scene_dir else Path(args.warmup_root).expanduser().resolve() / "openscene_meta_datas"
    scene_paths = sorted(scene_dir.glob("*.pkl"))
    if not scene_paths:
        raise FileNotFoundError(f"No scene pickle files found under {scene_dir}")
    return scene_paths


def load_scene_pickle(scene_path: Path):
    with open(scene_path, "rb") as f:
        scene_dict_list = pickle.load(f)
    if len(scene_dict_list) < 4:
        raise ValueError(f"Scene {scene_path} has only {len(scene_dict_list)} frames, expected at least 4.")
    return scene_dict_list


def frame_to_global_pose(frame) -> np.ndarray:
    translation = frame["ego2global_translation"]
    yaw = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0]
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def absolute_to_relative_pose(origin: np.ndarray, pose: np.ndarray) -> np.ndarray:
    dx = pose[0] - origin[0]
    dy = pose[1] - origin[1]
    c = math.cos(origin[2])
    s = math.sin(origin[2])
    x = c * dx + s * dy
    y = -s * dx + c * dy
    yaw = wrap_angle(pose[2] - origin[2])
    return np.array([x, y, yaw], dtype=np.float32)


def extract_reference_future(scene_dict_list, num_history_frames: int = 4, max_steps: int = 8) -> np.ndarray:
    current_idx = num_history_frames - 1
    origin = frame_to_global_pose(scene_dict_list[current_idx])
    future_frames = scene_dict_list[current_idx + 1 : current_idx + 1 + max_steps]
    if not future_frames:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([absolute_to_relative_pose(origin, frame_to_global_pose(frame)) for frame in future_frames], axis=0)


def build_scene_stub(scene_dict_list):
    current_frame = scene_dict_list[3]
    scene_metadata = SceneMetadata(
        log_name=current_frame["log_name"],
        scene_token=current_frame["scene_token"],
        map_name=current_frame["map_location"],
        initial_token=current_frame["token"],
        num_history_frames=4,
        num_future_frames=max(0, len(scene_dict_list) - 4),
    )
    return SimpleNamespace(scene_metadata=scene_metadata)


class InstrumentedNavsimAgent(NavsimCoTQwenAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_fallback = False

    def _fallback_to_constant_velocity(self, agent_input):
        self.last_fallback = True
        return super()._fallback_to_constant_velocity(agent_input)


class LocalCuriousVLAClientNPU:
    def __init__(
        self,
        model_dir: Path,
        device: torch.device,
        dtype: torch.dtype,
        max_new_tokens: int,
        width: int,
        height: int,
        use_raw_resolution: bool,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.width = width
        self.height = height
        self.use_raw_resolution = use_raw_resolution
        self.last_stats = {}
        self.last_output_text = ""

        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)

        load_start = time.perf_counter()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        torch.npu.synchronize()
        self.model_load_sec = time.perf_counter() - load_start

    def _build_inputs(self, messages):
        system_text = ""
        user_text = ""
        for message in messages["messages"]:
            if message["role"] == "system":
                system_text = message["content"]
            elif message["role"] == "user":
                user_text = message["content"]

        image_path = Path(messages["images"][0])
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size

        if self.use_raw_resolution:
            processed_image = image
        else:
            processed_image = image.resize((self.width, self.height), Image.Resampling.BICUBIC)

        chat_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": processed_image},
                    {"type": "text", "text": user_text.replace("<image>", "").strip()},
                ],
            },
        ]
        text = self.processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[processed_image], padding=True, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        input_meta = {
            "image_path": str(image_path),
            "original_size": [orig_width, orig_height],
            "processed_size": list(processed_image.size),
            "input_ids_shape": list(inputs["input_ids"].shape),
            "pixel_values_shape": list(inputs["pixel_values"].shape),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
        }
        return inputs, input_meta

    def forward(self, messages, use_yaw_parser=False, temperature=0.0, **kwargs):
        stats = {}

        prepare_start = time.perf_counter()
        inputs, input_meta = self._build_inputs(messages)
        torch.npu.synchronize()
        stats["prepare_inputs_sec"] = time.perf_counter() - prepare_start

        generate_start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=bool(temperature and temperature > 0),
                temperature=temperature,
            )
            torch.npu.synchronize()
        stats["generate_sec"] = time.perf_counter() - generate_start

        decode_start = time.perf_counter()
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        output_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        stats["decode_sec"] = time.perf_counter() - decode_start

        parse_start = time.perf_counter()
        if use_yaw_parser:
            parsed_trajectory = parse_trajectory_string_with_yaw(output_text)
        else:
            parsed_trajectory = parse_trajectory_string_with_yaw(output_text)
        stats["parse_sec"] = time.perf_counter() - parse_start

        stats["generated_tokens"] = int(new_tokens.shape[1])
        stats["total_forward_sec"] = (
            stats["prepare_inputs_sec"] + stats["generate_sec"] + stats["decode_sec"] + stats["parse_sec"]
        )
        stats.update(input_meta)

        self.last_stats = stats
        self.last_output_text = output_text

        if parsed_trajectory is None:
            raise ValueError(f"Failed to parse a valid trajectory from model output: {output_text}")

        return parsed_trajectory, output_text


def build_agent_input(scene_dict_list, sensor_blobs_path: Path, agent: NavsimCoTQwenAgent) -> AgentInput:
    return AgentInput.from_scene_dict_list(
        scene_dict_list,
        sensor_blobs_path=sensor_blobs_path,
        num_history_frames=4,
        sensor_config=agent.get_sensor_config(),
    )


def validate_trajectory(
    predicted_trajectory: np.ndarray,
    reference_future: np.ndarray,
    fallback: bool,
    args: argparse.Namespace,
) -> dict:
    metrics = {
        "non_fallback": not fallback,
        "shape_ok": predicted_trajectory.shape == (8, 3),
        "finite_ok": bool(np.isfinite(predicted_trajectory).all()),
        "reference_steps": int(reference_future.shape[0]),
    }

    if not metrics["shape_ok"] or not metrics["finite_ok"]:
        metrics.update(
            {
                "basic_valid": False,
                "sanity_valid": False,
                "reference_valid": False if metrics["reference_steps"] > 0 else None,
                "validated": False,
            }
        )
        return metrics

    xy = predicted_trajectory[:, :2]
    yaw = predicted_trajectory[:, 2]
    deltas_xy = np.diff(np.vstack([np.zeros((1, 2), dtype=np.float32), xy]), axis=0)
    step_distances = np.linalg.norm(deltas_xy, axis=1)
    yaw_steps = np.array([abs(wrap_angle(float(v))) for v in np.diff(np.concatenate([[0.0], yaw]))], dtype=np.float32)

    final_displacement = float(np.linalg.norm(xy[-1]))
    max_abs_yaw = float(np.max(np.abs(yaw)))
    max_step_distance = float(np.max(step_distances))
    max_yaw_step = float(np.max(yaw_steps))

    metrics.update(
        {
            "final_displacement_m": round(final_displacement, 6),
            "max_step_distance_m": round(max_step_distance, 6),
            "max_abs_yaw_rad": round(max_abs_yaw, 6),
            "max_yaw_step_rad": round(max_yaw_step, 6),
            "basic_valid": bool(not fallback and metrics["shape_ok"] and metrics["finite_ok"]),
        }
    )

    sanity_valid = bool(
        metrics["basic_valid"]
        and max_step_distance <= args.max_step_distance_m
        and final_displacement <= args.max_final_displacement_m
        and max_abs_yaw <= args.max_abs_yaw_rad
        and max_yaw_step <= args.max_yaw_step_rad
    )
    metrics["sanity_valid"] = sanity_valid

    if reference_future.shape[0] > 0:
        compare_steps = min(predicted_trajectory.shape[0], reference_future.shape[0])
        xy_errors = np.linalg.norm(predicted_trajectory[:compare_steps, :2] - reference_future[:compare_steps, :2], axis=1)
        yaw_errors = np.array(
            [
                abs(wrap_angle(float(predicted_trajectory[i, 2] - reference_future[i, 2])))
                for i in range(compare_steps)
            ],
            dtype=np.float32,
        )
        metrics.update(
            {
                "compare_steps": int(compare_steps),
                "ade_m": round(float(np.mean(xy_errors)), 6),
                "fde_m": round(float(xy_errors[-1]), 6),
                "mean_yaw_error_rad": round(float(np.mean(yaw_errors)), 6),
                "final_yaw_error_rad": round(float(yaw_errors[-1]), 6),
                "first_step_error_m": round(float(xy_errors[0]), 6),
                "first_step_yaw_error_rad": round(float(yaw_errors[0]), 6),
            }
        )
        reference_valid = bool(metrics["first_step_error_m"] <= args.max_first_step_error_m)
    else:
        metrics.update(
            {
                "compare_steps": 0,
                "ade_m": None,
                "fde_m": None,
                "mean_yaw_error_rad": None,
                "final_yaw_error_rad": None,
                "first_step_error_m": None,
                "first_step_yaw_error_rad": None,
            }
        )
        reference_valid = None

    metrics["reference_valid"] = reference_valid
    metrics["validated"] = bool(
        sanity_valid and (reference_valid if reference_valid is not None else True)
    )
    return metrics


def run_single_scene(
    scene_path: Path,
    sensor_blobs_path: Path,
    agent: InstrumentedNavsimAgent,
    args: argparse.Namespace,
):
    scene_dict_list = load_scene_pickle(scene_path)
    scene_stub = build_scene_stub(scene_dict_list)
    agent_input = build_agent_input(scene_dict_list, sensor_blobs_path, agent)
    reference_future = extract_reference_future(scene_dict_list)

    agent.last_fallback = False
    total_start = time.perf_counter()
    trajectory = agent.compute_trajectory(agent_input, scene_stub)
    total_sec = time.perf_counter() - total_start

    client_stats = dict(getattr(agent._client, "last_stats", {}))
    client_total_sec = float(client_stats.get("total_forward_sec", 0.0))
    agent_overhead_sec = max(0.0, total_sec - client_total_sec)
    validation = validate_trajectory(trajectory.poses, reference_future, agent.last_fallback, args)

    return {
        "scene_path": str(scene_path),
        "log_name": scene_stub.scene_metadata.log_name,
        "initial_token": scene_stub.scene_metadata.initial_token,
        "fallback": agent.last_fallback,
        "total_sec": total_sec,
        "client_total_sec": client_total_sec,
        "agent_overhead_sec": agent_overhead_sec,
        "client_stats": client_stats,
        "validation": validation,
        "reference_future_preview": reference_future[:2].tolist(),
        "trajectory_preview": trajectory.poses[:2].tolist(),
        "output_preview": getattr(agent._client, "last_output_text", "")[:1000],
    }


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    warmup_root = Path(args.warmup_root).expanduser().resolve()
    sensor_blobs_path = warmup_root / "sensor_blobs"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir does not exist: {model_dir}")
    if not sensor_blobs_path.exists():
        raise FileNotFoundError(f"Sensor blobs path does not exist: {sensor_blobs_path}")

    scene_paths = pick_scene_paths(args)
    total_runs = args.warmup_runs + args.benchmark_runs

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = resolve_dtype(args.dtype)

    print(f"Using device: {device}")
    print(f"Model dir: {model_dir}")
    print(f"Warmup root: {warmup_root}")
    print(f"Scenes available: {len(scene_paths)}")
    print(f"Resize mode: {'raw' if args.use_raw_resolution else f'{args.width}x{args.height}'}")

    backend = LocalCuriousVLAClientNPU(
        model_dir=model_dir,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        width=args.width,
        height=args.height,
        use_raw_resolution=args.use_raw_resolution,
    )

    config = CuriousVlaConfig(model_name_or_path=str(model_dir), log_path=args.agent_log_dir)
    agent = InstrumentedNavsimAgent(config=config)
    agent._client = backend

    print(f"Model load time: {backend.model_load_sec:.3f}s")

    warmup_reports = []
    benchmark_reports = []

    for idx in range(total_runs):
        scene_path = scene_paths[idx % len(scene_paths)]
        phase = "warmup" if idx < args.warmup_runs else "benchmark"
        phase_idx = idx + 1 if phase == "warmup" else idx - args.warmup_runs + 1
        phase_total = args.warmup_runs if phase == "warmup" else args.benchmark_runs
        print(f"{phase.capitalize()} {phase_idx}/{phase_total}: {scene_path.name}")

        report = run_single_scene(scene_path, sensor_blobs_path, agent, args)
        print(
            "  total={total:.3f}s client={client:.3f}s overhead={overhead:.3f}s fallback={fallback} validated={validated}".format(
                total=report["total_sec"],
                client=report["client_total_sec"],
                overhead=report["agent_overhead_sec"],
                fallback=report["fallback"],
                validated=report["validation"]["validated"],
            )
        )

        if phase == "warmup":
            warmup_reports.append(report)
        else:
            benchmark_reports.append(report)

    total_latencies = [r["total_sec"] for r in benchmark_reports]
    client_latencies = [r["client_total_sec"] for r in benchmark_reports]
    overhead_latencies = [r["agent_overhead_sec"] for r in benchmark_reports]
    fallback_count = sum(1 for r in benchmark_reports if r["fallback"])
    validated_reports = [r for r in benchmark_reports if r["validation"]["validated"]]
    non_fallback_count = sum(1 for r in benchmark_reports if r["validation"]["non_fallback"])
    sanity_valid_count = sum(1 for r in benchmark_reports if r["validation"]["sanity_valid"])
    reference_checked_count = sum(1 for r in benchmark_reports if r["validation"]["reference_valid"] is not None)
    reference_valid_count = sum(1 for r in benchmark_reports if r["validation"]["reference_valid"] is True)

    validated_total_latencies = [r["total_sec"] for r in validated_reports]
    validated_client_latencies = [r["client_total_sec"] for r in validated_reports]
    validated_overhead_latencies = [r["agent_overhead_sec"] for r in validated_reports]

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_dir": str(model_dir),
        "warmup_root": str(warmup_root),
        "scene_dir": str(Path(args.scene_dir).expanduser().resolve()) if args.scene_dir else str((warmup_root / "openscene_meta_datas").resolve()),
        "device": str(device),
        "dtype": args.dtype,
        "resize": {
            "mode": "raw" if args.use_raw_resolution else "fixed",
            "width": None if args.use_raw_resolution else args.width,
            "height": None if args.use_raw_resolution else args.height,
        },
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
        "max_new_tokens": args.max_new_tokens,
        "validation_thresholds": {
            "max_step_distance_m": args.max_step_distance_m,
            "max_final_displacement_m": args.max_final_displacement_m,
            "max_abs_yaw_rad": args.max_abs_yaw_rad,
            "max_yaw_step_rad": args.max_yaw_step_rad,
            "max_first_step_error_m": args.max_first_step_error_m,
        },
        "model_load_sec": round(backend.model_load_sec, 6),
        "mean_total_sec": round(statistics.mean(total_latencies), 6),
        "min_total_sec": round(min(total_latencies), 6),
        "max_total_sec": round(max(total_latencies), 6),
        "p50_total_sec": round(percentile(total_latencies, 50), 6),
        "p95_total_sec": round(percentile(total_latencies, 95), 6),
        "mean_client_sec": round(statistics.mean(client_latencies), 6),
        "mean_agent_overhead_sec": round(statistics.mean(overhead_latencies), 6),
        "fallback_count": fallback_count,
        "non_fallback_count": non_fallback_count,
        "sanity_valid_count": sanity_valid_count,
        "reference_checked_count": reference_checked_count,
        "reference_valid_count": reference_valid_count,
        "validated_count": len(validated_reports),
        "validated_mean_total_sec": round(statistics.mean(validated_total_latencies), 6) if validated_total_latencies else None,
        "validated_p50_total_sec": round(percentile(validated_total_latencies, 50), 6) if validated_total_latencies else None,
        "validated_p95_total_sec": round(percentile(validated_total_latencies, 95), 6) if validated_total_latencies else None,
        "validated_mean_client_sec": round(statistics.mean(validated_client_latencies), 6) if validated_client_latencies else None,
        "validated_mean_agent_overhead_sec": round(statistics.mean(validated_overhead_latencies), 6) if validated_overhead_latencies else None,
        "warmup_reports": warmup_reports,
        "benchmark_reports": benchmark_reports,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
