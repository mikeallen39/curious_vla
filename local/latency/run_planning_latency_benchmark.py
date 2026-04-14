#!/usr/bin/env python3

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import requests

import planning_latency_http_common as http_common


LOCAL_BACKENDS = {"hf", "transformers", "local"}
HTTP_BACKENDS = {"vllm", "lf"}


def normalize_backend(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in LOCAL_BACKENDS:
        return "hf"
    if lowered in HTTP_BACKENDS:
        return lowered
    raise ValueError(f"Unsupported backend: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified planning latency benchmark entrypoint for local HF and OpenAI-compatible backends."
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "transformers", "local", "vllm", "lf"],
        required=True,
        help="Execution backend. hf/transformers/local means in-process Transformers; vllm/lf means HTTP API.",
    )
    parser.add_argument(
        "--device-type",
        choices=["gpu", "npu"],
        required=True,
        help="Execution device type. For HTTP backends this is metadata and environment selection only.",
    )
    parser.add_argument("--model", required=True, help="Local model directory for hf, or served model name for HTTP backends.")
    parser.add_argument("--warmup-root", required=True, help="Warmup dataset root.")
    parser.add_argument("--scene-dir", default=None, help="Directory containing scene pickle files.")
    parser.add_argument("--scene-path", action="append", default=None, help="Optional explicit scene pickle path. Can be repeated.")
    parser.add_argument("--scene-limit", type=int, default=4, help="Number of unique scenes to use when scene-path is not provided.")
    parser.add_argument("--scene-offset", type=int, default=0, help="Start offset into the sorted scene list.")
    parser.add_argument("--selection-mode", choices=["sorted", "diverse-by-command"], default="diverse-by-command")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs.")
    parser.add_argument("--benchmark-runs", type=int, default=4, help="Number of measured runs.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length per request.")
    parser.add_argument("--max-tokens", type=int, default=None, help="HTTP max completion tokens override.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--timeout-sec", type=float, default=300.0, help="HTTP timeout per request.")
    parser.add_argument("--device", type=int, default=0, help="Local device index for hf backend.")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype for hf backend.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Resize width for VL image input.")
    parser.add_argument("--height", type=int, default=704, help="Resize height for VL image input.")
    parser.add_argument("--use-raw-resolution", action="store_true", help="Skip resizing and use raw image resolution.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL for HTTP backends.")
    parser.add_argument("--run-text-planning-control", action="store_true", help="Run the text-only planning control before HTTP benchmarking.")
    parser.add_argument("--min-pass-rate", type=float, default=1.0, help="Minimum overall valid rate required for recommendation.")
    parser.add_argument("--output-json", default=None, help="Optional JSON report path.")
    parser.add_argument("--agent-log-dir", default=None, help="Optional directory for agent JSONL logs in hf mode.")
    parser.add_argument("--max-step-distance-m", type=float, default=12.0, help="Max allowed distance between consecutive predicted trajectory points.")
    parser.add_argument("--max-final-displacement-m", type=float, default=40.0, help="Max allowed final displacement magnitude.")
    parser.add_argument("--max-abs-yaw-rad", type=float, default=3.2, help="Max allowed absolute yaw in prediction.")
    parser.add_argument("--max-yaw-step-rad", type=float, default=1.2, help="Max allowed yaw change between consecutive predicted points.")
    parser.add_argument("--max-first-step-error-m", type=float, default=2.5, help="Max allowed 0.5s XY error against available future ground truth.")
    return parser.parse_args()


def patch_torch_compiler(torch_module: Any) -> None:
    if not hasattr(torch_module, "compiler"):
        torch_module.compiler = SimpleNamespace(is_compiling=lambda: False)
    elif not hasattr(torch_module.compiler, "is_compiling"):
        torch_module.compiler.is_compiling = lambda: False


def resolve_dtype(torch_module: Any, name: str):
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    return mapping[name]


def percentile(values: list[float], q: float) -> float | None:
    return http_common.percentile(values, q)


def summarize_reports(reports: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(report[key]) for report in reports if report.get(key) is not None]
    return {
        f"mean_{key}": round(statistics.mean(values), 6) if values else None,
        f"p50_{key}": round(percentile(values, 50), 6) if values else None,
        f"p95_{key}": round(percentile(values, 95), 6) if values else None,
        f"min_{key}": round(min(values), 6) if values else None,
        f"max_{key}": round(max(values), 6) if values else None,
    }


def prefix_keys(data: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in data.items()}


def save_report(report: dict[str, Any], output_json: str | None) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if output_json:
        output_path = Path(output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report to {output_path}")


class GenericLocalCuriousVLAClient:
    def __init__(
        self,
        *,
        model_dir: Path,
        device: Any,
        dtype: Any,
        max_new_tokens: int,
        width: int,
        height: int,
        use_raw_resolution: bool,
        torch_module: Any,
        sync_device,
        image_module: Any,
        processor_cls: Any,
        model_cls: Any,
        parse_trajectory_fn,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.width = width
        self.height = height
        self.use_raw_resolution = use_raw_resolution
        self.torch = torch_module
        self.sync_device = sync_device
        self.image_module = image_module
        self.parse_trajectory_fn = parse_trajectory_fn
        self.last_stats: dict[str, Any] = {}
        self.last_output_text = ""

        self.processor = processor_cls.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)

        load_start = time.perf_counter()
        self.model = model_cls.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.sync_device()
        self.model_load_sec = time.perf_counter() - load_start

    def _build_inputs(self, messages: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        system_text = ""
        user_text = ""
        for message in messages["messages"]:
            if message["role"] == "system":
                system_text = message["content"]
            elif message["role"] == "user":
                user_text = message["content"]

        image_path = Path(messages["images"][0])
        image = self.image_module.open(image_path).convert("RGB")
        orig_width, orig_height = image.size

        if self.use_raw_resolution:
            processed_image = image
        else:
            processed_image = image.resize((self.width, self.height), self.image_module.Resampling.BICUBIC)

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

    def forward(self, messages: dict[str, Any], use_yaw_parser: bool = False, temperature: float = 0.0, **kwargs):
        del use_yaw_parser, kwargs
        stats: dict[str, Any] = {}

        prepare_start = time.perf_counter()
        inputs, input_meta = self._build_inputs(messages)
        self.sync_device()
        stats["prepare_inputs_sec"] = time.perf_counter() - prepare_start

        generate_start = time.perf_counter()
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=bool(temperature and temperature > 0),
                temperature=temperature,
            )
            self.sync_device()
        stats["generate_sec"] = time.perf_counter() - generate_start

        decode_start = time.perf_counter()
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        output_text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        stats["decode_sec"] = time.perf_counter() - decode_start

        parse_start = time.perf_counter()
        parsed_trajectory = self.parse_trajectory_fn(output_text)
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


class LocalPlanningRunner:
    def __init__(self, args: argparse.Namespace, warmup_root: Path, sensor_blobs_root: Path) -> None:
        self.args = args
        self.warmup_root = warmup_root
        self.sensor_blobs_root = sensor_blobs_root
        self.model_dir = Path(args.model).expanduser().resolve()
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model dir does not exist: {self.model_dir}")

        self.torch = __import__("torch")
        patch_torch_compiler(self.torch)

        if args.device_type == "npu":
            try:
                __import__("torch_npu")
            except ImportError as exc:
                raise RuntimeError("device-type=npu requires torch_npu in the active environment.") from exc
            self.torch.npu.set_device(args.device)
            self.device = self.torch.device(f"npu:{args.device}")
            self.sync_device = self.torch.npu.synchronize
        else:
            if not self.torch.cuda.is_available():
                raise RuntimeError("device-type=gpu requested, but CUDA is not available.")
            self.torch.cuda.set_device(args.device)
            self.device = self.torch.device(f"cuda:{args.device}")
            self.sync_device = self.torch.cuda.synchronize

        project_root = Path(__file__).resolve().parents[2]
        navsim_root = project_root / "navsim_eval"
        if str(navsim_root) not in sys.path:
            sys.path.insert(0, str(navsim_root))
        os.environ.setdefault("STATS_PATH", str(project_root / "stats" / "trajectory_stats_train.json"))

        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from navsim.agents.curious_vla.curious_vla_client import parse_trajectory_string_with_yaw
        from navsim.agents.curious_vla.curious_vla_config import CuriousVlaConfig
        from navsim.agents.curious_vla.navsim_qwen_norm_agent_cot import NavsimCoTQwenAgent
        from navsim.common.dataclasses import AgentInput, SceneMetadata

        self.AgentInput = AgentInput
        self.SceneMetadata = SceneMetadata
        self.CuriousVlaConfig = CuriousVlaConfig

        class InstrumentedNavsimAgent(NavsimCoTQwenAgent):
            def __init__(self, *inner_args, **inner_kwargs):
                super().__init__(*inner_args, **inner_kwargs)
                self.last_fallback = False

            def _fallback_to_constant_velocity(self, agent_input):
                self.last_fallback = True
                return super()._fallback_to_constant_velocity(agent_input)

        self.InstrumentedNavsimAgent = InstrumentedNavsimAgent

        backend = GenericLocalCuriousVLAClient(
            model_dir=self.model_dir,
            device=self.device,
            dtype=resolve_dtype(self.torch, args.dtype),
            max_new_tokens=args.max_new_tokens,
            width=args.width,
            height=args.height,
            use_raw_resolution=args.use_raw_resolution,
            torch_module=self.torch,
            sync_device=self.sync_device,
            image_module=Image,
            processor_cls=AutoProcessor,
            model_cls=Qwen2_5_VLForConditionalGeneration,
            parse_trajectory_fn=parse_trajectory_string_with_yaw,
        )
        self.backend = backend

        config = self.CuriousVlaConfig(model_name_or_path=str(self.model_dir), log_path=args.agent_log_dir)
        self.agent = self.InstrumentedNavsimAgent(config=config)
        self.agent._client = backend

    def build_scene_stub(self, scene_dict_list: list[dict[str, Any]]):
        current_frame = scene_dict_list[3]
        scene_metadata = self.SceneMetadata(
            log_name=current_frame["log_name"],
            scene_token=current_frame["scene_token"],
            map_name=current_frame["map_location"],
            initial_token=current_frame["token"],
            num_history_frames=4,
            num_future_frames=max(0, len(scene_dict_list) - 4),
        )
        return SimpleNamespace(scene_metadata=scene_metadata)

    def build_agent_input(self, scene_dict_list: list[dict[str, Any]]):
        return self.AgentInput.from_scene_dict_list(
            scene_dict_list,
            sensor_blobs_path=self.sensor_blobs_root,
            num_history_frames=4,
            sensor_config=self.agent.get_sensor_config(),
        )

    def validate_trajectory(
        self,
        predicted_trajectory: np.ndarray,
        reference_future: np.ndarray,
        fallback: bool,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {
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
        yaw_steps = np.array(
            [abs(http_common.wrap_angle(float(v))) for v in np.diff(np.concatenate([[0.0], yaw]))],
            dtype=np.float32,
        )

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
            and max_step_distance <= self.args.max_step_distance_m
            and final_displacement <= self.args.max_final_displacement_m
            and max_abs_yaw <= self.args.max_abs_yaw_rad
            and max_yaw_step <= self.args.max_yaw_step_rad
        )
        metrics["sanity_valid"] = sanity_valid

        if reference_future.shape[0] > 0:
            compare_steps = min(predicted_trajectory.shape[0], reference_future.shape[0])
            xy_errors = np.linalg.norm(
                predicted_trajectory[:compare_steps, :2] - reference_future[:compare_steps, :2],
                axis=1,
            )
            yaw_errors = np.array(
                [
                    abs(http_common.wrap_angle(float(predicted_trajectory[i, 2] - reference_future[i, 2])))
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
            reference_valid = bool(metrics["first_step_error_m"] <= self.args.max_first_step_error_m)
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
        metrics["validated"] = bool(sanity_valid and (reference_valid if reference_valid is not None else True))
        return metrics

    def run_scene_once(self, scene_path: Path) -> dict[str, Any]:
        scene_dict_list = http_common.load_scene_pickle(scene_path)
        scene_stub = self.build_scene_stub(scene_dict_list)
        agent_input = self.build_agent_input(scene_dict_list)
        reference_future = http_common.extract_reference_future(scene_dict_list)

        self.agent.last_fallback = False
        total_start = time.perf_counter()
        trajectory = self.agent.compute_trajectory(agent_input, scene_stub)
        total_sec = time.perf_counter() - total_start

        client_stats = dict(getattr(self.agent._client, "last_stats", {}))
        client_total_sec = float(client_stats.get("total_forward_sec", 0.0))
        agent_overhead_sec = max(0.0, total_sec - client_total_sec)
        validation = self.validate_trajectory(trajectory.poses, reference_future, self.agent.last_fallback)

        current_frame = scene_dict_list[3]
        return {
            "scene_path": str(scene_path),
            "scene_name": current_frame.get("scene_name"),
            "scene_token": current_frame.get("scene_token"),
            "log_name": current_frame.get("log_name"),
            "command": http_common.current_command_str(scene_dict_list),
            "fallback": self.agent.last_fallback,
            "total_sec": round(total_sec, 6),
            "client_total_sec": round(client_total_sec, 6),
            "agent_overhead_sec": round(agent_overhead_sec, 6),
            "client_stats": client_stats,
            "validation": validation,
            "reference_future_preview": reference_future[:2].tolist(),
            "trajectory_preview": trajectory.poses[:2].tolist(),
            "output_preview": getattr(self.agent._client, "last_output_text", "")[:1000],
        }

    def run(self, scene_paths: list[Path]) -> dict[str, Any]:
        total_runs = self.args.warmup_runs + self.args.benchmark_runs

        print(f"Using device: {self.device}")
        print(f"Model dir: {self.model_dir}")
        print(f"Warmup root: {self.warmup_root}")
        print(f"Scenes selected: {len(scene_paths)}")
        print(f"Resize mode: {'raw' if self.args.use_raw_resolution else f'{self.args.width}x{self.args.height}'}")
        print(f"Model load time: {self.backend.model_load_sec:.3f}s")

        warmup_reports: list[dict[str, Any]] = []
        benchmark_reports: list[dict[str, Any]] = []

        for idx in range(total_runs):
            scene_path = scene_paths[idx % len(scene_paths)]
            phase = "warmup" if idx < self.args.warmup_runs else "benchmark"
            phase_idx = idx + 1 if phase == "warmup" else idx - self.args.warmup_runs + 1
            phase_total = self.args.warmup_runs if phase == "warmup" else self.args.benchmark_runs
            print(f"{phase.capitalize()} {phase_idx}/{phase_total}: {scene_path.name}")

            report = self.run_scene_once(scene_path)
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

        fallback_count = sum(1 for r in benchmark_reports if r["fallback"])
        non_fallback_count = sum(1 for r in benchmark_reports if r["validation"]["non_fallback"])
        sanity_valid_count = sum(1 for r in benchmark_reports if r["validation"]["sanity_valid"])
        reference_checked_count = sum(1 for r in benchmark_reports if r["validation"]["reference_valid"] is not None)
        reference_valid_count = sum(1 for r in benchmark_reports if r["validation"]["reference_valid"] is True)
        validated_reports = [r for r in benchmark_reports if r["validation"]["validated"]]

        benchmark_summary: dict[str, Any] = {
            "count": len(benchmark_reports),
            "fallback_count": fallback_count,
            "non_fallback_count": non_fallback_count,
            "sanity_valid_count": sanity_valid_count,
            "reference_checked_count": reference_checked_count,
            "reference_valid_count": reference_valid_count,
            "validated_count": len(validated_reports),
            "non_fallback_rate": round(non_fallback_count / len(benchmark_reports), 6) if benchmark_reports else None,
            "sanity_valid_rate": round(sanity_valid_count / len(benchmark_reports), 6) if benchmark_reports else None,
            "reference_valid_rate": round(reference_valid_count / reference_checked_count, 6) if reference_checked_count else None,
            "validated_rate": round(len(validated_reports) / len(benchmark_reports), 6) if benchmark_reports else None,
        }
        benchmark_summary.update(summarize_reports(benchmark_reports, "total_sec"))
        benchmark_summary.update(summarize_reports(benchmark_reports, "client_total_sec"))
        benchmark_summary.update(summarize_reports(benchmark_reports, "agent_overhead_sec"))
        benchmark_summary.update(prefix_keys(summarize_reports(validated_reports, "total_sec"), "validated_"))
        benchmark_summary.update(prefix_keys(summarize_reports(validated_reports, "client_total_sec"), "validated_"))
        benchmark_summary.update(prefix_keys(summarize_reports(validated_reports, "agent_overhead_sec"), "validated_"))

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "backend": "hf",
            "device_type": self.args.device_type,
            "model": str(self.model_dir),
            "warmup_root": str(self.warmup_root),
            "scene_paths": [str(p) for p in scene_paths],
            "scene_limit": self.args.scene_limit,
            "scene_offset": self.args.scene_offset,
            "selection_mode": self.args.selection_mode,
            "device": str(self.device),
            "dtype": self.args.dtype,
            "resize": {
                "mode": "raw" if self.args.use_raw_resolution else "fixed",
                "width": None if self.args.use_raw_resolution else self.args.width,
                "height": None if self.args.use_raw_resolution else self.args.height,
            },
            "warmup_runs": self.args.warmup_runs,
            "benchmark_runs": self.args.benchmark_runs,
            "max_new_tokens": self.args.max_new_tokens,
            "validation_thresholds": {
                "max_step_distance_m": self.args.max_step_distance_m,
                "max_final_displacement_m": self.args.max_final_displacement_m,
                "max_abs_yaw_rad": self.args.max_abs_yaw_rad,
                "max_yaw_step_rad": self.args.max_yaw_step_rad,
                "max_first_step_error_m": self.args.max_first_step_error_m,
            },
            "model_load_sec": round(self.backend.model_load_sec, 6),
            "warmup_reports": warmup_reports,
            "benchmark_reports": benchmark_reports,
            "benchmark_summary": benchmark_summary,
        }


class HttpPlanningRunner:
    def __init__(self, args: argparse.Namespace, warmup_root: Path, sensor_blobs_root: Path) -> None:
        if not args.base_url:
            raise ValueError(f"backend={args.backend} requires --base-url.")
        self.args = args
        self.warmup_root = warmup_root
        self.sensor_blobs_root = sensor_blobs_root
        self.session = requests.Session()
        self.means, self.stds = http_common.load_stats()

    def run_scene_once(self, scene_path: Path) -> dict[str, Any]:
        start = time.perf_counter()
        scene_report = http_common.run_scene_validation(
            self.session,
            scene_path,
            self.sensor_blobs_root,
            self.args,
            self.means,
            self.stds,
        )
        total_scene_sec = time.perf_counter() - start
        request_latency_sec = scene_report.get("latency_sec")
        scene_report["total_scene_sec"] = round(total_scene_sec, 6)
        if request_latency_sec is not None:
            scene_report["client_overhead_sec"] = round(max(0.0, total_scene_sec - float(request_latency_sec)), 6)
        else:
            scene_report["client_overhead_sec"] = None
        return scene_report

    def run(self, scene_paths: list[Path]) -> dict[str, Any]:
        report: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "backend": self.args.backend,
            "device_type": self.args.device_type,
            "base_url": self.args.base_url,
            "model": self.args.model,
            "warmup_root": str(self.warmup_root),
            "scene_paths": [str(p) for p in scene_paths],
            "scene_limit": self.args.scene_limit,
            "scene_offset": self.args.scene_offset,
            "selection_mode": self.args.selection_mode,
            "warmup_runs": self.args.warmup_runs,
            "benchmark_runs": self.args.benchmark_runs,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "timeout_sec": self.args.timeout_sec,
            "resize": {
                "mode": "raw" if self.args.use_raw_resolution else "fixed",
                "width": None if self.args.use_raw_resolution else self.args.width,
                "height": None if self.args.use_raw_resolution else self.args.height,
            },
            "validation_thresholds": {
                "max_step_distance_m": self.args.max_step_distance_m,
                "max_final_displacement_m": self.args.max_final_displacement_m,
                "max_abs_yaw_rad": self.args.max_abs_yaw_rad,
                "max_yaw_step_rad": self.args.max_yaw_step_rad,
                "max_first_step_error_m": self.args.max_first_step_error_m,
                "min_pass_rate": self.args.min_pass_rate,
            },
        }

        if self.args.run_text_planning_control:
            print("Running text-only planning gate...")
            text_planning_control = http_common.run_text_planning_control(self.session, self.args, self.means, self.stds)
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

        total_runs = self.args.warmup_runs + self.args.benchmark_runs
        warmup_reports: list[dict[str, Any]] = []
        benchmark_reports: list[dict[str, Any]] = []

        for idx in range(total_runs):
            scene_path = scene_paths[idx % len(scene_paths)]
            phase = "warmup" if idx < self.args.warmup_runs else "benchmark"
            phase_idx = idx + 1 if phase == "warmup" else idx - self.args.warmup_runs + 1
            phase_total = self.args.warmup_runs if phase == "warmup" else self.args.benchmark_runs
            print(f"{phase.capitalize()} {phase_idx}/{phase_total}: {scene_path.name}")
            scene_report = self.run_scene_once(scene_path)
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

        benchmark_summary: dict[str, Any] = {
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
            benchmark_summary["overall_valid_rate"] is not None
            and benchmark_summary["overall_valid_rate"] >= self.args.min_pass_rate
        )

        report["warmup_reports"] = warmup_reports
        report["benchmark_reports"] = benchmark_reports
        report["benchmark_summary"] = benchmark_summary
        return report


def main() -> None:
    args = parse_args()
    args.backend = normalize_backend(args.backend)
    if args.max_tokens is None:
        args.max_tokens = args.max_new_tokens

    warmup_root = Path(args.warmup_root).expanduser().resolve()
    sensor_blobs_root = warmup_root / "sensor_blobs"
    if not sensor_blobs_root.exists():
        raise FileNotFoundError(f"Sensor blobs path does not exist: {sensor_blobs_root}")

    scene_paths = http_common.pick_scene_paths(args)

    if args.backend == "hf":
        runner = LocalPlanningRunner(args, warmup_root, sensor_blobs_root)
    else:
        runner = HttpPlanningRunner(args, warmup_root, sensor_blobs_root)

    report = runner.run(scene_paths)
    save_report(report, args.output_json)


if __name__ == "__main__":
    main()
