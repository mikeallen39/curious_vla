#!/usr/bin/env python3

import argparse
import json
import os
import statistics
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


SYSTEM_PROMPT = "You are an expert driver."
USER_PROMPT = """Suppose you are driving. Let's complete the following tasks step by step.
Input:
- 1 frame of front-view image collected from the ego-vehicle at the present timestep
Picture 1: <image> the front view of the ego-vehicle
- Current high-level intent (string): go straight
- 1.5-second past trajectory(3 steps at 2 Hz): - t-2: (+0.00, +0.00, +0.00), - t-1: (+0.00, +0.00, +0.00), - t-0: (+0.00, +0.00, +0.00)
Each trajectory point format: (x:float, y:float, heading:float)
Task 4: Future Trajectory Prediction
Given the input, predict the optimal 4-second normalized future trajectory (8 steps at 2 Hz) of the ego vehicle.
Output strict JSON with key future_trajectory and 8 points of (x, y, heading).
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Curious-VLA latency benchmark on Ascend NPU.")
    parser.add_argument("--model-dir", required=True, help="Local model directory.")
    parser.add_argument("--warmup-root", default=None, help="Warmup dataset root used to auto-pick an image.")
    parser.add_argument("--image-path", default=None, help="Image path to benchmark.")
    parser.add_argument("--width", type=int, default=1280, help="Resize width before tokenization.")
    parser.add_argument("--height", type=int, default=704, help="Resize height before tokenization.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup iterations.")
    parser.add_argument("--benchmark-runs", type=int, default=3, help="Number of measured iterations.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Generation length per request.")
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype on NPU.",
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device index.")
    parser.add_argument("--output-json", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


def find_default_image(warmup_root: Path) -> Path:
    sensor_root = warmup_root / "sensor_blobs"
    matches = sorted(sensor_root.glob("*/CAM_F0/*.jpg"))
    if not matches:
        raise FileNotFoundError(f"No CAM_F0 jpg files found under {sensor_root}")
    return matches[0]


def build_inputs(processor: AutoProcessor, image: Image.Image, device: torch.device):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir does not exist: {model_dir}")

    if args.image_path:
        image_path = Path(args.image_path).expanduser().resolve()
    elif args.warmup_root:
        image_path = find_default_image(Path(args.warmup_root).expanduser().resolve())
    else:
        raise ValueError("Either --image-path or --warmup-root must be provided.")

    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = resolve_dtype(args.dtype)

    print(f"Using image: {image_path}")
    print(f"Using device: {device}")
    print(f"Requested dtype: {args.dtype}")

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)

    load_start = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    torch.npu.synchronize()
    load_sec = time.perf_counter() - load_start

    image = Image.open(image_path).convert("RGB").resize((args.width, args.height), Image.Resampling.BICUBIC)
    inputs = build_inputs(processor, image, device)

    print(f"Model load time: {load_sec:.3f}s")
    print(f"input_ids shape: {tuple(inputs['input_ids'].shape)}")
    print(f"pixel_values shape: {tuple(inputs['pixel_values'].shape)}")

    with torch.no_grad():
        for idx in range(args.warmup_runs):
            print(f"Warmup {idx + 1}/{args.warmup_runs}")
            _ = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            torch.npu.synchronize()

        latencies = []
        last_text = ""
        for idx in range(args.benchmark_runs):
            print(f"Benchmark {idx + 1}/{args.benchmark_runs}")
            start = time.perf_counter()
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            torch.npu.synchronize()
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
            last_text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
            print(f"  latency: {elapsed:.3f}s")

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_dir": str(model_dir),
        "image_path": str(image_path),
        "device": str(device),
        "dtype": args.dtype,
        "resize": {"width": args.width, "height": args.height},
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
        "max_new_tokens": args.max_new_tokens,
        "model_load_sec": round(load_sec, 6),
        "latencies_sec": [round(v, 6) for v in latencies],
        "mean_sec": round(statistics.mean(latencies), 6),
        "min_sec": round(min(latencies), 6),
        "max_sec": round(max(latencies), 6),
        "p50_sec": round(percentile(latencies, 50), 6),
        "p95_sec": round(percentile(latencies, 95), 6),
        "output_preview": last_text[:1000],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
