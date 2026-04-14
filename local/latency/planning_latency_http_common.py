#!/usr/bin/env python3

import argparse
import base64
import io
import json
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = PROJECT_ROOT / "stats" / "trajectory_stats_train.json"

CRITICAL_OBJECT_KEYS = [
    "nearby_vehicle",
    "conflicting_pedestrian",
    "cyclist",
    "construction",
    "traffic_element",
    "weather_condition",
    "road_hazard",
    "emergency_vehicle",
    "animal",
    "special_vehicle",
    "conflicting_vehicle",
    "door_opening_vehicle",
]

SPEED_CHOICES = {"keep", "accelerate", "decelerate", "other"}
COMMAND_CHOICES = {
    "straight",
    "yield",
    "left_turn",
    "right_turn",
    "lane_follow",
    "lane_change_left",
    "lane_change_right",
    "reverse",
    "other",
}
YES_NO = {"yes", "no"}
NAV_COMMANDS = ["turn left", "go straight", "turn right", "unknown"]
INTENT_TO_ALLOWED_META_COMMANDS = {
    "turn left": {"left_turn", "yield", "lane_follow"},
    "go straight": {"straight", "yield", "lane_follow"},
    "turn right": {"right_turn", "yield", "lane_follow"},
    "unknown": COMMAND_CHOICES,
}


def load_stats() -> tuple[np.ndarray, np.ndarray]:
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    means = np.asarray(stats["mean"], dtype=np.float64)
    stds = np.asarray(stats["std"], dtype=np.float64)
    return means, stds


def wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def format_number(n: float, decimal_places: int = 2) -> str | float:
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    return f"{n:+.{decimal_places}f}"


def frame_to_global_pose(frame: dict[str, Any]) -> np.ndarray:
    translation = frame["ego2global_translation"]
    w, x, y, z = frame["ego2global_rotation"]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([translation[0], translation[1], yaw], dtype=np.float64)


def absolute_to_relative_pose(origin: np.ndarray, pose: np.ndarray) -> np.ndarray:
    dx = pose[0] - origin[0]
    dy = pose[1] - origin[1]
    c = math.cos(origin[2])
    s = math.sin(origin[2])
    x = c * dx + s * dy
    y = -s * dx + c * dy
    yaw = wrap_angle(pose[2] - origin[2])
    return np.array([x, y, yaw], dtype=np.float64)


def pick_scene_paths(args: argparse.Namespace) -> list[Path]:
    if args.scene_path:
        return [Path(p).expanduser().resolve() for p in args.scene_path]

    warmup_root = Path(args.warmup_root).expanduser().resolve()
    scene_dir = Path(args.scene_dir).expanduser().resolve() if args.scene_dir else warmup_root / "openscene_meta_datas"
    scene_paths = sorted(scene_dir.glob("*.pkl"))
    if not scene_paths:
        raise FileNotFoundError(f"No scene pickle files found under {scene_dir}")

    if args.selection_mode == "diverse-by-command":
        grouped: dict[str, list[Path]] = {}
        for scene_path in scene_paths:
            try:
                grouped.setdefault(scene_command_from_path(scene_path), []).append(scene_path)
            except Exception:
                grouped.setdefault("unknown", []).append(scene_path)

        labels = [label for label in sorted(grouped.keys()) if grouped[label]]
        picked: list[Path] = []
        group_offset = max(args.scene_offset, 0)
        if group_offset:
            for label in labels:
                grouped[label] = grouped[label][group_offset:]

        while len(picked) < args.scene_limit:
            progressed = False
            for label in labels:
                bucket = grouped[label]
                if bucket:
                    picked.append(bucket.pop(0))
                    progressed = True
                    if len(picked) >= args.scene_limit:
                        break
            if not progressed:
                break
    else:
        start = args.scene_offset
        end = start + args.scene_limit
        picked = scene_paths[start:end]

    if not picked:
        raise ValueError(
            f"Scene selection is empty. scene_offset={args.scene_offset}, "
            f"scene_limit={args.scene_limit}, total={len(scene_paths)}"
        )
    return picked


def load_scene_pickle(scene_path: Path) -> list[dict[str, Any]]:
    with open(scene_path, "rb") as f:
        scene_dict_list = pickle.load(f)
    if len(scene_dict_list) < 5:
        raise ValueError(f"Scene {scene_path} has only {len(scene_dict_list)} frames, expected at least 5.")
    return scene_dict_list


def build_history_trajectory(scene_dict_list: list[dict[str, Any]], current_idx: int = 3) -> str:
    origin = frame_to_global_pose(scene_dict_list[current_idx])
    history_frames = scene_dict_list[:current_idx]
    status_lines = []
    size = len(history_frames)
    for i, frame in enumerate(history_frames):
        rel_pose = absolute_to_relative_pose(origin, frame_to_global_pose(frame))
        status_lines.append(
            f"   - t-{size - i - 1}: "
            f"({format_number(float(rel_pose[0]))}, {format_number(float(rel_pose[1]))}, {format_number(float(rel_pose[2]))})"
        )
    status_lines.append(f"   - t-{0}: ({format_number(0.0)}, {format_number(0.0)}, {format_number(0.0)})")
    return ", ".join(status_lines)


def extract_reference_future(scene_dict_list: list[dict[str, Any]], current_idx: int = 3, max_steps: int = 8) -> np.ndarray:
    origin = frame_to_global_pose(scene_dict_list[current_idx])
    future_frames = scene_dict_list[current_idx + 1 : current_idx + 1 + max_steps]
    if not future_frames:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([absolute_to_relative_pose(origin, frame_to_global_pose(frame)) for frame in future_frames], axis=0)


def scene_image_path(scene_dict_list: list[dict[str, Any]], sensor_blobs_root: Path, current_idx: int = 3) -> Path:
    rel_path = scene_dict_list[current_idx]["cams"]["CAM_F0"]["data_path"]
    image_path = sensor_blobs_root / rel_path
    if not image_path.exists():
        raise FileNotFoundError(f"Front image not found: {image_path}")
    return image_path


def current_command_str(scene_dict_list: list[dict[str, Any]], current_idx: int = 3) -> str:
    one_hot = scene_dict_list[current_idx]["driving_command"]
    if hasattr(one_hot, "tolist"):
        one_hot = one_hot.tolist()
    labels = [NAV_COMMANDS[i] for i, v in enumerate(one_hot) if int(v) == 1]
    return labels[0] if labels else "unknown"


def scene_command_from_path(scene_path: Path) -> str:
    scene_dict_list = load_scene_pickle(scene_path)
    return current_command_str(scene_dict_list)


def build_planning_prompt(scene_dict_list: list[dict[str, Any]]) -> str:
    command_str = current_command_str(scene_dict_list)
    history_trajectory = build_history_trajectory(scene_dict_list)
    q1 = f"""Suppose you are driving. Let's complete the following tasks step by step.
Input:
- 1 frame of front-view image collected from the ego-vehicle at the present timestep
Picture 1: <image> the front view of the ego-vehicle
- Current high-level intent (string): {command_str}
- 1.5-second past trajectory(3 steps at 2 Hz): {history_trajectory}
Each trajectory point format: (x:float, y:float, heading:float)""" + """
Task 1: Critical Objects and Conditions Detection
Decide whether at least one critical instance of each class could influence the ego-vehicle's future path (no omissions). A vehicle can be a car, bus, truck, motorcyclist, scooter, etc. traffic_element includes traffic signs and traffic lights. road_hazard may include hazardous road conditions, road debris, obstacles, etc. A conflicting_vehicle is a vehicle that may potentially conflict with the ego's future path. Output "yes" or "no" for every class (no omissions).
   Object classes to audit:
     - nearby_vehicle
     - conflicting_pedestrian
     - cyclist
     - construction
     - traffic_element
     - weather_condition
     - road_hazard
     - emergency_vehicle
     - animal
     - special_vehicle
     - conflicting_vehicle
     - door_opening_vehicle
"""
    q2 = """Task 2: Natural Language Explanation
Compose a concise natural-language description of the optimal future 5-second trajectory for the ego vehicle that the expert driver (you) plans and explain why the expert driver plans to execute this trajectory.
   - Mention only the classes you marked "yes" in the previous task.
   - Describe how each of those critical objects or conditions influences the optimal trajectory.
   - Do not invent objects or conditions not present in the input.
"""
    q3 = """Task 3: Meta-Behaviour Selection
Assign exactly one category from each list. Choose the label that best summarises the overall behaviour of the optimal future trajectory:
   - speed ∈ { keep, accelerate, decelerate }
   - command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse }
   Choose the label that best summarises the overall behaviour of the optimal future trajectory.
   - If none fits, use 'other', but do this sparingly.
"""
    q4 = """Task 4: Future Trajectory Prediction
(answer output should be wrapped in ...)
Given the input, critical objects/conditions, natural language explanation, and meta-behaviour, predict the optimal 4-second normalized future trajectory (8 steps at 2 Hz) of the ego vehicle. Predict 8 normalized future trajectory points in [PT, ...] format. Each point is (x, y, heading).
Output format (strict JSON, no extra keys, no markdown codeblock chars(```), no commentary):
{
  "critical_objects": {
    "nearby_vehicle": "yes | no",
    "conflicting_pedestrian": "yes | no",
    "cyclist": "yes | no",
    "construction": "yes | no",
    "traffic_element": "yes | no",
    "weather_condition": "yes | no",
    "road_hazard": "yes | no",
    "emergency_vehicle": "yes | no",
    "animal": "yes | no",
    "special_vehicle": "yes | no",
    "conflicting_vehicle": "yes | no",
    "door_opening_vehicle": "yes | no"
  },
  "explanation": "100-word description that references only the classes marked 'yes'",
  "meta_behaviour": {
    "speed": "keep | accelerate | decelerate | other",
    "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left | lane_change_right | reverse | other"
  },
  "future_trajectory": [PT, ...]
}
"""
    return f"{q1}\n{q2}\n{q3}\n{q4}"


def image_to_data_uri(image_path: Path, args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    original_size = list(image.size)
    if args.use_raw_resolution:
        processed = image
    else:
        processed = image.resize((args.width, args.height), Image.Resampling.BICUBIC)

    buffer = io.BytesIO()
    processed.save(buffer, format="JPEG", quality=90)
    image_bytes = buffer.getvalue()
    metadata = {
        "original_size": original_size,
        "processed_size": list(processed.size),
        "use_raw_resolution": bool(args.use_raw_resolution),
    }
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}", metadata


def build_openai_messages(image_path: Path, user_text: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_uri, image_meta = image_to_data_uri(image_path, args)
    return [
        {"role": "system", "content": "You are an expert driver."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": user_text.replace("<image>", "").strip()},
            ],
        },
    ], image_meta


def build_text_planning_control_messages() -> list[dict[str, Any]]:
    prompt = """Suppose you are driving. Return only strict JSON, with no markdown and no extra text.
No image is provided for this control case.
Current high-level intent (string): go straight
1.5-second past trajectory(3 steps at 2 Hz): - t-2: (-1.2, 0.0, 0.0), - t-1: (-0.6, 0.0, 0.0), - t-0: (0.0, 0.0, 0.0)

Assume there are no critical objects or conditions, and the expert should keep a smooth lane-following trajectory.
Output format:
{
  "critical_objects": {
    "nearby_vehicle": "yes | no",
    "conflicting_pedestrian": "yes | no",
    "cyclist": "yes | no",
    "construction": "yes | no",
    "traffic_element": "yes | no",
    "weather_condition": "yes | no",
    "road_hazard": "yes | no",
    "emergency_vehicle": "yes | no",
    "animal": "yes | no",
    "special_vehicle": "yes | no",
    "conflicting_vehicle": "yes | no",
    "door_opening_vehicle": "yes | no"
  },
  "explanation": "brief explanation",
  "meta_behaviour": {
    "speed": "keep | accelerate | decelerate | other",
    "command": "straight | yield | left_turn | right_turn | lane_follow | lane_change_left | lane_change_right | reverse | other"
  },
  "future_trajectory": [[x, y, heading], ... exactly 8 points]
}
"""
    return [
        {"role": "system", "content": "You are an expert driver."},
        {"role": "user", "content": prompt},
    ]


def post_chat_completion(
    session: requests.Session,
    base_url: str,
    model_name: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
) -> tuple[dict[str, Any], float]:
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    start = time.perf_counter()
    response = session.post(
        base_url.rstrip("/") + "/chat/completions",
        json=payload,
        timeout=timeout_sec,
    )
    latency_sec = time.perf_counter() - start
    response.raise_for_status()
    return response.json(), latency_sec


def extract_message_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : idx + 1]
                try:
                    obj = json.loads(candidate)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def parse_triplets_from_text(text: str) -> np.ndarray | None:
    matches = re.findall(
        r"[\[\(]\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*[\]\)]",
        text,
    )
    if len(matches) < 8:
        return None
    points = [(float(x), float(y), float(yaw)) for x, y, yaw in matches[:8]]
    return np.asarray(points, dtype=np.float64)


def parse_triplets(candidate: Any) -> np.ndarray | None:
    if isinstance(candidate, list):
        points = []
        for item in candidate:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    points.append((float(item[0]), float(item[1]), float(item[2])))
                except Exception:
                    return None
            else:
                return None
        if len(points) >= 8:
            return np.asarray(points[:8], dtype=np.float64)
        return None
    if isinstance(candidate, str):
        return parse_triplets_from_text(candidate)
    return None


def extract_trajectory(obj: dict[str, Any] | None, raw_text: str) -> np.ndarray | None:
    if obj and "future_trajectory" in obj:
        parsed = parse_triplets(obj["future_trajectory"])
        if parsed is not None:
            return parsed
    return parse_triplets_from_text(raw_text)


def denormalize_trajectory(norm_traj: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return norm_traj * stds + means


def validate_output_contract(obj: dict[str, Any] | None, raw_text: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "json_parse_ok": obj is not None,
        "required_top_keys_ok": False,
        "critical_objects_ok": False,
        "critical_objects_complete": False,
        "explanation_ok": False,
        "meta_behaviour_ok": False,
        "repetitive_no_artifact": False,
        "contract_valid": False,
    }

    if not raw_text.strip():
        return result

    lowered = raw_text.lower()
    result["repetitive_no_artifact"] = (
        raw_text.count('"no"') >= 3
        and raw_text.count("{") <= 1
        and lowered.count("future_trajectory") == 0
    )

    if obj is None:
        return result

    required = {"critical_objects", "explanation", "meta_behaviour", "future_trajectory"}
    result["required_top_keys_ok"] = required.issubset(set(obj.keys()))

    critical_objects = obj.get("critical_objects")
    if isinstance(critical_objects, dict):
        critical_keys = set(critical_objects.keys())
        result["critical_objects_complete"] = critical_keys == set(CRITICAL_OBJECT_KEYS)
        result["critical_objects_ok"] = result["critical_objects_complete"] and all(
            isinstance(v, str) and v.strip().lower() in YES_NO for v in critical_objects.values()
        )

    explanation = obj.get("explanation")
    if isinstance(explanation, str):
        word_count = len(explanation.split())
        result["explanation_word_count"] = word_count
        result["explanation_ok"] = 3 <= word_count <= 180

    meta = obj.get("meta_behaviour")
    if isinstance(meta, dict):
        speed = meta.get("speed")
        command = meta.get("command")
        result["meta_behaviour_ok"] = (
            isinstance(speed, str)
            and isinstance(command, str)
            and speed in SPEED_CHOICES
            and command in COMMAND_CHOICES
        )

    result["contract_valid"] = bool(
        result["json_parse_ok"]
        and result["required_top_keys_ok"]
        and result["critical_objects_ok"]
        and result["explanation_ok"]
        and result["meta_behaviour_ok"]
    )
    return result


def validate_intent_alignment(obj: dict[str, Any] | None, expected_intent: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "expected_intent": expected_intent,
        "meta_command_present": False,
        "meta_command": None,
        "meta_command_allowed": None,
        "intent_alignment_valid": False,
    }
    if obj is None:
        return result

    meta = obj.get("meta_behaviour")
    if not isinstance(meta, dict):
        return result

    command = meta.get("command")
    if not isinstance(command, str):
        return result

    allowed_commands = INTENT_TO_ALLOWED_META_COMMANDS.get(expected_intent, COMMAND_CHOICES)
    result["meta_command_present"] = True
    result["meta_command"] = command
    result["allowed_meta_commands"] = sorted(allowed_commands)
    result["meta_command_allowed"] = command in allowed_commands
    result["intent_alignment_valid"] = bool(result["meta_command_allowed"])
    return result


def validate_text_planning_direction(denorm_traj: np.ndarray | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "forward_progress_valid": False,
        "lateral_deviation_valid": False,
        "heading_valid": False,
        "text_planning_valid": False,
    }
    if denorm_traj is None or denorm_traj.shape != (8, 3) or not np.isfinite(denorm_traj).all():
        return result

    final_x = float(denorm_traj[-1, 0])
    final_y = float(denorm_traj[-1, 1])
    final_yaw = float(denorm_traj[-1, 2])

    result["final_x_m"] = round(final_x, 6)
    result["final_y_m"] = round(final_y, 6)
    result["final_yaw_rad"] = round(final_yaw, 6)
    result["forward_progress_valid"] = final_x > 0.5
    result["lateral_deviation_valid"] = abs(final_y) <= 4.0
    result["heading_valid"] = abs(final_yaw) <= 1.0
    result["text_planning_valid"] = bool(
        result["forward_progress_valid"] and result["lateral_deviation_valid"] and result["heading_valid"]
    )
    return result


def validate_denormalized_trajectory(
    denorm_traj: np.ndarray | None,
    reference_future: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "trajectory_parse_ok": denorm_traj is not None,
        "trajectory_shape_ok": False,
        "trajectory_finite_ok": False,
        "sanity_valid": False,
        "reference_valid": None,
        "trajectory_valid": False,
        "reference_steps": int(reference_future.shape[0]),
    }
    if denorm_traj is None:
        return metrics

    metrics["trajectory_shape_ok"] = denorm_traj.shape == (8, 3)
    metrics["trajectory_finite_ok"] = bool(np.isfinite(denorm_traj).all())
    if not metrics["trajectory_shape_ok"] or not metrics["trajectory_finite_ok"]:
        return metrics

    xy = denorm_traj[:, :2]
    yaw = denorm_traj[:, 2]
    deltas_xy = np.diff(np.vstack([np.zeros((1, 2), dtype=np.float64), xy]), axis=0)
    step_distances = np.linalg.norm(deltas_xy, axis=1)
    yaw_steps = np.asarray([abs(wrap_angle(float(v))) for v in np.diff(np.concatenate([[0.0], yaw]))], dtype=np.float64)

    final_displacement = float(np.linalg.norm(xy[-1]))
    max_step_distance = float(np.max(step_distances))
    max_abs_yaw = float(np.max(np.abs(yaw)))
    max_yaw_step = float(np.max(yaw_steps))
    metrics.update(
        {
            "final_displacement_m": round(final_displacement, 6),
            "max_step_distance_m": round(max_step_distance, 6),
            "max_abs_yaw_rad": round(max_abs_yaw, 6),
            "max_yaw_step_rad": round(max_yaw_step, 6),
        }
    )

    metrics["sanity_valid"] = bool(
        max_step_distance <= args.max_step_distance_m
        and final_displacement <= args.max_final_displacement_m
        and max_abs_yaw <= args.max_abs_yaw_rad
        and max_yaw_step <= args.max_yaw_step_rad
    )

    if reference_future.shape[0] > 0:
        compare_steps = min(denorm_traj.shape[0], reference_future.shape[0])
        xy_errors = np.linalg.norm(denorm_traj[:compare_steps, :2] - reference_future[:compare_steps, :2], axis=1)
        yaw_errors = np.asarray(
            [abs(wrap_angle(float(denorm_traj[i, 2] - reference_future[i, 2]))) for i in range(compare_steps)],
            dtype=np.float64,
        )
        metrics.update(
            {
                "compare_steps": int(compare_steps),
                "ade_m": round(float(np.mean(xy_errors)), 6),
                "fde_m": round(float(xy_errors[-1]), 6),
                "mean_yaw_error_rad": round(float(np.mean(yaw_errors)), 6),
                "first_step_error_m": round(float(xy_errors[0]), 6),
            }
        )
        metrics["reference_valid"] = bool(metrics["first_step_error_m"] <= args.max_first_step_error_m)
    else:
        metrics.update(
            {
                "compare_steps": 0,
                "ade_m": None,
                "fde_m": None,
                "mean_yaw_error_rad": None,
                "first_step_error_m": None,
            }
        )

    metrics["trajectory_valid"] = bool(
        metrics["sanity_valid"] and (metrics["reference_valid"] if metrics["reference_valid"] is not None else True)
    )
    return metrics


def run_text_planning_control(
    session: requests.Session,
    args: argparse.Namespace,
    means: np.ndarray,
    stds: np.ndarray,
) -> dict[str, Any]:
    messages = build_text_planning_control_messages()
    try:
        response_json, latency_sec = post_chat_completion(
            session=session,
            base_url=args.base_url,
            model_name=args.model_name,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
        )
        raw_text = extract_message_content(response_json)
        obj = extract_first_json_object(raw_text)
        contract = validate_output_contract(obj, raw_text)
        intent_alignment = validate_intent_alignment(obj, "go straight")
        trajectory = extract_trajectory(obj, raw_text)
        denorm = denormalize_trajectory(trajectory, means, stds) if trajectory is not None else None
        traj_validation = validate_denormalized_trajectory(denorm, np.zeros((0, 3), dtype=np.float64), args)
        direction_validation = validate_text_planning_direction(denorm)
        return {
            "request_ok": True,
            "latency_sec": round(latency_sec, 6),
            "raw_text_preview": raw_text[:1000],
            "contract": contract,
            "intent_alignment": intent_alignment,
            "trajectory": traj_validation,
            "direction": direction_validation,
            "overall_valid": bool(
                contract["contract_valid"]
                and intent_alignment["intent_alignment_valid"]
                and traj_validation["trajectory_valid"]
                and direction_validation["text_planning_valid"]
            ),
        }
    except Exception as exc:
        return {
            "request_ok": False,
            "error": str(exc),
            "contract": {"contract_valid": False},
            "intent_alignment": {"intent_alignment_valid": False},
            "trajectory": {"trajectory_valid": False},
            "direction": {"text_planning_valid": False},
            "overall_valid": False,
        }


def run_scene_validation(
    session: requests.Session,
    scene_path: Path,
    sensor_blobs_root: Path,
    args: argparse.Namespace,
    means: np.ndarray,
    stds: np.ndarray,
) -> dict[str, Any]:
    scene_dict_list = load_scene_pickle(scene_path)
    image_path = scene_image_path(scene_dict_list, sensor_blobs_root)
    prompt = build_planning_prompt(scene_dict_list)
    messages, image_meta = build_openai_messages(image_path, prompt, args)
    reference_future = extract_reference_future(scene_dict_list)

    expected_intent = current_command_str(scene_dict_list)
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

    try:
        response_json, latency_sec = post_chat_completion(
            session=session,
            base_url=args.base_url,
            model_name=args.model_name,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout_sec,
        )
        raw_text = extract_message_content(response_json)
        obj = extract_first_json_object(raw_text)
        contract = validate_output_contract(obj, raw_text)
        intent_alignment = validate_intent_alignment(obj, expected_intent)
        norm_traj = extract_trajectory(obj, raw_text)
        denorm_traj = denormalize_trajectory(norm_traj, means, stds) if norm_traj is not None else None
        trajectory = validate_denormalized_trajectory(denorm_traj, reference_future, args)
        overall_valid = bool(
            contract["contract_valid"]
            and intent_alignment["intent_alignment_valid"]
            and trajectory["trajectory_valid"]
        )
        return {
            **base_report,
            "request_ok": True,
            "latency_sec": round(latency_sec, 6),
            "raw_text_preview": raw_text[:2000],
            "contract": contract,
            "intent_alignment": intent_alignment,
            "trajectory": trajectory,
            "overall_valid": overall_valid,
        }
    except Exception as exc:
        return {
            **base_report,
            "request_ok": False,
            "error": str(exc),
            "contract": {"contract_valid": False},
            "intent_alignment": {"intent_alignment_valid": False},
            "trajectory": {"trajectory_valid": False},
            "overall_valid": False,
            "raw_text_preview": "",
            "latency_sec": None,
        }
