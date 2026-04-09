# Copyright 2026 Curious-VLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import warnings
from typing import Dict
import json

import numpy as np
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from PIL import Image

from .curious_vla_config import CuriousVlaConfig
from .curious_vla_client import CuriousVLAClient
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory, Cameras, Camera, EgoStatus
from navsim.common.dataclasses import Scene

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

QWEN_SYSTEM_MESSAGE = "You are an expert driver."

stats_path = os.environ.get('STATS_PATH', '../stats/trajectory_stats_train.json')
global means, stds
# navtrain 103k
with open(stats_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    means = np.array(data['mean'])
    stds = np.array(data['std'])

def denormalize(poses):
    result = np.array(poses) * stds + means
    return result

def format_number(n, decimal_places=2):
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)

class NavsimCoTQwenAgent(AbstractAgent):
    """
    Agent for evaluating the Navsim SFT baseline.
    - Input: 1.5s history prompt matching training data format
    - Output: [x, y, yaw] trajectory
    """

    def __init__(
        self,
        config: CuriousVlaConfig,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4.0, interval_length=0.5),
        cam_type: str = 'single',     # 'single', 'multi_view', 'cont'
    ):
        super().__init__(trajectory_sampling)
        self._config = config
        self._client = None
        self._temp_dir = tempfile.mkdtemp()
        self._step = -1

        self.cam_type = cam_type

        # Tell Navsim we need the full scene object
        self.requires_scene = True

        # Read log save path from config
        self.log_path = self._config.log_path
        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)

    def name(self) -> str:
        return "Navsim_Qwen_Agent"

    def initialize(self) -> None:
        self._client = CuriousVLAClient(
            model_name_or_path=self._config.model_name_or_path,
            codebook_path=self._config.codebook_path,
            server_urls=[self._config.api_base_url] if self._config.api_base_url else None,
            api_key=self._config.api_key,
            max_tokens=self._config.max_tokens,
            default_temperature=self._config.temperature,
        )

    def get_sensor_config(self) -> SensorConfig:
        """Dynamically request required sensor data based on cam_type."""
        if self.cam_type == 'multi_view':
            return SensorConfig.build_all_sensors(include=[3])
        elif self.cam_type == 'cont':
            return SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
                                cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)
        else: # 'single' or default
            return SensorConfig(cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
                                cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)

    def _build_prompt_messages(self, agent_input: AgentInput, image_paths: Dict[str, str]) -> Dict:
        """
        :param agent_input: Input from the simulator.
        :param image_paths: A dict of {camera_name: path}.
        :return: A dict conforming to the Client input format.
        """
        ego_statuses = agent_input.ego_statuses
        status_lines = []
        size = len(ego_statuses)
        for i in range(size):
            s = ego_statuses[i]
            status_lines.append(
                f'   - t-{size - i - 1}: ({format_number(s.ego_pose[0])}, {format_number(s.ego_pose[1])}, {format_number(s.ego_pose[2])})'
            )

        status_lines.append(f'   - t-{0}: ({format_number(0)}, {format_number(0)}, {format_number(0)})')

        high_command_one_hot = ego_statuses[-1].driving_command
        nav_commands = ['turn left', 'go straight', 'turn right', 'unknown']
        command_str = [nav_commands[i] for i, v in enumerate(high_command_one_hot) if v == 1]
        command_str = command_str[0] if command_str else "unknown"
        history_trajectory = ", ".join(status_lines)
        Q1 = f"""Suppose you are driving. Let's complete the following tasks step by step.
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
        Q2 = """Task 2: Natural Language Explanation
Compose a concise natural-language description of the optimal future 5-second trajectory for the ego vehicle that the expert driver (you) plans and explain why the expert driver plans to execute this trajectory.
   - Mention only the classes you marked "yes" in the previous task.
   - Describe how each of those critical objects or conditions influences the optimal trajectory.
   - Do not invent objects or conditions not present in the input.
"""
        Q3 = """Task 3: Meta-Behaviour Selection
Assign exactly one category from each list. Choose the label that best summarises the overall behaviour of the optimal future trajectory:
   - speed ∈ { keep, accelerate, decelerate }
   - command ∈ { straight, yield, left_turn, right_turn, lane_follow, lane_change_left, lane_change_right, reverse }
   Choose the label that best summarises the overall behaviour of the optimal future trajectory.
   - If none fits, use 'other', but do this sparingly.
"""
        Q4 = """Task 4: Future Trajectory Prediction
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

        final_content = f'{Q1}\n{Q2}\n{Q3}\n{Q4}'
        return {
            "images": list(image_paths.values()),
            "messages": [
                {"role": "system", "content": QWEN_SYSTEM_MESSAGE},
                {"role": "user", "content": final_content}
            ]
        }

    def _fallback_to_constant_velocity(self, agent_input: AgentInput) -> Trajectory:
        current_velocity = agent_input.ego_statuses[-1].ego_velocity[0].item()
        num_poses = self._trajectory_sampling.num_poses
        interval = self._trajectory_sampling.interval_length
        poses = np.zeros((num_poses, 3), dtype=np.float32)
        for i in range(num_poses):
            poses[i, 0] = current_velocity * (i + 1) * interval
        return Trajectory(poses, self._trajectory_sampling)

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        self.eval()
        self._step += 1

        token = scene.scene_metadata.initial_token
        log_file = os.path.join(self.log_path, "detailed_logs.jsonl") if self.log_path else None
        current_cams = agent_input.cameras[-1] if agent_input.cameras else None
        front_camera = current_cams.cam_f0 if current_cams is not None else None
        image_np = front_camera.image if front_camera is not None else None
        if image_np is None:
            warnings.warn(
                f"Step {self._step}: Front camera image is missing for token {token}. Falling back."
            )
            return self._fallback_to_constant_velocity(agent_input)

        image_paths: Dict[str, str] = {}
        cameras_to_process = []
        if self.cam_type == 'single':
            cameras_to_process.append(('cam_f0', front_camera))
        elif self.cam_type == 'multi_view':
            cameras_to_process.extend([
                ('cam_f0', current_cams.cam_f0), ('cam_l0', current_cams.cam_l0),
                ('cam_r0', current_cams.cam_r0), ('cam_l2', current_cams.cam_l2),
                ('cam_r2', current_cams.cam_r2), ('cam_b0', current_cams.cam_b0)
            ])
        elif self.cam_type == 'cont':
            for i in range(len(agent_input.cameras)):
                 cameras_to_process.append((f'cam_f0_t{i}', agent_input.cameras[i].cam_f0))

        for name, cam_data in cameras_to_process:
            if cam_data and cam_data.image is not None:
                temp_image_path = os.path.join(self._temp_dir, f"step_{self._step}_{name}.jpg")
                try:
                    Image.fromarray(cam_data.image.astype('uint8'), 'RGB').save(temp_image_path)
                    image_paths[name] = temp_image_path
                except Exception as e:
                    warnings.warn(f"Step {self._step}: Failed to save image {name}: {e}.")

        if not image_paths:
            logger.warning(f"Step {self._step}: No valid image data found. Falling back.")
            return self._fallback_to_constant_velocity(agent_input)

        messages = self._build_prompt_messages(agent_input, image_paths)


        try:
            with torch.no_grad():
                parsed_trajectory, raw_server_output = self._client.forward(messages, use_yaw_parser=True)

            if parsed_trajectory is None:
                raise ValueError(f"Failed to parse a valid trajectory from model output: {raw_server_output}")

            if not isinstance(parsed_trajectory, torch.Tensor) or parsed_trajectory.shape != (8, 3):
                raise TypeError(f"Client returned unexpected shape or type: {parsed_trajectory.shape}")

        except Exception as e:
            logger.warning(f"Step {self._step}: VLA client failed: {e}. Falling back.")
            return self._fallback_to_constant_velocity(agent_input)

        parsed_trajectory = parsed_trajectory.cpu().numpy()
        parsed_trajectory = denormalize(parsed_trajectory)
        final_trajectory = Trajectory(parsed_trajectory, self._trajectory_sampling)

        if log_file:
            record = {
                "token": token,
                "server_output": raw_server_output,
                "trajectory_se2": final_trajectory.poses.tolist(),
            }
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, separators=(',', ':'))
                    f.write("\n")
            except Exception as e:
                print(f"Failed to write JSONL log: {e}")

        return final_trajectory

    def __del__(self):
        temp_dir = getattr(self, "_temp_dir", None)
        if isinstance(temp_dir, str) and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
