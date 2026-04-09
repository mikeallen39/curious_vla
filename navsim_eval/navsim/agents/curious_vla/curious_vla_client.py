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

import warnings
import torch
import torch.nn as nn
import re
from openai import OpenAI
import base64
from .base_module import BaseModule
from .convert_chat_template import convert_to_openai_format
import json
import math
import itertools
import random
import os
import threading
import logging
import time
import httpx

http_client = httpx.Client(trust_env=False)

logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_trajectory_string(text: str) -> torch.Tensor:
    """Parse [x, y] trajectory from VLA model output text."""
    try:
        planning_match = re.search(r'<PLANNING>(.*?)</PLANNING>', text, re.DOTALL)
        if planning_match:
            text_to_parse = planning_match.group(1)
        else:
            text_to_parse = text

        matches_2d = re.findall(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', text_to_parse)
        if matches_2d and len(matches_2d) >= 8:
            return torch.tensor([(float(x), float(y)) for x, y in matches_2d[:8]])

        return None

    except Exception as e:
        logger.warning(f"An unexpected exception occurred during robust trajectory parsing: {e}")
        return None

def parse_trajectory_string_with_yaw(text: str) -> torch.Tensor:
    """Parser for models that output [x, y, yaw] format."""
    try:
        text_to_parse = text
        matches_3d = re.findall(r'[\[\(]\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*[\]\)]', text_to_parse)
        if matches_3d and len(matches_3d) >= 8:
            points = [(float(x), float(y), float(yaw)) for x, y, yaw in matches_3d[:8]]
            return torch.tensor(points)

        return None

    except Exception as e:
        logger.warning(f"An unexpected exception occurred during yaw trajectory parsing: {e}")
        return None

def parse_trajectory_string_after_tag(text: str, tag='"future_trajectory"') -> torch.Tensor:
    """
    Robustly parse [x, y, yaw] data after a given tag field.
    Does not rely on <answer> tags; only depends on field name order and bracket structure.
    """
    try:
        parts = text.split(tag)

        if len(parts) < 2:
            return None

        content_after_key = parts[-1]
        list_match = re.search(r'\[(.*?)\]', content_after_key, re.DOTALL)

        if not list_match:
            return None

        target_content = list_match.group(1)
        coord_pattern = r'\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\)'
        matches_3d = re.findall(coord_pattern, target_content)

        if matches_3d and len(matches_3d) >= 8:
            points = [(float(x), float(y), float(yaw)) for x, y, yaw in matches_3d[:8]]
            return torch.tensor(points)

        return None

    except Exception as e:
        logger.warning(f"An unexpected exception occurred during yaw trajectory parsing: {e}")
        return None

def parse_waypoints_from_action_output(text: str) -> torch.Tensor:
    """
    Parser for mixed output containing Waypoints and Action Tokens.
    Precisely isolates the waypoint string and parses it.
    """
    try:
        planning_match = re.search(r'<PLANNING>(.*?)</PLANNING>', text, re.DOTALL)
        if not planning_match:
            return None
        text_to_parse = planning_match.group(1)

        start_keyword = "The Waypoints output is formatted as [x, y]:"
        start_index = text_to_parse.find(start_keyword)
        if start_index == -1:
            return None

        waypoint_substring = text_to_parse[start_index + len(start_keyword):]

        end_keyword = "The Actions Token output is:"
        end_index = waypoint_substring.find(end_keyword)
        if end_index != -1:
            waypoint_substring = waypoint_substring[:end_index]

        matches_2d = re.findall(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', waypoint_substring)
        if matches_2d and len(matches_2d) >= 8:
            points = [(float(x), float(y)) for x, y in matches_2d[:8]]
            return torch.tensor(points)

        return None

    except Exception as e:
        logger.warning(f"An unexpected exception occurred during action token output parsing: {e}")
        return None

def decode_indices_to_trajectory(indices: list, codebook) -> torch.Tensor:
    """
    Decode action token index sequence into an absolute coordinate trajectory.

    Returns:
        A tensor of shape [N, 3] representing (x, y, yaw) trajectory.
    """
    trajectory = []
    current_pose = torch.tensor([0.0, 0.0, 0.0])  # [x, y, theta]

    for index in indices:
        if index >= len(codebook):
            continue

        local_delta_x, local_delta_y, delta_theta = codebook[index]

        theta = current_pose[2]
        world_delta_x = local_delta_x * math.cos(theta) - local_delta_y * math.sin(theta)
        world_delta_y = local_delta_x * math.sin(theta) + local_delta_y * math.cos(theta)

        new_x = current_pose[0] + world_delta_x
        new_y = current_pose[1] + world_delta_y
        new_theta = normalize_angle(current_pose[2] + delta_theta)

        current_pose = torch.tensor([new_x, new_y, new_theta])
        trajectory.append([new_x.item(), new_y.item(), new_theta.item()])

    return torch.tensor(trajectory)

def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def parse_action_tokens(text: str, codebook) -> list:
    """Extract Action Tokens list from model output."""
    try:
        text_to_parse = text
        action_indices_str = re.findall(r"<action_(\d+)>", text_to_parse)
        if not action_indices_str:
            return None
        action_indices = [int(idx) for idx in action_indices_str]
        decoded_trajectory = decode_indices_to_trajectory(action_indices, codebook)
        if len(decoded_trajectory) > 8:
            return decoded_trajectory[:8]
        return decoded_trajectory

    except Exception as e:
        logger.warning(f"An unexpected exception occurred during action token parsing: {e}")
        return None


class CuriousVLAClient(BaseModule):
    def __init__(self,
                 model_name_or_path: str,
                 codebook_path: str = None,
                 server_urls=None,
                 api_key: str = "0",
                 max_tokens: int = 4096,
                 default_temperature: float = 0.0,
                 **kwargs):
        super().__init__()

        n_instances = 8
        if server_urls is None:
            server_urls = [f"http://0.0.0.0:{8192 + i}/v1" for i in range(n_instances)]

        # Detect available servers and create persistent OpenAI clients
        self.clients = []
        self.rnd = random.Random(int(threading.get_ident()) * 100000 + int(os.getpid()))
        for url in server_urls:
            try:
                resp = http_client.get(f"{url}/models", timeout=2)
                if resp.status_code == 200:
                    self.clients.append(OpenAI(api_key=api_key, base_url=url, http_client=http_client))
            except Exception:
                pass

        if not self.clients:
            raise RuntimeError("No available server found. Please check your server URLs.")
        logger.info(f"CuriousVLAClient initialized with {len(self.clients)} server(s).")

        self.model_name_or_path = model_name_or_path
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.default_temperature = default_temperature
        self.codebook_path = codebook_path
        if self.codebook_path:
            with open(self.codebook_path, 'r') as f:
                json_data = json.load(f)
                self.codebook = torch.tensor(json_data["codebook"])
        else:
            self.codebook = torch.tensor([])

    # TODO: refactor parser selection to use string matching instead of multiple booleans
    def forward(self, messages,
                use_tag=None,
                use_yaw_parser=False,
                use_action_parser=False,
                parse_action_traj=False,
                temperature=None,
                **kwargs):
        llm_messages = convert_to_openai_format(messages, dataset_dir=None)
        request_temperature = self.default_temperature if temperature is None else temperature

        last_exc = None

        for attempt in range(3):
            client = self.rnd.choice(self.clients)
            try:
                result = client.chat.completions.create(
                    messages=llm_messages,
                    model=self.model_name_or_path,
                    max_tokens=self.max_tokens,
                    temperature=request_temperature,
                )

                output_text = result.choices[0].message.content

                if use_action_parser:
                    parsed_trajectory = parse_waypoints_from_action_output(output_text)
                elif use_tag is not None:
                    parsed_trajectory = parse_trajectory_string_after_tag(output_text, tag=use_tag)
                elif use_yaw_parser:
                    parsed_trajectory = parse_trajectory_string_with_yaw(output_text)
                elif parse_action_traj:
                    parsed_trajectory = parse_action_tokens(output_text, self.codebook)
                else:
                    parsed_trajectory = parse_trajectory_string(output_text)

                if parsed_trajectory is None:
                    raise ValueError(f"Failed to parse a valid trajectory from model output: {output_text}")

                return parsed_trajectory, output_text

            except Exception as e:
                last_exc = e
                logger.warning(
                    f"CuriousVLAClient.forward failed (attempt {attempt + 1}/3) "
                    f"base_url={client.base_url}: {type(e).__name__}: {e}"
                )
                if attempt < 2:
                    time.sleep(0.2 * (2 ** attempt) + self.rnd.random() * 0.1)
                    continue
                raise last_exc
