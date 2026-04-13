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

from dataclasses import dataclass

@dataclass
class CuriousVlaConfig:
    """
    Configuration for CuriousVLAAgent.
    """
    # --- Model and API connection ---
    model_name_or_path: str
    codebook_path: str = None
    api_base_url: str = "http://0.0.0.0:8192/v1"
    api_key: str = "0"

    # --- API call parameters ---
    max_tokens: int = 3072
    temperature: float = 0.0

    # If False, missing front-view images are treated as a hard data error.
    allow_missing_front_camera_fallback: bool = False

    # Path to save detailed logs. If None or empty, no logs are saved.
    log_path: str = None
