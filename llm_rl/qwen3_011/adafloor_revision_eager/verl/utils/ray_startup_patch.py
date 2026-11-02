# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Any


def _env_enabled(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() not in ("0", "false", "no", "off")


def _maybe_inject_dashboard_agent_flags(command: list[str]) -> list[str]:
    patched_command = list(command)
    prefix = "--dashboard_agent_command="
    for i, arg in enumerate(patched_command):
        if not arg.startswith(prefix):
            continue
        dashboard_command = arg
        if "--disable-metrics-collection" not in dashboard_command:
            dashboard_command = f"{dashboard_command} --disable-metrics-collection"
        if _env_enabled("VERL_RAY_DASHBOARD_AGENT_MINIMAL", "0") and " --minimal" not in dashboard_command:
            dashboard_command = f"{dashboard_command} --minimal"
        patched_command[i] = dashboard_command
        break
    return patched_command


def apply_ray_startup_patches(ray_init_kwargs: dict[str, Any]) -> dict[str, Any]:
    patched_kwargs = dict(ray_init_kwargs)

    disable_head_dashboard = _env_enabled("VERL_RAY_DISABLE_HEAD_DASHBOARD", "0")
    if disable_head_dashboard and "include_dashboard" not in patched_kwargs:
        patched_kwargs["include_dashboard"] = False

    disable_dashboard_metrics = _env_enabled("VERL_RAY_DISABLE_DASHBOARD_METRICS", "1")
    if not disable_dashboard_metrics:
        return patched_kwargs

    # Ray 2.x registers an OpenTelemetry MetricsService even when the dashboard
    # agent's periodic metrics collection is disabled. Disable it before ray.init
    # starts the agent, otherwise failed exports can flood dashboard_agent.log and
    # interfere with large worker-group startup.
    os.environ.setdefault("RAY_enable_open_telemetry", "0")

    import ray._private.ray_constants as ray_constants
    import ray._private.node as ray_node
    import ray._private.services as ray_services

    if getattr(ray_services, "_verl_dashboard_metrics_patch_applied", False):
        return patched_kwargs

    original_start_ray_process = ray_services.start_ray_process

    def patched_start_ray_process(command, process_type, fate_share, **kwargs):
        if process_type == ray_constants.PROCESS_TYPE_RAYLET:
            command = _maybe_inject_dashboard_agent_flags(list(command))
        return original_start_ray_process(command, process_type, fate_share, **kwargs)

    ray_services.start_ray_process = patched_start_ray_process
    ray_services._verl_dashboard_metrics_patch_applied = True

    if (_env_enabled("VERL_RAY_DISABLE_API_SERVER", "1")
            and not getattr(ray_node.Node, "_verl_api_server_patch_applied", False)):
        original_start_api_server = ray_node.Node.start_api_server

        def patched_start_api_server(self, *, include_dashboard, raise_on_failure):
            self._webui_url = ""
            return None

        ray_node.Node._verl_original_start_api_server = original_start_api_server
        ray_node.Node.start_api_server = patched_start_api_server
        ray_node.Node._verl_api_server_patch_applied = True

    return patched_kwargs
