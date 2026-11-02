# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for selecting and loading models."""
import contextlib
import inspect
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from typing_extensions import assert_never

from vllm.attention import Attention
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import QKVCrossParallelLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.models.adapters import (
    as_embedding_model, as_reward_model, as_seq_cls_model,
    try_create_mm_pooling_model_cls)
from vllm.model_executor.models.interfaces import (SupportsQuant,
                                                   supports_multimodal)
from vllm.utils import is_pin_memory_available

logger = init_logger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in (
        "1", "true", "yes", "on")


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return max(minimum, default)


def _mode1_diag_layers_enabled(layer_idx: int) -> bool:
    raw = os.getenv("VLLM_ASCEND_MODE1_PROCESS_WEIGHT_DIAG_LAYERS", "0")
    if raw.strip().lower() in ("all", "*"):
        return True
    try:
        return int(layer_idx) in {
            int(item.strip())
            for item in raw.split(",") if item.strip()
        }
    except (TypeError, ValueError):
        return int(layer_idx) == 0


def _tensor_storage_nbytes(tensor: object) -> int:
    if not torch.is_tensor(tensor):
        return 0
    try:
        return int(tensor.untyped_storage().nbytes())
    except Exception:
        try:
            return int(tensor.storage().nbytes())
        except Exception:
            return 0


def _tensor_row_fingerprint(tensor: object, row: int,
                            sample_values: int) -> dict[str, object]:
    if not torch.is_tensor(tensor):
        return {"ok": False, "why": "not_tensor"}
    if getattr(tensor, "ndim", 0) < 1:
        return {"ok": False, "why": "scalar"}
    rows = int(tensor.shape[0])
    if row < 0 or row >= rows:
        return {"ok": False, "why": f"row_oob:{row}/{rows}"}
    try:
        values = tensor[int(row)].reshape(-1)[:sample_values].detach().to(
            dtype=torch.float32, device="cpu")
        if values.numel() == 0:
            return {"ok": False, "why": "empty"}
        head = [
            round(float(v), 6)
            for v in values[:min(4, int(values.numel()))].tolist()
        ]
        return {
            "ok": True,
            "sum": round(float(values.sum().item()), 6),
            "abs_mean": round(float(values.abs().mean().item()), 6),
            "head": head,
        }
    except Exception as exc:
        return {"ok": False, "why": repr(exc)}


def _lossless_mode1_process_weight_diag(module: nn.Module,
                                        stage: str) -> None:
    if not _env_flag("VLLM_ASCEND_MODE1_PROCESS_WEIGHT_DIAG", "0"):
        return
    if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
        return
    layer_idx = int(getattr(module, "layer_idx", -1))
    if not _mode1_diag_layers_enabled(layer_idx):
        return
    sample_values = _env_int("VLLM_ASCEND_MODE1_PROCESS_WEIGHT_DIAG_VALUES",
                             8,
                             minimum=1)
    expert_map = getattr(module, "expert_map", None)
    loaded_map = getattr(module, "loaded_expert_map", None)

    def _map_head(tensor: object) -> list[int] | str:
        if not torch.is_tensor(tensor):
            return "none"
        try:
            return [int(v) for v in tensor.detach().cpu().tolist()[:16]]
        except Exception:
            return "copy_failed"

    active_slot = 0
    loaded_slot = 0
    try:
        if torch.is_tensor(expert_map):
            active_values = [
                int(v) for v in expert_map.detach().cpu().tolist()
            ]
            active_slot = next((slot for slot in active_values if slot >= 0),
                               0)
        if torch.is_tensor(loaded_map):
            loaded_values = [
                int(v) for v in loaded_map.detach().cpu().tolist()
            ]
            loaded_slot = next((slot for slot in loaded_values if slot >= 0),
                               0)
    except Exception:
        active_slot = 0
        loaded_slot = 0

    w13 = getattr(module, "w13_weight", None)
    w2 = getattr(module, "w2_weight", None)
    runtime_w13 = getattr(module, "runtime_w13_weight", None)
    runtime_w2 = getattr(module, "runtime_w2_weight", None)
    logger.warning(
        "Mode1 process-weights diag: stage=%s layer=%s target_floor_env=%s "
        "active_local=%s local_num=%s moe_local=%s loaded_local=%s "
        "loaded_capacity=%s w13_shape=%s w2_shape=%s runtime_w13_shape=%s "
        "runtime_w2_shape=%s w13_storage=%s w2_storage=%s expert_map_head=%s "
        "loaded_map_head=%s sample_active_slot=%s sample_loaded_slot=%s "
        "w13_active=%s w13_loaded=%s w13_runtime=%s w2_active=%s "
        "w2_loaded=%s w2_runtime=%s",
        stage,
        layer_idx,
        os.getenv("VLLM_ASCEND_MODE1_WEIGHT_RELOAD_TARGET_FLOOR",
                  os.getenv("VLLM_ASCEND_MODE1_PARITY_CURRENT_FLOOR")),
        int(getattr(module, "active_local_num_experts", -1)),
        int(getattr(module, "local_num_experts", -1)),
        int(getattr(getattr(module, "moe_config", None),
                    "num_local_experts", -1)),
        int(getattr(module, "loaded_local_num_experts", -1)),
        int(getattr(module, "loaded_weight_capacity", -1)),
        tuple(getattr(w13, "shape", ())),
        tuple(getattr(w2, "shape", ())),
        tuple(getattr(runtime_w13, "shape", ())),
        tuple(getattr(runtime_w2, "shape", ())),
        _tensor_storage_nbytes(w13),
        _tensor_storage_nbytes(w2),
        _map_head(expert_map),
        _map_head(loaded_map),
        active_slot,
        loaded_slot,
        _tensor_row_fingerprint(w13, active_slot, sample_values),
        _tensor_row_fingerprint(w13, loaded_slot, sample_values),
        _tensor_row_fingerprint(runtime_w13, active_slot, sample_values),
        _tensor_row_fingerprint(w2, active_slot, sample_values),
        _tensor_row_fingerprint(w2, loaded_slot, sample_values),
        _tensor_row_fingerprint(runtime_w2, active_slot, sample_values),
    )


def _lossless_stdout_module_snapshot(module: nn.Module, stage: str) -> None:
    if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
        return
    try:
        w13 = module.w13_weight
        rows = min(4, int(w13.shape[0])) if getattr(w13, "ndim", 0) >= 1 else 0
        head = []
        for i in range(rows):
            head.append((i, round(float(w13[i].float().abs().mean().item()), 6)))
        pass  # debug log removed
    except Exception as exc:
        pass  # debug log removed


def _is_lossless_ascend_fused_moe(module: nn.Module,
                                  quant_method: QuantizeMethodBase) -> bool:
    return (hasattr(module, "w13_weight") and hasattr(module, "w2_weight")
            and type(quant_method).__name__ == "AscendUnquantizedFusedMoEMethod"
            and type(quant_method).__module__.startswith("vllm_ascend"))


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: Optional[type[nn.Module]] = None,
    model_config: Optional[ModelConfig] = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    if model_config is None:
        model_config = vllm_config.model_config
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config,
                                     check_compile=True,
                                     prefix=prefix):
            return model_class(vllm_config=vllm_config, prefix=prefix)

    msg = ("vLLM model class should accept `vllm_config` and `prefix` as "
           "input arguments. Possibly you have an old-style model class"
           " registered from out of tree and it is used for new vLLM version. "
           "Check https://docs.vllm.ai/en/latest/design/arch_overview.html "
           "for the design and update the model class accordingly.")
    warnings.warn(msg, DeprecationWarning, stacklevel=2)

    logger.warning(
        "Trying to guess the arguments for old-style model class %s",
        model_class,
    )
    # try to be compatible with old-style model class
    kwargs = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    if "cache_config" in all_params:
        kwargs["cache_config"] = vllm_config.cache_config
    if "quant_config" in all_params:
        kwargs["quant_config"] = vllm_config.quant_config
    if "lora_config" in all_params:
        kwargs["lora_config"] = vllm_config.lora_config
    if "scheduler_config" in all_params:
        kwargs["scheduler_config"] = vllm_config.scheduler_config
    with set_current_vllm_config(vllm_config,
                                 check_compile=True,
                                 prefix=prefix):
        return model_class(**kwargs)


def process_weights_after_loading(model: nn.Module, model_config: ModelConfig,
                                  target_device: torch.device) -> None:
    for _, module in model.named_modules():
        if isinstance(module, QKVCrossParallelLinear):
            # NOTE(Isotr0py): special case for cross QKV layer because
            # q and kv proj aren't registered as submodules intentionally
            module.process_weights_after_loading()
            continue
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            if _is_lossless_ascend_fused_moe(module, quant_method):
                _lossless_stdout_module_snapshot(module,
                                                "before_direct_process_weights_after_loading")
                _lossless_mode1_process_weight_diag(
                    module, "before_direct_process_weights_after_loading")
                quant_method.process_weights_after_loading(module)
                _lossless_stdout_module_snapshot(module,
                                                "after_direct_process_weights_after_loading")
                _lossless_mode1_process_weight_diag(
                    module, "after_direct_process_weights_after_loading")
                continue
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            _lossless_stdout_module_snapshot(module,
                                            "before_device_loading_context")
            with device_loading_context(module, target_device):
                _lossless_stdout_module_snapshot(
                    module, "inside_device_loading_context_before_process")
                quant_method.process_weights_after_loading(module)
                _lossless_stdout_module_snapshot(
                    module, "inside_device_loading_context_after_process")
            _lossless_stdout_module_snapshot(module,
                                            "after_device_loading_context")

    # Currently only used by MLA.
    # NOTE: This intentionally happens after other modules so we can easily
    # decompress the weights for MLA.
    for _, module in model.named_modules():
        if isinstance(module, Attention) and \
            hasattr(module, "process_weights_after_loading"):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            module.process_weights_after_loading(model_config.dtype)


@contextmanager
def device_loading_context(module: torch.nn.Module,
                           target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: dict[str, torch.device] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                if original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched


_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
"""Caches the outputs of `_get_model_architecture`."""


def _get_model_architecture(
        model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])

    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )

    if arch == model_config._get_transformers_backend_cls():
        assert model_config.model_impl != "vllm"
        if model_config.model_impl == "auto":
            logger.warning_once(
                "%s has no vLLM implementation, falling back to Transformers "
                "implementation. Some features may not be supported and "
                "performance may not be optimal.", arch)

    convert_type = model_config.convert_type
    if convert_type != "none" and supports_multimodal(model_cls):
        logger.debug_once("Detected conversion of Multi Modal model.")
        converted = try_create_mm_pooling_model_cls(model_cls)
        if converted is not None:
            logger.debug_once("Creating wrapper class to forward pooler.")
            return converted, arch
        else:
            logger.debug_once("Attempting direct conversion.")

    if convert_type == "none":
        pass
    elif convert_type == "embed":
        logger.debug_once("Converting to embedding model.")
        model_cls = as_embedding_model(model_cls)
    elif convert_type == "classify":
        logger.debug_once("Converting to sequence classification model.")
        model_cls = as_seq_cls_model(model_cls)
    elif convert_type == "reward":
        logger.debug_once("Converting to reward model.")
        model_cls = as_reward_model(model_cls)
    else:
        assert_never(convert_type)

    return model_cls, arch


def get_model_architecture(
        model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    key = hash((
        model_config.model,
        model_config.convert_type,
        model_config.runner_type,
        model_config.trust_remote_code,
        model_config.model_impl,
        tuple(getattr(model_config.hf_config, "architectures", [])),
    ))
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]

    model_arch = _get_model_architecture(model_config)
    _MODEL_ARCH_BY_HASH[key] = model_arch
    return model_arch


def get_model_cls(model_config: ModelConfig) -> type[nn.Module]:
    return get_model_architecture(model_config)[0]


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


@dataclass
class ParamMapping:
    """
    A class to handle parameter mapping for model weight loading.
    It creates a bidirectional mapping between packed parameters and their 
    constituent parts.
    """
    packed_mapping: dict[str, list[str]]
    inverse_packed_mapping: dict[str, tuple[str,
                                            int]] = field(default_factory=dict)

    def __post_init__(self):
        for packed_name, sub_params in self.packed_mapping.items():
            # Skip self-contained cases (e.g., {"W_pack": ["W_pack"]})
            if len(sub_params) == 1 and sub_params[0] == packed_name:
                continue
            for index, param_name in enumerate(sub_params):
                self.inverse_packed_mapping[param_name] = (
                    packed_name,
                    index,
                )

    def get_sub_modules(self,
                        module_name: str) -> Optional[tuple[str, list[str]]]:
        for key, value in self.packed_mapping.items():
            if module_name.endswith(key):
                return key, value
        return None


def configure_quant_config(quant_config: QuantizationConfig,
                           model_class: type[nn.Module]):
    """
    Pass packed_modules_mapping by reference to quant_config so that
    quant_config can properly match fused modules

    Note that model attributes are passed by reference to quant_config,
    enabling them to be updated by model_class.__new__ (ex. chatglm, qwen)

    Once the `SupportsQuant` mixin has been added to all models, this
    function can be removed
    """
    if not issubclass(model_class, SupportsQuant):
        hf_to_vllm_mapper = getattr(model_class, "hf_to_vllm_mapper", None)
        packed_mapping = getattr(model_class, "packed_modules_mapping", None)

        # pass mappings by reference to quant_config
        if hf_to_vllm_mapper is not None:
            quant_config.apply_vllm_mapper(hf_to_vllm_mapper)
        if packed_mapping is not None:
            quant_config.packed_modules_mapping = packed_mapping
