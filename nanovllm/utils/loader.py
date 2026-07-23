import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from nanovllm.layers.nvfp4_gemm import quantize_nvfp4


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _route(model: nn.Module, packed_modules_mapping: dict, name: str, tensor: torch.Tensor):
    """Apply packed_modules_mapping (q_proj->qkv_proj etc.) and call the target param's weight_loader."""
    for k in packed_modules_mapping:
        if k in name:
            v, shard_id = packed_modules_mapping[k]
            param = model.get_parameter(name.replace(k, v))
            param.weight_loader(param, tensor, shard_id)
            return
    param = model.get_parameter(name)
    getattr(param, "weight_loader", default_weight_loader)(param, tensor)


def _is_nvfp4_param(model: nn.Module, packed_modules_mapping: dict, weight_name: str) -> bool:
    """True if <prefix>.weight maps to a kernel-mode NVFP4 param (uint8)."""
    name = weight_name
    for k in packed_modules_mapping:
        if k in name:
            name = name.replace(k, packed_modules_mapping[k][0])
            break
    try:
        return model.get_parameter(name).dtype == torch.uint8
    except AttributeError:
        return False


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    group_size = getattr(model, "nvfp4_group_size", 16)
    scale_fp8 = getattr(model, "nvfp4_scale_fp8", False)
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            keys = set(f.keys())
            for weight_name in keys:
                if weight_name.endswith((".input_scale", ".k_scale", ".v_scale")):
                    continue    # activation/kv scales: unused (weight-only)
                if weight_name.endswith(".weight_scale_2"):
                    if not scale_fp8:
                        continue    # fp16 path: folded into weight_scale below
                    # fp8 path: keep scale_2 separate as a per-output [out] global-scale vector; expand a
                    # per-tensor scalar to the shard's output width so the shard-aware loader fills the
                    # right merged qkv/gate_up sub-range.
                    s2 = f.get_tensor(weight_name).cuda().float().reshape(-1)
                    out = f.get_slice(weight_name[:-len("_scale_2")]).get_shape()[0]
                    vec = s2.expand(out).contiguous() if s2.numel() == 1 else s2
                    _route(model, packed_modules_mapping, weight_name, vec)
                    continue
                if weight_name.endswith(".weight_scale"):
                    # NVFP4 per-group block scale (fp8_e4m3 on disk).
                    if scale_fp8:
                        # store natively as fp8 (lossless, half the VRAM); scale_2 handled above.
                        _route(model, packed_modules_mapping, weight_name, f.get_tensor(weight_name).cuda())
                        continue
                    # fp16 path: fold the global scale_2 in and store fp16 so merged qkv_proj/gate_up_proj
                    # (each shard has its own global scale) stay correct. Cast on GPU: fp8->float unsupported on CPU.
                    scale = f.get_tensor(weight_name).cuda().float()
                    s2_name = weight_name[:-len("_scale")] + "_scale_2"
                    if s2_name in keys:
                        s2 = f.get_tensor(s2_name).cuda().float()
                        scale = scale * (s2 if s2.numel() == 1 else s2.reshape(-1, 1))
                    _route(model, packed_modules_mapping, weight_name, scale.to(torch.float16))
                    continue
                loaded = f.get_tensor(weight_name)
                if weight_name.endswith(".weight") and loaded.dtype != torch.uint8 \
                        and _is_nvfp4_param(model, packed_modules_mapping, weight_name):
                    # local path: a bf16 checkpoint into a kernel-mode NVFP4 model -> pack on the fly,
                    # matching the real checkpoint's (packed weight + per-group scale) layout.
                    packed, scale = quantize_nvfp4(loaded, group_size)
                    _route(model, packed_modules_mapping, weight_name, packed)
                    _route(model, packed_modules_mapping, weight_name + "_scale", scale)
                    continue
                _route(model, packed_modules_mapping, weight_name, loaded)
