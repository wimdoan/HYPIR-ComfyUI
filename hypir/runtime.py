from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

from .enhancer.sd2 import SD2Enhancer

_LORA_MODULES = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "conv",
    "conv1",
    "conv2",
    "conv_shortcut",
    "conv_out",
    "proj_in",
    "proj_out",
    "ff.net.2",
    "ff.net.0.proj",
)
_LORA_RANK = 256
_MODEL_T = 200
_COEFF_T = 200


@dataclass(frozen=True)
class RuntimeConfig:
    base_model_path: str
    weight_path: str
    device: torch.device | str
    weight_dtype: Optional[torch.dtype] = None


class HypirRuntime:
    """Stateful wrapper that caches HYPIR models by (base path, weight)."""

    def __init__(self, config: RuntimeConfig):
        self._config = config
        self.device = _ensure_device(config.device)
        self.weight_dtype = config.weight_dtype or select_weight_dtype(self.device)
        self._ensure_paths_exist()

    def enhance(
        self,
        images: torch.Tensor,
        prompt: str,
        upscale: float,
        patch_size: int,
        stride: int,
        scale_by: str,
        target_longest_side: int | None = None,
        enhancement_strength: float = 1.0,
        detail_boost: float = 0.0,
        detail_sigma: float = 1.0,
    ) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W); received shape {tuple(images.shape)}")

        model = _load_model(
            self._config.base_model_path,
            self._config.weight_path,
            _device_to_key(self.device),
            _dtype_to_name(self.weight_dtype),
        )

        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        use_autocast = self.device.type == "cuda"
        with torch.autocast(device_type=autocast_device, enabled=use_autocast):
            result = model.enhance(
                lq=images.to(device=self.device, dtype=torch.float32),
                prompt=prompt,
                scale_by=scale_by,
                upscale=int(round(upscale)),
                target_longest_side=target_longest_side,
                patch_size=int(patch_size),
                stride=int(stride),
                enhancement_strength=float(enhancement_strength),
                detail_boost=float(detail_boost),
                detail_sigma=float(detail_sigma),
                return_type="pt",
            )
        return result

    def _ensure_paths_exist(self) -> None:
        weight_path = self._config.weight_path
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"HYPIR weight not found: {weight_path}. Download `HYPIR_sd2.pth` and place it in the models folder."
            )
        base_path = self._config.base_model_path
        if os.path.isdir(base_path):
            expected = ["scheduler", "text_encoder", "tokenizer", "unet", "vae"]
            missing = [d for d in expected if not os.path.exists(os.path.join(base_path, d))]
            if missing:
                raise FileNotFoundError(
                    "The base diffusers directory is missing required subfolders: " + ", ".join(missing)
                )
        else:
            # Allow Hugging Face repo ids and other remote descriptors; only raise if the string
            # resolves to an actual (but missing) filesystem location.
            if os.path.isabs(base_path) or os.path.exists(base_path):
                raise FileNotFoundError(
                    "Base model path does not exist. Provide a diffusers directory or a Hugging Face repo id (e.g. 'stabilityai/stable-diffusion-2-1-base')."
                )


def select_weight_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        try:
            index = device.index if device.index is not None else torch.cuda.current_device()
            major, _minor = torch.cuda.get_device_capability(index)
            if major >= 8:
                return torch.bfloat16
            return torch.float16
        except RuntimeError:
            return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _ensure_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _device_to_key(device: torch.device) -> str:
    if device.type == "cuda" and device.index is not None:
        return f"cuda:{device.index}"
    return device.type


def _dtype_to_name(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
    }
    if dtype in mapping:
        return mapping[dtype]
    repr_name = str(dtype)
    return repr_name.split(".")[-1]


@lru_cache(maxsize=4)
def _load_model(base_model_path: str, weight_path: str, device_key: str, dtype_name: str) -> SD2Enhancer:
    device = torch.device(device_key)
    dtype = getattr(torch, dtype_name)
    model = SD2Enhancer(
        base_model_path=base_model_path,
        weight_path=weight_path,
        lora_modules=_LORA_MODULES,
        lora_rank=_LORA_RANK,
        model_t=_MODEL_T,
        coeff_t=_COEFF_T,
        device=device,
        weight_dtype=dtype,
    )
    model.init_models()
    return model
