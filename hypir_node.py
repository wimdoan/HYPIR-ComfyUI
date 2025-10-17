"""Hyper-ComfyUI HYPIR Image Restore node implementation."""

# ---------------------------------------------------------------------------
# ComfyUI Node: Hyper-ComfyUI HYPIR Image Restore
# Description:
#   Restores and optionally upscales scanned or generated images using the
#   HYPIR LoRA fine-tuned for Stable Diffusion 2.1. The node supports tiled
#   VAE processing for large resolutions, blends the restored output with the
#   original image, and can refine prompts through an OpenAI-compatible LM
#   Studio endpoint prior to inference.
#
# Author: Eric Hiss (GitHub: EricRollei)
# Contact: eric@historic.camera, eric@rollei.us
# Version: 1.0.0
# Date: October 2025
# License: Dual License (Non-Commercial and Commercial Use)
# Copyright (c) 2025 Eric Hiss. All rights reserved.
#
# Dual License:
# 1. Non-Commercial Use: Creative Commons Attribution-NonCommercial 4.0
#    International (http://creativecommons.org/licenses/by-nc/4.0/)
# 2. Commercial Use: Contact Eric Hiss (eric@historic.camera, eric@rollei.us)
#    for licensing options.
#
# Integrated upstream assets are subject to their respective licenses:
#   - HYPIR project (XPixelGroup) – non-commercial license
#   - Stable Diffusion 2.1 (Stability AI) – CreativeML Open RAIL++-M License
#   - Diffusers / Transformers / Accelerate / PEFT – Apache 2.0
#
# Dependencies:
#   torch, diffusers, transformers, peft, accelerate, numpy, Pillow, tqdm,
#   requests (for LM Studio integration)
# ---------------------------------------------------------------------------
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from accelerate.utils import set_seed
import comfy.model_management as model_management
from folder_paths import get_filename_list, get_full_path

from .hypir.runtime import HypirRuntime, RuntimeConfig
from .lmstudio import LMStudioClient, LMStudioConfig, PromptEnhancementError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _NodeDefaults:
    base_model_path: str = "stabilityai/stable-diffusion-2-1-base"
    weight_default: str = "HYPIR_sd2.pth"
    upscale: float = 4.0
    patch_size: int = 512
    stride: int = 256
    scale_by: str = "factor"
    target_longest_side: int = 2048
    seed: int = -1


@dataclass(frozen=True)
class _LMDefaults:
    endpoint: str = "http://127.0.0.1:1234"
    model: str = "lmstudio"
    system_prompt: str = (
        "You are an expert photographic conservator working with Stable Diffusion 2.1 and the HYPIR restoration LoRA. "
        "Rewrite the provided notes into a concise 1-2 sentence restoration prompt that keeps subjects, era, and mood intact. "
        "Emphasise realistic fine detail, natural lighting, gentle noise reduction, and faithful colour recovery. "
        "Do not invent new elements, props, or camera changes unless explicitly requested. Return only the final prompt text without quotes. "
        "If the user supplies no notes, respond with: 'Restored vintage photograph, faithful colours, natural film grain preserved, realistic detail and gentle contrast.'\n\n"
        "Example: Restored 1950s family portrait, smiling couple on front porch, warm evening light, crisp yet natural detail, "
        "subtle grain preserved."
    )
    fallback_prompt: str = (
        "Restored vintage photograph, faithful colours, natural film grain preserved, realistic detail and gentle contrast."
    )
    temperature: float = 0.6
    max_tokens: int = 120


class HYPIRRestoreNode:
    """Upscale & restore an image using the HYPIR LoRA on SD2.1."""

    NODE_NAME = "hyper_comfyui_hypir_restore"
    FUNCTION = "restore"
    CATEGORY = "Hyper Image Restoration/HYPIR"
    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("image", "seed", "prompt")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Tuple]]:
        defaults = _NodeDefaults()
        weight_choices = get_filename_list("hyper_hypir")
        if defaults.weight_default not in weight_choices:
            weight_choices = weight_choices + [defaults.weight_default] if weight_choices else [defaults.weight_default]

        lm_defaults = _LMDefaults()

        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "base_model_path": (
                    "STRING",
                    {
                        "default": defaults.base_model_path,
                        "tooltip": "Diffusers-format directory or Hugging Face repo id (e.g. stabilityai/stable-diffusion-2-1-base).",
                    },
                ),
                "weight_name": (
                    "STRING",
                    {
                        "default": defaults.weight_default,
                        "choices": sorted(set(weight_choices)),
                        "tooltip": "Name of the HYPIR LoRA weight file placed in the node's models folder.",
                    },
                ),
                "upscale": (
                    "FLOAT",
                    {
                        "default": defaults.upscale,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.5,
                        "tooltip": "Upscale factor applied before restoration (matches HYPIR defaults).",
                    },
                ),
                "patch_size": (
                    "INT",
                    {
                        "default": defaults.patch_size,
                        "min": 256,
                        "max": 1024,
                        "step": 64,
                        "tooltip": "Tile size used for latent processing and VAE tiling.",
                    },
                ),
                "stride": (
                    "INT",
                    {
                        "default": defaults.stride,
                        "min": 128,
                        "max": 1024,
                        "step": 64,
                        "tooltip": "Stride between tiles. Lower values improve seam quality at the cost of time.",
                    },
                ),
                "scale_by": (
                    "STRING",
                    {
                        "default": defaults.scale_by,
                        "choices": ["factor", "longest_side"],
                        "tooltip": "Choose between fixed upscale factor or fitting to a target longest side.",
                        "display": "combo",
                    },
                ),
            },
            "optional": {
                "target_longest_side": (
                    "INT",
                    {
                        "default": defaults.target_longest_side,
                        "min": 256,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "Used when scale_by=longest_side. Final longest edge after upscaling.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": defaults.seed,
                        "tooltip": "Random seed for deterministic noise sampling. Use -1 for random seed each run.",
                    },
                ),
                "enhance_prompt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Toggle to rewrite the prompt using an LM Studio chat completion before restoration.",
                    },
                ),
                "lm_endpoint": (
                    "STRING",
                    {
                        "default": lm_defaults.endpoint,
                        "tooltip": "LM Studio base URL (without /v1/chat/completions).",
                    },
                ),
                "lm_model": (
                    "STRING",
                    {
                        "default": lm_defaults.model,
                        "tooltip": "Model identifier exposed by LM Studio (see the UI > Provider > Model).",
                    },
                ),
                "lm_system_prompt": (
                    "STRING",
                    {
                        "default": lm_defaults.system_prompt,
                        "multiline": True,
                        "tooltip": "System prompt fed to LM Studio when enhancing text prompts.",
                    },
                ),
                "lm_temperature": (
                    "FLOAT",
                    {
                        "default": lm_defaults.temperature,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature passed to LM Studio.",
                    },
                ),
                "lm_max_tokens": (
                    "INT",
                    {
                        "default": lm_defaults.max_tokens,
                        "min": 16,
                        "max": 512,
                        "step": 8,
                        "tooltip": "Maximum number of tokens generated by LM Studio for the enhanced prompt.",
                    },
                ),
                "enhancement_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Blend between the original image (0) and the full HYPIR result (1).",
                    },
                ),
                "detail_boost": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Add extra high-frequency detail on top of the restored image.",
                    },
                ),
                "detail_sigma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Controls the radius of the detail boost blur (higher = broader).",
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **_kwargs) -> float:
        # Always run to reflect input changes; heavy caching happens inside HypirRuntime itself.
        return float(random.random())

    def restore(
        self,
        image: torch.Tensor,
        prompt: str,
        base_model_path: str,
        weight_name: str,
        upscale: float,
        patch_size: int,
        stride: int,
        scale_by: str,
        target_longest_side: int | None = None,
        seed: int = -1,
        enhance_prompt: bool = False,
        lm_endpoint: str | None = None,
        lm_model: str | None = None,
        lm_system_prompt: str | None = None,
        lm_temperature: float | None = None,
        lm_max_tokens: int | None = None,
        enhancement_strength: float | None = None,
        detail_boost: float | None = None,
        detail_sigma: float | None = None,
    ) -> Tuple[torch.Tensor, int, str]:
        if image is None:
            raise ValueError("Input image tensor is required.")

        weight_path = self._resolve_weight_path(weight_name)
        device = model_management.get_torch_device()
        runtime = HypirRuntime(RuntimeConfig(
            base_model_path=base_model_path.strip(),
            weight_path=weight_path,
            device=device,
        ))

        actual_seed = self._prepare_seed(seed)
        set_seed(actual_seed)

        batched = image.permute(0, 3, 1, 2).contiguous().to(device=device, dtype=torch.float32)

        effective_prompt = prompt or ""
        if enhance_prompt:
            effective_prompt = self._maybe_enhance_prompt(
                effective_prompt,
                lm_endpoint=lm_endpoint,
                lm_model=lm_model,
                lm_system_prompt=lm_system_prompt,
                lm_temperature=lm_temperature,
                lm_max_tokens=lm_max_tokens,
            )

        restored = runtime.enhance(
            images=batched,
            prompt=effective_prompt,
            upscale=upscale,
            patch_size=patch_size,
            stride=stride,
            scale_by=scale_by,
            target_longest_side=target_longest_side,
            enhancement_strength=enhancement_strength if enhancement_strength is not None else 1.0,
            detail_boost=detail_boost if detail_boost is not None else 0.0,
            detail_sigma=detail_sigma if detail_sigma is not None else 1.0,
        )

        model_management.soft_empty_cache()

        output = restored.to(torch.float32).permute(0, 2, 3, 1).contiguous().cpu()
        return output, int(actual_seed), effective_prompt

    @staticmethod
    def _resolve_weight_path(weight_name: str) -> str:
        if not weight_name:
            raise ValueError("Select a HYPIR weight file in the node's models folder.")
        path = get_full_path("hyper_hypir", weight_name)
        if path is None:
            raise FileNotFoundError(
                f"Could not resolve weight '{weight_name}'. Place the LoRA in the models folder registered for Hyper-ComfyUI."
            )
        return path

    @staticmethod
    def _prepare_seed(seed: int) -> int:
        if seed is None or seed < 0:
            return random.randint(0, 2**32 - 1)
        return int(seed)

    def _maybe_enhance_prompt(
        self,
        prompt: str,
        *,
        lm_endpoint: str | None,
        lm_model: str | None,
        lm_system_prompt: str | None,
        lm_temperature: float | None,
        lm_max_tokens: int | None,
    ) -> str:
        defaults = _LMDefaults()

        endpoint = (lm_endpoint or defaults.endpoint).strip()
        model = (lm_model or defaults.model).strip()
        system_prompt = lm_system_prompt or defaults.system_prompt
        temperature = lm_temperature if lm_temperature is not None else defaults.temperature
        max_tokens = lm_max_tokens if lm_max_tokens is not None else defaults.max_tokens

        if not endpoint:
            logger.warning("LM Studio endpoint is empty; skipping prompt enhancement.")
            return prompt
        if not model:
            logger.warning("LM Studio model is empty; skipping prompt enhancement.")
            return prompt

        config = LMStudioConfig(
            endpoint=endpoint,
            model=model,
            system_prompt=system_prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        client = LMStudioClient(config)

        try:
            enhanced = client.enhance_prompt(prompt)
        except PromptEnhancementError as exc:
            message = f"LM Studio prompt enhancement failed: {exc}"
            logger.warning(message)
            print(f"[HYPIR] {message}")
            return prompt

        fallback_prompt = defaults.fallback_prompt
        stripped_input = prompt.strip()
        enhanced = (enhanced or "").strip()

        if not enhanced:
            print("[HYPIR] LM Studio returned empty content; using fallback prompt.")
            return fallback_prompt if not stripped_input else prompt

        if not stripped_input:
            lower = enhanced.lower()
            if "?" in enhanced or "provide" in lower and ("notes" in lower or "prompt" in lower):
                print("[HYPIR] LM Studio asked for more input; using fallback prompt instead.")
                return fallback_prompt

        snippet = (enhanced.replace("\n", " ")[:120] + ("…" if len(enhanced) > 120 else ""))
        print(f"[HYPIR] LM Studio prompt applied: {snippet}")
        return enhanced
