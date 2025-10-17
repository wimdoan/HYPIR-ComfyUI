"""Adapted HYPIR base enhancer for integration with Hyper-ComfyUI.

Original implementation: https://github.com/XPixelGroup/HYPIR
Licensed for non-commercial use only by the original authors.
"""

from __future__ import annotations

from typing import Literal, List, overload

import math

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.nn import functional as F

from ..utils.common import make_tiled_fn, wavelet_reconstruction
from ..utils.tiled_vae import enable_tiled_vae


class BaseEnhancer:
    def __init__(
        self,
        base_model_path,
        weight_path,
        lora_modules,
        lora_rank,
        model_t,
        coeff_t,
        device,
        weight_dtype: torch.dtype | None = None,
    ):
        self.base_model_path = base_model_path
        self.weight_path = weight_path
        self.lora_modules = lora_modules
        self.lora_rank = lora_rank
        self.model_t = model_t
        self.coeff_t = coeff_t
        self.weight_dtype = weight_dtype or torch.bfloat16
        self.device = torch.device(device) if not isinstance(device, torch.device) else device

    def init_models(self):
        self.init_scheduler()
        self.init_text_models()
        self.init_vae()
        self.init_generator()

    @overload
    def init_scheduler(self):
        ...

    @overload
    def init_text_models(self):
        ...

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.vae.eval().requires_grad_(False)

    @overload
    def init_generator(self):
        ...

    @overload
    def prepare_inputs(self, batch_size, prompt):
        ...

    @overload
    def forward_generator(self, z_lq: torch.Tensor) -> torch.Tensor:
        ...

    @torch.no_grad()
    def enhance(
        self,
        lq: torch.Tensor,
        prompt: str,
        scale_by: Literal["factor", "longest_side"] = "factor",
        upscale: int = 1,
        target_longest_side: int | None = None,
        patch_size: int = 512,
        stride: int = 256,
        enhancement_strength: float = 1.0,
        detail_boost: float = 0.0,
        detail_sigma: float = 1.0,
        return_type: Literal["pt", "np", "pil"] = "pt",
    ) -> torch.Tensor | np.ndarray | List[Image.Image]:
        if stride <= 0:
            raise ValueError("Stride must be greater than 0.")
        if patch_size <= 0:
            raise ValueError("Patch size must be greater than 0.")
        if patch_size < stride:
            raise ValueError("Patch size must be greater than or equal to stride.")

        bs = len(lq)
        if scale_by == "factor":
            lq = F.interpolate(lq, scale_factor=upscale, mode="bicubic")
        elif scale_by == "longest_side":
            if target_longest_side is None:
                raise ValueError("target_longest_side must be specified when scale_by is 'longest_side'.")
            h, w = lq.shape[2:]
            if h >= w:
                new_h = target_longest_side
                new_w = int(w * (target_longest_side / h))
            else:
                new_w = target_longest_side
                new_h = int(h * (target_longest_side / w))
            lq = F.interpolate(lq, size=(new_h, new_w), mode="bicubic")
        else:
            raise ValueError(f"Unsupported scale_by method: {scale_by}")

        ref = lq
        h0, w0 = lq.shape[2:]
        if min(h0, w0) <= patch_size:
            lq = self.resize_at_least(lq, size=patch_size)  # type: ignore[attr-defined]

        lq = (lq * 2 - 1).to(dtype=self.weight_dtype, device=self.device)
        h1, w1 = lq.shape[2:]
        vae_scale_factor = 8
        ph = (h1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - h1
        pw = (w1 + vae_scale_factor - 1) // vae_scale_factor * vae_scale_factor - w1
        lq = F.pad(lq, (0, pw, 0, ph), mode="constant", value=0)

        with enable_tiled_vae(self.vae, is_decoder=False, tile_size=patch_size, dtype=self.weight_dtype):
            z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()

        self.prepare_inputs(batch_size=bs, prompt=prompt)
        z = make_tiled_fn(
            fn=lambda z_tile: self.forward_generator(z_tile),
            size=patch_size // vae_scale_factor,
            stride=stride // vae_scale_factor,
            progress=False,
        )(z_lq)

        with enable_tiled_vae(
            self.vae,
            is_decoder=True,
            tile_size=patch_size // vae_scale_factor,
            dtype=self.weight_dtype,
        ):
            x = self.vae.decode(z.to(self.weight_dtype)).sample.float()

        x = x[..., :h1, :w1]
        x = (x + 1) / 2
        x = F.interpolate(input=x, size=(h0, w0), mode="bicubic", antialias=True)
        ref_for_mix = ref.to(device=self.device, dtype=x.dtype)
        x = wavelet_reconstruction(x, ref_for_mix)

        blend = float(max(0.0, min(1.0, enhancement_strength)))
        if blend < 1.0:
            x = torch.lerp(ref_for_mix, x, blend)

        if detail_boost and detail_boost > 0:
            sigma = float(max(detail_sigma, 0.1))
            boost = float(min(max(detail_boost, 0.0), 1.0))
            kernel_size = int(max(3, 2 * math.ceil(sigma * 2) + 1))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = _gaussian_kernel(kernel_size, sigma, device=self.device, dtype=x.dtype)
            kernel = kernel.repeat(x.shape[1], 1, 1, 1)
            blurred = F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.shape[1])
            detail = x - blurred
            x = (x + boost * detail).clamp(0, 1)

        if return_type == "pt":
            return x.clamp(0, 1).cpu()
        if return_type == "np":
            return self.tensor2image(x)  # type: ignore[attr-defined]
        return [Image.fromarray(img) for img in self.tensor2image(x)]  # type: ignore[attr-defined]

def _gaussian_kernel(size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.unsqueeze(0).unsqueeze(0)
    @staticmethod
    def tensor2image(img_tensor):
        return (
            (img_tensor * 255.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

    @staticmethod
    def resize_at_least(imgs: torch.Tensor, size: int) -> torch.Tensor:
        _, _, h, w = imgs.size()
        if h == w:
            new_h, new_w = size, size
        elif h < w:
            new_h, new_w = size, int(w * (size / h))
        else:
            new_h, new_w = int(h * (size / w)), size
        return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)
