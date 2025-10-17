"""Utility helpers adapted from the HYPIR project (non-commercial license)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from tqdm import tqdm


def wavelet_blur(image: Tensor, radius: int) -> Tensor:
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels: int = 5) -> Tuple[Tensor, Tensor]:
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2 ** i
        low_freq = wavelet_blur(image, radius)
        high_freq += image - low_freq
        image = low_freq
    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    content_high_freq, _ = wavelet_decomposition(content_feat)
    _, style_low_freq = wavelet_decomposition(style_feat)
    return content_high_freq + style_low_freq


def sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> Iterable[Tuple[int, int, int, int]]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)

    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)

    for hi in hi_list:
        for wi in wi_list:
            yield hi, hi + tile_size, wi, wi + tile_size


def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (latent_width - 1) / 2
    x_probs = [
        np.exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)
    ]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)
    ]
    weights = np.outer(y_probs, x_probs)
    return weights


@dataclass(frozen=True)
class TileIndex:
    hi: int
    hi_end: int
    wi: int
    wi_end: int


def make_tiled_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    stride: int,
    scale_type: Literal["up", "down"] = "up",
    scale: int = 1,
    channel: int | None = None,
    weight: Literal["uniform", "gaussian"] = "gaussian",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    progress: bool = True,
    desc: str | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def tiled_fn(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if scale_type == "up":
            scale_fn = lambda n: int(n * scale)
        else:
            scale_fn = lambda n: int(n // scale)

        b, c, h, w = x.size()
        out_dtype = dtype or x.dtype
        out_device = device or x.device
        out_channel = channel or c
        out = torch.zeros((b, out_channel, scale_fn(h), scale_fn(w)), dtype=out_dtype, device=out_device)
        count = torch.zeros_like(out, dtype=torch.float32)

        weight_size = scale_fn(size)
        if weight == "gaussian":
            weights = gaussian_weights(weight_size, weight_size)[None, None]
        else:
            weights = np.ones((1, 1, weight_size, weight_size))
        weights = torch.tensor(weights, dtype=out_dtype, device=out_device)

        indices = list(sliding_windows(h, w, size, stride))
        iterator = tqdm(indices, desc=f"[{desc}]: Tiled Processing" if desc else "Tiled Processing", disable=not progress)
        for hi, hi_end, wi, wi_end in iterator:
            x_tile = x[..., hi:hi_end, wi:wi_end]
            out_hi, out_hi_end, out_wi, out_wi_end = map(scale_fn, (hi, hi_end, wi, wi_end))
            if len(args) or len(kwargs):
                kwargs.update(index=TileIndex(hi=hi, hi_end=hi_end, wi=wi, wi_end=wi_end))
            tile_out = fn(x_tile, *args, **kwargs)
            out[..., out_hi:out_hi_end, out_wi:out_wi_end] += tile_out * weights
            count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights

        out = out / count
        return out

    return tiled_fn
