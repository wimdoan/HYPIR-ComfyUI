"""Minimal VAE tiling helpers to keep memory use manageable.

The upstream HYPIR project installs custom hooks that tile the VAE during
encoding and decoding. Rather than porting those hooks verbatim we rely on
diffusers' built-in tiling support, which became available in recent
versions of ``AutoencoderKL``.
"""

from __future__ import annotations

from contextlib import contextmanager


def _compute_latent_tile(tile_size: int) -> int:
    latent = int(tile_size // 8)
    return max(latent, 16)


@contextmanager
def enable_tiled_vae(vae, *, is_decoder: bool, tile_size: int, dtype):
    """Temporarily enable diffusers' tiled VAE execution."""

    if tile_size is None or tile_size <= 0 or not hasattr(vae, "enable_tiling"):
        yield
        return

    # Capture existing tiling configuration so we can restore it after the
    # encode/decode call finishes. diffusers keeps the parameters as mutable
    # attributes on the AutoencoderKL instance.
    prior_use_tiling = getattr(vae, "use_tiling", False)
    prior_sample_height = getattr(vae, "tile_sample_min_height", None)
    prior_sample_width = getattr(vae, "tile_sample_min_width", None)
    prior_latent_height = getattr(vae, "tile_latent_min_height", None)
    prior_latent_width = getattr(vae, "tile_latent_min_width", None)
    prior_overlap = getattr(vae, "tile_overlap_factor", None)

    vae.enable_tiling()
    vae.tile_sample_min_height = tile_size
    vae.tile_sample_min_width = tile_size

    latent_tile = _compute_latent_tile(tile_size)
    vae.tile_latent_min_height = latent_tile
    vae.tile_latent_min_width = latent_tile

    # Use a conservative overlap so seams remain hidden while keeping memory low.
    vae.tile_overlap_factor = 0.25

    try:
        yield
    finally:
        if not prior_use_tiling:
            vae.disable_tiling()
        else:
            vae.enable_tiling()

        if prior_sample_height is not None:
            vae.tile_sample_min_height = prior_sample_height
        if prior_sample_width is not None:
            vae.tile_sample_min_width = prior_sample_width
        if prior_latent_height is not None:
            vae.tile_latent_min_height = prior_latent_height
        if prior_latent_width is not None:
            vae.tile_latent_min_width = prior_latent_width
        if prior_overlap is not None:
            vae.tile_overlap_factor = prior_overlap
