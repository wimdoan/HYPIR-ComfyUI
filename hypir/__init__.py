"""Runtime helpers for embedding HYPIR inside ComfyUI.

Original HYPIR project: https://github.com/XPixelGroup/HYPIR
Authored by Xinqi Lin, Fanghua Yu, Jinfan Hu, Zhiyuan You, Wu Shi,
Jimmy S. Ren, Jinjin Gu, and collaborators. Released for
**non-commercial use only**. This module ships a lightly adapted subset of
that repository for inference within Hyper-ComfyUI.
"""

from .runtime import HypirRuntime, RuntimeConfig

__all__ = ["HypirRuntime", "RuntimeConfig"]
