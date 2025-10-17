"""Hyper-ComfyUI HYPIR custom node package.

Wraps the open-source `HYPIR` image restoration project by Xinqi Lin,
Fanghua Yu, Jinfan Hu, Zhiyuan You, Wu Shi, Jimmy S. Ren, Jinjin Gu, and
collaborators. Usage of the upstream model is limited to **non-commercial**
scenarios; see the original repository for details. This package adds
integration glue for ComfyUI under Eric Hiss's dual-license model
documented in `LICENSE.md`.
"""

from __future__ import annotations

import os

from folder_paths import add_model_folder_path

from .hypir_node import HYPIRRestoreNode

# Place the downloadable weights next to this module under the local models folder.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_PACKAGE_DIR, "models")
os.makedirs(_MODELS_DIR, exist_ok=True)

# Register a dedicated model type so weights can live alongside the node.
add_model_folder_path("hyper_hypir", _MODELS_DIR)

NODE_CLASS_MAPPINGS = {
    "HyperComfyUIHYPIRRestore": HYPIRRestoreNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HyperComfyUIHYPIRRestore": "HYPIR Image Restore",
}

NODE_SET_NAME = "Hyper Image Restoration"

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NODE_SET_NAME",
]
