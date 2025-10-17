"""Stable Diffusion 2.1 enhancer adapted from the HYPIR project."""

from __future__ import annotations

import logging

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer

from .base import BaseEnhancer

logger = logging.getLogger(__name__)


class SD2Enhancer(BaseEnhancer):
    def init_scheduler(self):
        self.scheduler = DDPMScheduler.from_pretrained(self.base_model_path, subfolder="scheduler")

    def init_text_models(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(self.base_model_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_model_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.text_encoder.eval().requires_grad_(False)

    def init_generator(self):
        self.G: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            self.base_model_path,
            subfolder="unet",
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        target_modules = self.lora_modules
        lora_cfg = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.G.add_adapter(lora_cfg)

        logger.info("Loading HYPIR LoRA weights from %s", self.weight_path)
        state_dict = _load_state_dict(self.weight_path)
        missing, unexpected = self.G.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing expected LoRA keys: %s", ", ".join(sorted(missing)))
        if unexpected:
            logger.warning("Unexpected LoRA keys: %s", ", ".join(sorted(unexpected)))
        self.G.eval().requires_grad_(False)

    def prepare_inputs(self, batch_size, prompt):
        txt_ids = self.tokenizer(
            [prompt] * batch_size,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        text_embed = self.text_encoder(txt_ids.to(self.device))[0]
        c_txt = {"text_embed": text_embed}
        timesteps = torch.full((batch_size,), self.model_t, dtype=torch.long, device=self.device)
        self.inputs = {
            "c_txt": c_txt,
            "timesteps": timesteps,
        }

    def forward_generator(self, z_lq):
        z_in = z_lq * self.vae.config.scaling_factor
        eps = self.G(
            z_in,
            self.inputs["timesteps"],
            encoder_hidden_states=self.inputs["c_txt"]["text_embed"],
        ).sample
        z = self.scheduler.step(eps, self.coeff_t, z_in).pred_original_sample
        z_out = z / self.vae.config.scaling_factor
        return z_out


def _load_state_dict(weight_path: str) -> dict:
    try:
        return torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location="cpu")
