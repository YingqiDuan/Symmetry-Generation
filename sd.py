from dataclasses import dataclass, field
from typing import Any, Optional
import torch
import torch.nn as nn
from diffusers import DDIMScheduler, StableDiffusionPipeline


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, list) or len(value) not in (3, 4):
        raise TypeError(
            type(value),
            value,
        )

    if len(value) == 3:
        value = [0] + value

    start_step, start_value, end_value, end_step = value
    current_step = global_step if isinstance(end_step, int) else epoch

    return start_value + (end_value - start_value) * max(
        min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
    )


@dataclass
class Config:
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
    guidance_scale: float = 100.0
    grad_clip: list = field(default_factory=lambda: [0, 2.0, 8.0, 1000])
    half_precision_weights: bool = True
    min_step_percent: float = 0.02
    max_step_percent: float = 0.98
    var_red: bool = True
    weighting_strategy: str = "sds"
    token_merging_params: Optional[dict] = field(default_factory=dict)


class StableDiffusion(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configure()

    def configure(self):
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=self.weights_dtype,
        ).to(self.device)

        self.vae, self.unet, self.tokenizer, self.text_encoder = (
            self.pipe.vae,
            self.pipe.unet,
            self.pipe.tokenizer,
            self.pipe.text_encoder,
        )

        for model in [self.vae, self.unet, self.text_encoder]:
            for p in model.parameters():
                p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        self.grad_clip_val: Optional[float] = None
        print(f"Loaded Stable Diffusion")
