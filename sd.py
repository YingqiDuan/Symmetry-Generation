from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from jaxtyping import Float, Int
from torch import Tensor


@dataclass
class Config:
    model: str = "stabilityai/stable-diffusion-2-1-base"
    guidance_scale: float = 100.0
    grad_clip: Optional[Any] = field(default_factory=lambda: [0, 2.0, 8.0, 1000])
    min_step_percent: float = 0.02
    max_step_percent: float = 0.98
    weighting_strategy: str = "sds"


class StableDiffusion(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configure()

    def configure(self) -> None:
        self.weights_dtype = torch.float16
        pipe_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.model,
            **pipe_kwargs,
        ).to(self.device)

        self.vae, self.unet = self.pipe.vae, self.pipe.unet
        self.tokenizer, self.text_encoder = self.pipe.tokenizer, self.pipe.text_encoder

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.model,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.max_step = int(self.num_train_timesteps * self.cfg.max_step_percent)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.grad_clip_val = None

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return self.text_encoder(inputs.input_ids.to(self.device))[0]

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        out = self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        )
        return out.sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = latents / self.vae.config.scaling_factor
        img = self.vae.decode(latents.to(self.weights_dtype)).sample
        return (img * 0.5 + 0.5).clamp(0, 1).to(input_dtype)

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        t: Int[Tensor, "B"],
    ):
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            lm_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                lm_input, torch.cat([t] * 2), text_embeddings
            )

        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if self.cfg.weighting_strategy == "sds":
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        return w * (noise_pred - noise)

    def train_step(
        self,
        rgb: Float[Tensor, "B H W C"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        rgb_as_latents=False,
    ):
        bsz = rgb.size(0)
        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            latents = self.encode_images(rgb_BCHW_512)

        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            (bsz,),
            dtype=torch.long,
            device=self.device,
        )
        grad = self.compute_grad_sds(latents, text_embeddings, t)
        grad = torch.nan_to_num(grad)
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / bsz
        return loss, t
