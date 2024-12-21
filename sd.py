from dataclasses import dataclass, field
from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(self, latents, t, encoder_hidden_states):
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs):
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0  # 缩放到 [-1, 1]
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents,
        latent_height: int = 64,
        latent_width: int = 64,
    ):
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents /= self.vae.config.scaling_factor
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        return (image * 0.5 + 0.5).clamp(0, 1).to(input_dtype)

    def compute_grad_sds(self, latents, text_embeddings, t):
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
            )

        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_text + self.cfg.guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        return w * (noise_pred - noise)

    def train_step(self, rgb, text_embeddings, rgb_as_latents=False):
        batch_size = rgb.shape[0]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            latents = self.encode_images(
                F.interpolate(
                    rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
                )
            )

        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        grad_fn = self.compute_grad_sjc if self.cfg.use_sjc else self.compute_grad_sds
        grad = torch.nan_to_num(grad_fn(latents, text_embeddings, t))

        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
        return loss, t

    def update_step(self, epoch, global_step, on_load_weights=False):
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)
