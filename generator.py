from config.config import load_config, save_config
from mesh import square_mesh, split_square_boundary
import os
import torch
import nvdiffrast.torch as dr
import sd
import numpy as np
import igl


class Generator:
    def __init__(self) -> None:
        self.args = load_config()
        save_config(self.args)
        self.temp = os.path.join(self.args.OUTPUT_DIR, "temp")
        os.makedirs(self.temp, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.glctx = dr.RasterizeCudaContext()
        self.mv = self.proj = torch.eye(4, device=self.device).unsqueeze(0)

        self.black_bg = torch.zeros(1, 3, 512, 512).cuda()
        self.white_bg = torch.ones(1, 3, 512, 512)

    def load_model(self):
        config = sd.Config(
            pretrained_model_name_or_path=self.args.PRETRAINED_MODEL_NAME_OR_PATH,
            guidance_scale=self.args.GUIDANCE_SCALE,
            grad_clip=[0, 2.0, 8.0, 1000],
        )
        self.model = sd.StableDiffusion(config)

    def embedding(self):
        prompts = self.args.PROMPT
        if isinstance(prompts, str):
            prompts = [prompts]
        n_prompt = len(prompts)
        batch_size = self.args.BATCH_SIZE
        self.args.PROMPT = prompts

        if n_prompt == 1:
            batch_size_per_prompt = batch_size
        else:
            if not (n_prompt == 2 or (n_prompt**0.5).is_integer()):
                raise ValueError("Number of prompts must be 1, 2, or a square number")
            if batch_size % n_prompt != 0:
                raise ValueError(
                    "Batch size must be a multiple of the number of prompts"
                )
            batch_size_per_prompt = batch_size // n_prompt
            self.batch_size_per_prompt = batch_size_per_prompt

        with torch.no_grad():
            negative_embedding = self.model.get_text_embeds(self.args.NEGATIVE_PROMPT)
            prompt_embedding = [
                self.model.get_text_embeds(prompt) for prompt in self.args.PROMPT
            ]

            positive_embedding = []
            for embed in prompt_embedding:
                positive_embedding.extend([embed] * batch_size_per_prompt)
            self.text_embeds = torch.cat(
                positive_embedding + [negative_embedding] * batch_size
            )

        del self.model.text_encoder
