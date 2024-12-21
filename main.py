import time
from config.config import load_config, save_config
import os
import torch
import nvdiffrast.torch as dr
import sd

start = time.time()


class Generator:
    def __init__(self) -> None:
        self.args = load_config()
        save_config(self.args)
        self.temp = os.path.join(self.args.OUTPUT_DIR, "temp")
        os.makedirs(self.temp, exist_ok=True)
        self.device = torch.device(self.args.DEVICE)

        self.glctx = dr.RasterizeCudaContext()
        self.mv = self.proj = torch.eye(4, device=self.device).unsqueeze(0)

        self.black_bg = torch.zeros(1, 3, 512, 512).cuda()
        self.white_bg = torch.ones(1, 3, 512, 512)

    def load_model(self):
        config = sd.Config(
            pretrained_model_name_or_path=self.args.PRETRAINED_MODEL_NAME_OR_PATH,
            guidance_scale=self.args.GUIDANCE_SCALE,
            half_precision_weights=self.args.USE_HALF_PRECISION,
            grad_clip=[0, 2.0, 8.0, 1000],
        )
        self.model = sd.StableDiffusion(config)
