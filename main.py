import time
from config.config import load_config, save_config
import os
import torch
import nvdiffrast.torch as dr

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
