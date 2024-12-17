import time
from config.config import load_config

start = time.time()


class Generator:
    def __init__(self) -> None:
        args = load_config()
