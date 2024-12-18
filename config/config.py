from omegaconf import OmegaConf
from pathlib import Path
import os
from pprint import pprint


def load_config(default_conf="01.yaml"):
    cli_conf = OmegaConf.from_cli()
    return OmegaConf.merge(
        OmegaConf.load(
            Path(__file__).resolve().parent / cli_conf.get("CONF_FILE", default_conf)
        ),
        cli_conf,
    )


def save_config(args):
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    OmegaConf.save(config=args, f=os.path.join(args.OUTPUT_DIR, "config.yaml"))
    pprint(OmegaConf.to_container(args, resolve=True))
