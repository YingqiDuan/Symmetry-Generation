from omegaconf import OmegaConf
from pathlib import Path


def load_config(default_conf="01.yaml"):
    cli_conf = OmegaConf.from_cli()
    return OmegaConf.merge(
        OmegaConf.load(
            Path(__file__).resolve().parent / cli_conf.get("CONF_FILE", default_conf)
        ),
        cli_conf,
    )
