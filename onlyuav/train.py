from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from onlyuav.models import load_default_components
from onlyuav.rl.runner import run_training

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    load_default_components()
    run_training(cfg)


if __name__ == "__main__":
    main()
