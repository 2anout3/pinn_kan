"""Запуск эксперимента PINN через граф операций.

    python -m experiments.pinn [overrides]

Конфигурации находятся в `conf/`. Параметр `--config-name` выбирает
эксперимент. По умолчанию используется `transport`.
"""

import logging

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from experiments.pinn.build import buildPipeline, report, setup

log = logging.getLogger(__name__)


def pickDevice(name: str) -> str:
    """`${pick_device:auto}` выбирает CUDA при наличии, иначе CPU. Другие имена сохраняются."""
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


OmegaConf.register_new_resolver("pick_device", pickDevice, replace=True)


@hydra.main(version_base=None, config_path="conf", config_name="transport")
def main(cfg: DictConfig) -> None:
    setup(cfg)
    built = buildPipeline(cfg)
    built.pipe.run()
    report(built)


if __name__ == "__main__":
    main()
