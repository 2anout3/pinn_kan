"""Генерирует YAML-файлы опций для узлов пайплайна.

    python -m experiments.pinn.genconf [--overwrite]

Для каждой фабрики из `experiments.pinn.nodes`, объявленной с
`@hydra_config(name, group)`, создаёт `conf/<group>/<name>.yaml`
со значениями по умолчанию. Флаг `--overwrite` включает перезапись существующих файлов.
"""

import argparse
import logging
import pathlib

import experiments.pinn.nodes  # noqa: F401  (registers the node configs)
from pinn_platform.conf import write_config_tree

CONF = pathlib.Path(__file__).parent / "conf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing files")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for path in write_config_tree(CONF, overwrite=args.overwrite):
        print(f"written {path.relative_to(CONF)}")


if __name__ == "__main__":
    main()
