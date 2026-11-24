import logging
from pathlib import Path
from typing import Any, override

import torch
from torch import nn

from pinn_platform.core.artifact import Artifact
from pinn_platform.operation._operation import OpArg, Operation

log = logging.getLogger(__name__)


class SaveModelCheckpoint(Operation):
    """Записывает `state_dict` входной модели `model` в `path` и передаёт модель дальше.

    На выходе `model` находится та же модель. Узлы, которым нужно выполниться
    после сохранения, например оценка модели, могут читать этот выход.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)

    @override
    def run(self, inputs: OpArg) -> OpArg:
        source = inputs["model"]
        model: nn.Module = source.value
        self._path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self._path)
        log.info(f"[checkpoint] saved {self._path}")
        return {"model": Artifact(value=model, meta=source.meta | self.getMeta())}

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "checkpoint": str(self._path)}


class LoadModelCheckpoint(Operation):
    """Загружает `state_dict` из `path` во входную модель и возвращает её как `model`.

    Создайте модель по той же конфигурации, что и при сохранении. Параметр
    `strict=False` допускает отсутствующие и неожиданные ключи. Тензоры
    считываются на CPU и копируются на устройство модели.
    """

    def __init__(self, path: str | Path, strict: bool = True):
        self._path = Path(path)
        self._strict = strict

    @override
    def run(self, inputs: OpArg) -> OpArg:
        source = inputs["model"]
        model: nn.Module = source.value
        if not self._path.is_file():
            raise FileNotFoundError(f"{type(self).__name__}: checkpoint {self._path} not found")
        state = torch.load(self._path, map_location="cpu", weights_only=True)
        result = model.load_state_dict(state, strict=self._strict)
        if result.missing_keys or result.unexpected_keys:
            log.warning(
                f"[checkpoint] {self._path}: missing={result.missing_keys} "
                f"unexpected={result.unexpected_keys}"
            )
        model.eval()
        log.info(f"[checkpoint] loaded {self._path}")
        return {"model": Artifact(value=model, meta=source.meta | self.getMeta())}

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "checkpoint": str(self._path), "strict": self._strict}
