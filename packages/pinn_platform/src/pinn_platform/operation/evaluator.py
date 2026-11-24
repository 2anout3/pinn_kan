from typing import Any, override

import torch

from pinn_platform.core.artifact import Artifact
from pinn_platform.operation._operation import OpArg, Operation


class ModelEvaluator(Operation):
    """Вычисляет предсказания входной модели на области и возвращает их как `values`.

    Координаты точек лежат на последней оси входа `domain`. Например, сетка
    `DomainRectProducer` имеет форму `(*grid, inputs)`. Предсказания сохраняют
    форму сетки `(*grid, outputs)`. Параметр `batchSize` позволяет вычислять
    их частями для экономии памяти. Градиенты отключены.
    """

    def __init__(self, batchSize: int | None = None):
        if batchSize is not None and batchSize <= 0:
            raise ValueError(f"batchSize must be positive or None, got {batchSize}")
        self._batchSize = batchSize

    @override
    def run(self, inputs: OpArg) -> OpArg:
        domain = inputs["domain"]
        model = inputs["model"].value

        points = domain.value.reshape(-1, domain.value.shape[-1])
        chunks = points.split(self._batchSize) if self._batchSize is not None else (points,)
        with torch.no_grad():
            values = torch.cat([model(chunk) for chunk in chunks])
        values = values.reshape(*domain.value.shape[:-1], -1)

        return {"values": Artifact(value=values, meta=domain.meta | self.getMeta())}

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "batchSize": self._batchSize}
