"""Операции эксперимента с одной пространственной координатой и временем.

Область задаётся сеткой `(t, x)` формы `(times, xs, 2)` из `DomainRectProducer`.
"""

from collections.abc import Callable
from typing import Any, override

import torch
from torch import nn

from pinn_platform.core import Artifact, OpArg, Operation
from pinn_platform.operation import FuncCondition, ModelProducer
from pinn_platform.operation.condition import requireTensorSource

type TargetFn = Callable[[torch.Tensor], torch.Tensor]


class DomainParts(Operation):
    """Делит сетку `domain` на наборы точек для PINN.

    - `initial` формы `(xs, 2)` содержит точки линии `t = timeStart`.
    - `edges` формы `(times - 1, 2, 2)` содержит пары левых и правых граничных
      точек для `t > timeStart`.
    - `inner` формы `(N, 2)` содержит внутренние точки для `t > timeStart`.

    Граничные точки отбираются парами с одинаковым временем для периодического
    условия. Условие Дирихле разворачивает пары в плоский набор точек.
    """

    @override
    def run(self, inputs: OpArg) -> OpArg:
        domain = inputs["domain"]
        grid = domain.value
        if grid.dim() != 3 or grid.shape[-1] != 2:
            raise ValueError(f"{type(self).__name__}: expected a (t, x) grid, got {grid.shape}")
        parts = {
            "initial": grid[0],
            "edges": torch.stack([grid[1:, 0], grid[1:, -1]], dim=1),
            "inner": grid[1:, 1:-1].reshape(-1, 2),
        }
        return {
            name: Artifact(value=value, meta=domain.meta | {"part": name} | self.getMeta())
            for name, value in parts.items()
        }

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__}


class SampleSource(Operation):
    """Оборачивает фиксированную выборку в источник тензоров или передаёт готовый источник."""

    @override
    def run(self, inputs: OpArg) -> OpArg:
        sample = inputs["data"]
        value = sample.value
        if isinstance(value, torch.Tensor):

            def readSample() -> torch.Tensor:
                return value

            source = readSample
        else:
            source = requireTensorSource(value, "data")
        return {"source": Artifact(value=source, meta=sample.meta)}


class TargetConditionProducer(Operation):
    """Создаёт начальное условие или условие Дирихле по `target(points)`.

    Источник области возвращает точки формы `(..., 2)`.
    """

    def __init__(self, target: TargetFn, label: str):
        self._target = target
        self._label = label

    @override
    def run(self, inputs: OpArg) -> OpArg:
        loss = inputs["loss"].value
        domain = requireTensorSource(inputs["domain"].value, "domain")

        def evaluate(model: nn.Module) -> torch.Tensor:
            points = domain()
            points = points.reshape(-1, points.shape[-1])
            return loss(self._target(points) - model(points))

        condition = FuncCondition(evaluate)
        return {"condition": Artifact(value=condition, meta={"kind": "target"} | self.getMeta())}

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "target": self._label}


class PeriodicEdgesConditionProducer(Operation):
    """Строит периодическое `condition` из `domain` и `loss`.

    Источник `domain` возвращает пары граничных точек из `DomainParts` в форме
    `(n, 2, 2)` при каждом вычислении потерь. Условие требует равенства
    предсказаний модели в точках каждой пары.
    """

    @override
    def run(self, inputs: OpArg) -> OpArg:
        domain = requireTensorSource(inputs["domain"].value, "domain")
        loss = inputs["loss"].value

        def evaluate(model: nn.Module) -> torch.Tensor:
            edges = domain()
            return loss(model(edges[:, 1]) - model(edges[:, 0]))

        condition = FuncCondition(evaluate)
        return {"condition": Artifact(value=condition, meta={"kind": "periodic"} | self.getMeta())}

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__}


def _seedGenerators(seed: int, device: torch.device) -> None:
    """Задаёт начальное состояние генератора CPU и генератора GPU модели, если она на GPU.
    Вызов `torch.manual_seed` задал бы состояние генераторов всех GPU."""
    torch.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


class FactoryModelProducer(ModelProducer):
    """Создаёт модель через `factory()` с начальным состоянием генераторов по `seed`.

    Одинаковый `seed` даёт воспроизводимые начальные веса для той же архитектуры.
    `fork_rng` сохраняет и восстанавливает состояния глобальных генераторов CPU
    и GPU модели, поэтому создание модели сохраняет последовательность случайных чисел
    для остального кода.
    """

    def __init__(
        self,
        factory: Callable[[], nn.Module],
        seed: int,
        device: torch.device,
        meta: dict[str, Any],
    ):
        self._factory = factory
        self._seed = seed
        self._device = torch.device(device)
        self._meta = meta

    @override
    def createModel(self) -> nn.Module:
        gpus = [self._device] if self._device.type == "cuda" else []
        with torch.random.fork_rng(devices=gpus):
            _seedGenerators(self._seed, self._device)
            return self._factory()

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "seed": self._seed} | self._meta
