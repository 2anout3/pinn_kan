from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, override

import torch
from torch import nn

from pinn_platform.core.artifact import Artifact
from pinn_platform.operation._operation import OpArg, Operation
from pinn_platform.operation.condition import (
    ExpectedCondition,
    PDECondition,
    PeriodicCondition,
    WeightedAccCondition,
    requireCondition,
)


class DomainRectProducer(Operation):
    def __init__(
        self,
        start: list[float],
        end: list[float],
        steps: list[int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        if len(start) != len(end) and len(start) != len(steps):
            raise  #
        self.start = start
        self.end = end
        self.steps = steps
        self.dtype = dtype
        self.device = device

    @override
    def run(self, inputs: OpArg) -> OpArg:
        axis = [
            torch.linspace(s, e, st, dtype=self.dtype, device=self.device)
            for s, e, st in zip(self.start, self.end, self.steps, strict=True)
        ]
        grids = torch.meshgrid(axis, indexing="ij")
        result = torch.stack(grids, dim=-1)
        return {"domain": Artifact(value=result, meta=self.getMeta())}


class ExpectedConditionProducer(Operation):
    """Создаёт условие на ожидаемые значения из вызываемых входов `domain` и `expect`."""

    @override
    def run(self, inputs: OpArg) -> OpArg:
        condition = ExpectedCondition(
            inputs["domain"].value,
            inputs["expect"].value,
            inputs["transform"].value,
            inputs["loss"].value,
        )
        return {"condition": Artifact(value=condition, meta={"kind": "expected"})}


class PeriodicConditionProducer(Operation):
    """Создаёт периодическое условие из вызываемых входов `start` и `end`."""

    @override
    def run(self, inputs: OpArg) -> OpArg:
        condition = PeriodicCondition(
            inputs["start"].value, inputs["end"].value, inputs["loss"].value
        )
        return {"condition": Artifact(value=condition, meta={"kind": "periodic"})}


class PDEConditionProducer(Operation):
    """Создаёт `condition` из входов `domain`, `pde` и `loss`.

    Вход `domain` — функция, возвращающая тензор точек.
    """

    @override
    def run(self, inputs: OpArg) -> OpArg:
        condition = PDECondition(inputs["domain"].value, inputs["pde"].value, inputs["loss"].value)
        return {"condition": Artifact(value=condition, meta={"kind": "pde"} | inputs["pde"].meta)}


class WeightedAccConditionProducer(Operation):
    """Объединяет входные условия в один `WeightedAccCondition`.

    Имена входов должны точно совпадать с именами весов.
    """

    def __init__(self, weights: Mapping[str, float]):
        if not weights:
            raise ValueError("weights must not be empty")
        negative = {name: weight for name, weight in weights.items() if weight < 0}
        if negative:
            raise ValueError(f"weights must be non-negative, got {negative}")
        self._weights = dict(weights)

    @override
    def run(self, inputs: OpArg) -> OpArg:
        owner = type(self).__name__
        conds = {
            name: requireCondition(artifact.value, owner, name) for name, artifact in inputs.items()
        }
        if conds.keys() != self._weights.keys():
            raise ValueError(
                f"{owner}: conditions {sorted(conds)} do not match weights {sorted(self._weights)}"
            )
        condition = WeightedAccCondition(conds, self._weights)
        return {"condition": Artifact(value=condition, meta={"kind": "weighted"} | self.getMeta())}

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "weights": dict(self._weights)}


def _meanSquare(residual: torch.Tensor) -> torch.Tensor:
    return residual.square().mean()


class MSELossProducer(Operation):
    """Возвращает функцию `loss = mean(residual^2)` для вычисления потерь по невязке."""

    @override
    def run(self, inputs: OpArg) -> OpArg:
        return {"loss": Artifact(value=_meanSquare, meta={"kind": "loss"} | self.getMeta())}

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__}


def _identity(value: torch.Tensor) -> torch.Tensor:
    return value


class IdentityTransformProducer(Operation):
    """Создаёт тождественное преобразование, возвращающее входное значение."""

    @override
    def run(self, inputs: OpArg) -> OpArg:
        return {"transform": Artifact(value=_identity, meta={"kind": "identity"})}


class PDETransformProducer(Operation):
    """Возвращает `pde = self.pde`, функцию невязки PDE. Формулы определены в `pde.py`."""

    @abstractmethod
    def pde(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Вычисляет невязку формы `(..., equations)` по значениям `y` формы
        `(..., outputs)` в точках `x`.
        """

    @override
    def run(self, inputs: OpArg) -> OpArg:
        return {"pde": Artifact(value=self.pde, meta={"kind": "pde"} | self.getMeta())}


class ModelProducer(Operation):
    @abstractmethod
    def createModel(self) -> nn.Module: ...

    @override
    def run(self, inputs: OpArg) -> OpArg:
        model = self.createModel()
        return {"model": Artifact(value=model, meta={"kind": "model"} | self.getMeta())}
