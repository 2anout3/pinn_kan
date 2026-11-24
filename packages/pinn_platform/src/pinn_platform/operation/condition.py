from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, override

import torch
from torch import nn

type TensorFn = Callable[[torch.Tensor], torch.Tensor]
type TensorSource = Callable[[], torch.Tensor]


def requireTensorSource(value: object, name: str) -> Callable:
    if callable(value):
        return value
    raise TypeError(f"{name} must be a callable returning a Tensor, got {type(value).__name__}")


class Condition(ABC):
    """Задаёт функцию потерь модели. Вызов `condition(model)` возвращает тензор потерь.

    `evaluate(model)` возвращает потери и словарь их именованных составляющих.
    У простого условия словарь пуст. `WeightedAccCondition` возвращает
    потери отдельных условий, чтобы тренер мог записать их в лог.
    """

    @abstractmethod
    def __call__(self, model: nn.Module) -> torch.Tensor: ...

    def evaluate(self, model: nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self(model), {}


class FuncCondition(Condition):
    """Оборачивает функцию, вычисляющую потери модели, в объект `Condition`."""

    def __init__(self, func: Callable[[nn.Module], torch.Tensor]):
        self._func = func

    @override
    def __call__(self, model: nn.Module) -> torch.Tensor:
        return self._func(model)


class ExpectedCondition(Condition):
    """Требует от модели ожидаемых значений по формуле `loss(expect - transform(model(domain)))`.

    При каждом вычислении потерь вызываются источник точек и затем источник целевых
    значений. Каждый вызывается один раз. Строки в их результатах должны соответствовать.
    """

    def __init__(
        self,
        domain: TensorSource,
        expect: TensorSource,
        transform: TensorFn,
        loss: TensorFn,
    ):
        self._domain = requireTensorSource(domain, "domain")
        self._expect = requireTensorSource(expect, "expect")
        self._transform = transform
        self._loss = loss

    @override
    def __call__(self, model: nn.Module) -> torch.Tensor:
        points = self._domain()
        expected = self._expect()
        return self._loss(expected - self._transform(model(points)))


class PeriodicCondition(Condition):
    """Требует равенства предсказаний в соответствующих точках из `start()` и `end()`."""

    def __init__(self, start: TensorSource, end: TensorSource, loss: TensorFn):
        self._start = requireTensorSource(start, "start")
        self._end = requireTensorSource(end, "end")
        self._loss = loss

    @override
    def __call__(self, model: nn.Module) -> torch.Tensor:
        return self._loss(model(self._end()) - model(self._start()))


type PDEFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class PDECondition(Condition):
    """Вычисляет `loss(pde(model(domain), domain))` в точках области.

    Источник точек вызывается один раз за вычисление потерь. Точки отсоединяются
    от предыдущего графа, приводятся к форме `(-1, inputs)` и помечаются
    для вычисления градиентов, чтобы `pde` могла взять производные.
    """

    def __init__(self, domain: TensorSource, pde: PDEFn, loss: TensorFn):
        self._domain = requireTensorSource(domain, "domain")
        self._pde = pde
        self._loss = loss

    @override
    def __call__(self, model: nn.Module) -> torch.Tensor:
        domain = self._domain()
        points = domain.detach().reshape(-1, domain.shape[-1]).requires_grad_(True)
        return self._loss(self._pde(model(points), points))


class WeightedAccCondition(Condition):
    """Объединяет именованные условия во взвешенную сумму."""

    def __init__(self, conds: Mapping[str, Condition], weights: Mapping[str, float]):
        self._conds = dict(conds)
        self._weights = dict(weights)

    @override
    def evaluate(self, model: nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Возвращает общие потери и взвешенные потери каждого условия по имени."""
        losses = {name: self._weights[name] * cond(model) for name, cond in self._conds.items()}
        values = list(losses.values())
        return sum(values[1:], values[0]), losses

    @override
    def __call__(self, model: nn.Module) -> torch.Tensor:
        return self.evaluate(model)[0]


def requireCondition(value: Any, owner: str, name: str) -> Condition:
    if not isinstance(value, Condition):
        raise TypeError(
            f"{owner}: input {name!r} must be a Condition, got {type(value).__name__}; "
            "wrap a plain function with FuncCondition"
        )
    return value
