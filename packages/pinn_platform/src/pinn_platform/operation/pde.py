"""Формулы невязок PDE для `PDETransformProducer`.

Функция `pde(y, x)` получает предсказания `y` формы `(..., outputs)`
в точках `x` формы `(..., 2)` со столбцами `(t, x)`. Она возвращает
невязки формы `(..., equations)`. Для `x` должны вычисляться градиенты,
а `y` должно быть получено из `x` (см. `PDECondition`).
"""

from typing import Any, override

import torch

from pinn_platform.operation.producer import PDETransformProducer


def jacobian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Вычисляет производные `dy_i/dx_j` в каждой точке с формой `(..., outputs, inputs)`.

    Граф сохраняется через `create_graph=True`, чтобы вычислить градиенты невязки
    по параметрам модели. Для компоненты, независимой от `x`, производные равны нулю.
    """
    rows = [
        torch.autograd.grad(
            y[..., i],
            x,
            torch.ones_like(y[..., i]),
            create_graph=True,
            retain_graph=True,
            materialize_grads=True,
        )[0]
        for i in range(y.shape[-1])
    ]
    return torch.stack(rows, dim=-2)


def _checkOutputs(owner: str, y: torch.Tensor, count: int) -> None:
    if y.shape[-1] != count:
        raise ValueError(f"{owner}: expected {count} model outputs, got {y.shape[-1]}")


class BurgersPDEProducer(PDETransformProducer):
    """Невязкое уравнение Бюргерса `u_t + u u_x = 0`. Выход модели `(u,)`."""

    @override
    def pde(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        _checkOutputs(type(self).__name__, y, 1)
        d = jacobian(y, x)
        return d[..., 0, 0:1] + y * d[..., 0, 1:2]

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "equations": ["burgers"]}


class TransportPDEProducer(PDETransformProducer):
    """Невязка переноса с переменной скоростью `u_t + cos(pi x) u_x = 0`. Выход `(u,)`."""

    @override
    def pde(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        _checkOutputs(type(self).__name__, y, 1)
        d = jacobian(y, x)
        return d[..., 0, 0:1] + torch.cos(torch.pi * x[..., 1:2]) * d[..., 0, 1:2]

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "equations": ["transport"]}


class ShallowWaterPDEProducer(PDETransformProducer):
    """Невязки одномерных уравнений мелкой воды в консервативных переменных.

    Выходы модели имеют вид `(h, q)`, где `q = h u`.

        h_t + q_x = 0
        q_t + 2 u q_x + (g h - u^2) h_x = 0     (= q_t + (q^2/h + g h^2/2)_x)
    """

    def __init__(self, gravity: float = 10.0):
        if gravity <= 0:
            raise ValueError(f"gravity must be positive, got {gravity}")
        self._gravity = gravity

    @override
    def pde(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        _checkOutputs(type(self).__name__, y, 2)
        d = jacobian(y, x)
        h, q = y[..., 0], y[..., 1]
        hT, hX = d[..., 0, 0], d[..., 0, 1]
        qT, qX = d[..., 1, 0], d[..., 1, 1]
        u = q / h
        mass = hT + qX
        momentum = qT + 2.0 * u * qX + (self._gravity * h - u.square()) * hX
        return torch.stack([mass, momentum], dim=-1)

    @override
    def getMeta(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "equations": ["mass", "momentum"],
            "gravity": self._gravity,
        }


class PCPWPDEProducer(PDETransformProducer):
    """Невязки системы PCPW для выходов `(H, q)`.

        H_t + fluxCoefficient q q_x = 0
        q_t + heightCoefficient H^2 H_x + momentumCoefficient q^2 / H^2 = 0

    Перед возведением в квадрат `H` ограничивается снизу значением `heightFloor`.
    У необученной модели `H` может быть близко к нулю, и множитель `1/H^2`
    приведёт к резкому росту градиентов. При нормировке `(H, q)` масштабами
    `(H0, q0)` задайте коэффициенты следующим образом.

        fluxCoefficient = q0^2/H0
        heightCoefficient = heightCoefficient H0^3/q0
        momentumCoefficient = momentumCoefficient q0/H0^2
    """

    def __init__(
        self,
        fluxCoefficient: float = 1.0,
        momentumCoefficient: float = 1.0,
        heightCoefficient: float = 5.0,
        heightFloor: float = 1e-3,
    ):
        if heightFloor <= 0:
            raise ValueError(f"heightFloor must be positive, got {heightFloor}")
        self._fluxCoefficient = fluxCoefficient
        self._momentumCoefficient = momentumCoefficient
        self._heightCoefficient = heightCoefficient
        self._heightFloor = heightFloor

    @override
    def pde(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        _checkOutputs(type(self).__name__, y, 2)
        d = jacobian(y, x)
        q = y[..., 1]
        h2 = y[..., 0].clamp_min(self._heightFloor).square()
        hT, hX = d[..., 0, 0], d[..., 0, 1]
        qT, qX = d[..., 1, 0], d[..., 1, 1]
        height = hT + self._fluxCoefficient * q * qX
        momentum = qT + self._heightCoefficient * h2 * hX + self._momentumCoefficient * q * q / h2
        return torch.stack([height, momentum], dim=-1)

    @override
    def getMeta(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "equations": ["height", "momentum"],
            "fluxCoefficient": self._fluxCoefficient,
            "momentumCoefficient": self._momentumCoefficient,
            "heightCoefficient": self._heightCoefficient,
            "heightFloor": self._heightFloor,
        }
