"""Слои KAN на полиномах Чебышёва первого рода.

Каждое ребро от входа `i` к выходу `o` имеет свою обучаемую функцию
`sum_k c[i, o, k] * T_k(z_i)`. Значение выхода равно сумме по входящим рёбрам.
Полиномы вычисляются на `[-1, 1]`. При заданных границах входа слой
аффинно переводит `[low, high]` в `[-1, 1]`. При отсутствующих границах
слой применяет `tanh`.
"""

import torch
from torch import nn

from ..initial.initialization import (
    KANInit,
    glorot_std,
    lecun_std,
    power_law_std,
    uniform_samples,
    uniform_variance,
    value_and_derivative,
)


def chebyshev_basis(z: torch.Tensor, degree: int) -> torch.Tensor:
    """Собирает `T_0(z) .. T_degree(z)` по новой последней оси.

    Использует рекурсию `T_{k+1} = 2 z T_k - T_{k-1}`, устойчивую на `[-1, 1]`
    по сравнению с вычислением через степенной базис.
    """
    if degree < 0:
        raise ValueError("Chebyshev degree must be non-negative")
    terms = [torch.ones_like(z)]
    if degree >= 1:
        terms.append(z)
    for _ in range(2, degree + 1):
        terms.append(2.0 * z * terms[-1] - terms[-2])
    return torch.stack(terms, dim=-1)


class ChebyKANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        degree: int,
        input_boundary: torch.Tensor | None = None,
        *,
        init: KANInit | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if degree < 0:
            raise ValueError("Chebyshev degree must be non-negative")
        self._degree = degree
        self._squash = input_boundary is None
        if input_boundary is None:
            boundary = torch.tensor([-1.0, 1.0], dtype=dtype, device=device)[:, None].expand(
                2, in_features
            )
        else:
            boundary = torch.as_tensor(input_boundary, dtype=dtype, device=device)
            if boundary.shape != (2, in_features):
                raise ValueError(
                    f"input_boundary must have shape (2, {in_features}), "
                    f"got {tuple(boundary.shape)}"
                )
            if (boundary[0] >= boundary[1]).any().item():
                raise ValueError("input_boundary needs min < max for every input")
        self.register_buffer("_low", boundary[0].clone())
        self.register_buffer("_high", boundary[1].clone())
        self.coefficients = nn.Parameter(
            torch.empty(
                in_features, out_features, degree + 1, dtype=dtype, device=device
            )
        )
        self._initialize_parameters(init if init is not None else KANInit())

    def _initialize_parameters(self, init: KANInit) -> None:
        in_features, _, components = self.coefficients.shape
        with torch.no_grad():
            if init.scheme == "default":
                self.coefficients.copy_(
                    torch.randn_like(self.coefficients) / (in_features * components)
                )
                return
            if init.scheme == "baseline":
                std = (1 / (in_features * components)) ** 0.5
            elif init.scheme == "power_law":
                std = power_law_std(in_features, components, init.beta)
            else:
                std = self._moment_std(init)[:, None, None]
            self.coefficients.copy_(torch.randn_like(self.coefficients) * std)

    def _moment_std(self, init: KANInit) -> torch.Tensor:
        in_features, out_features, components = self.coefficients.shape
        basis, derivative = value_and_derivative(
            lambda x: chebyshev_basis(self._to_unit_interval(x), self._degree),
            uniform_samples(self._low, self._high),
        )
        moment = basis.square().mean(dim=(0, -1))
        if init.scheme == "lecun":
            return lecun_std(
                uniform_variance(self._low, self._high), in_features, components, moment
            )
        return glorot_std(
            in_features,
            out_features,
            components,
            moment,
            derivative.square().mean(dim=(0, -1)),
        )

    def _to_unit_interval(self, x: torch.Tensor) -> torch.Tensor:
        if self._squash:
            return torch.tanh(x)
        return (2.0 * x - (self._low + self._high)) / (self._high - self._low)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = chebyshev_basis(self._to_unit_interval(x), self._degree)
        return torch.einsum("...ik,iok->...o", basis, self.coefficients)
