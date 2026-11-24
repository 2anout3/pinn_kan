from collections.abc import Sequence
from typing import override

import torch
from torch import nn

from pinn_networks.kan.cheby.cheby import ChebyKANLayer
from pinn_networks.kan.initial.initialization import KANInit
from pinn_networks.network import NeuralNetwork


class ChebyKANModel(NeuralNetwork):
    """KAN из слоёв `ChebyKANLayer`.

    Первый слой получает `inputBoundary` и приводит входы к `[-1, 1]`.
    Скрытые слои применяют `tanh` к своим входам.
    """

    def __init__(
        self,
        features: Sequence[int],
        degree: int,
        *,
        init: KANInit | None = None,
        inputBoundary: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        widths = tuple(features)
        if len(widths) < 2 or any(width <= 0 for width in widths):
            raise ValueError(f"features must contain at least 2 positive widths, got {widths}")
        self._layers = nn.ModuleList(
            ChebyKANLayer(
                widths[index],
                widths[index + 1],
                degree,
                inputBoundary if index == 0 else None,
                init=init,
                dtype=dtype,
                device=device,
            )
            for index in range(len(widths) - 1)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            x = layer(x)
        return x

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, index: int) -> nn.Module:
        return self._layers[index]
