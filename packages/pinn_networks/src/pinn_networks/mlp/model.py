from collections.abc import Sequence

import torch
from torch import nn

from pinn_networks.network import NeuralNetwork


class MLP(nn.Sequential, NeuralNetwork):
    """Полносвязная сеть для экспериментов PINN с MLP."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: int = 10,
        hidden_width: int = 64,
        *,
        hidden_sizes: Sequence[int] | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
    ) -> None:
        if input_size <= 0 or output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        if hidden_sizes is None:
            if hidden_layers <= 0 or hidden_width <= 0:
                raise ValueError("hidden_layers and hidden_width must be positive")
            widths = (hidden_width,) * hidden_layers
        else:
            widths = tuple(hidden_sizes)
            if not widths or any(width <= 0 for width in widths):
                raise ValueError("hidden_sizes must contain positive values")

        # Xavier's uniform bound is derived for a unit-gain activation; tanh
        # scales its input's variance down, so hidden layers need the tanh
        # gain (5/3) to keep activations from shrinking layer over layer,
        # which otherwise starves gradients in deep stacks like the default
        # 10-layer network. The output layer has no activation after it, so
        # it keeps the default unit gain.
        tanh_gain = nn.init.calculate_gain("tanh")

        layers: list[nn.Module] = []
        previous_width = input_size
        for width in widths:
            linear = nn.Linear(previous_width, width, dtype=dtype, device=device)
            nn.init.xavier_uniform_(linear.weight, gain=tanh_gain)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.Tanh())
            previous_width = width

        output_layer = nn.Linear(previous_width, output_size, dtype=dtype, device=device)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        super().__init__(*layers)
