from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StorageBSplines:
    local_coefficients: torch.Tensor


def bsplines_storage(
    nodes_shape: torch.Size,
    degree: int,
    dtype=torch.float64,
    device=None,
) -> StorageBSplines:

    def produce(*size):
        return torch.empty(*size, dtype=dtype, device=device)

    device = device if device is not None else torch.device("cpu")

    # On every knot interval only degree + 1 basis splines are non-zero.
    # The last dimension includes one zero interval on either side.
    local_coefficients = produce(
        torch.Size((degree + 1,))
        + nodes_shape[1:]
        + torch.Size((degree + 1, nodes_shape[0] + 1))
    )

    return StorageBSplines(local_coefficients=local_coefficients)
