from collections.abc import Callable
from dataclasses import dataclass

import torch

SCHEMES = ("default", "baseline", "lecun", "glorot", "power_law")
MOMENT_SAMPLE_COUNT = 4096


@dataclass(frozen=True)
class KANInit:
    scheme: str = "default"
    alpha: float = 0.25
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError(f"unknown KAN init scheme {self.scheme!r}; expected one of {SCHEMES}")


def uniform_samples(
    low: torch.Tensor, high: torch.Tensor, count: int = MOMENT_SAMPLE_COUNT
) -> torch.Tensor:
    alpha = (torch.arange(count, dtype=low.dtype, device=low.device) + 0.5) / count
    return low + (high - low) * alpha[:, None]


def uniform_variance(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return (high - low).square() / 12


def value_and_derivative(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        return torch.func.jvp(fn, (x,), (torch.ones_like(x),))


def _positive(moment: torch.Tensor) -> torch.Tensor:
    return moment.clamp_min(torch.finfo(moment.dtype).eps)


def lecun_std(
    variance: torch.Tensor, fan_in: int, components: int, moment: torch.Tensor
) -> torch.Tensor:
    return (variance / (fan_in * components * _positive(moment))).sqrt()


def glorot_std(
    fan_in: int,
    fan_out: int,
    components: int,
    moment: torch.Tensor,
    derivative_moment: torch.Tensor,
) -> torch.Tensor:
    return (
        2 / (components * (fan_in * _positive(moment) + fan_out * _positive(derivative_moment)))
    ).sqrt()


def power_law_std(fan_in: int, components: int, exponent: float) -> float:
    return (fan_in * components) ** -exponent
