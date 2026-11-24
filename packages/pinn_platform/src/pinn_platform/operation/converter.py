import logging
from abc import abstractmethod
from typing import Any, override

import torch

from pinn_platform.core.artifact import Artifact
from pinn_platform.operation._operation import OpArg, Operation

log = logging.getLogger(__name__)

type Sample = torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]


def _length(data: Sample) -> int:
    tensors = (data,) if isinstance(data, torch.Tensor) else tuple(data)
    lengths = {len(tensor) for tensor in tensors}
    if len(lengths) != 1:
        raise ValueError(f"data tensors must share their first dimension, got {sorted(lengths)}")
    return lengths.pop()


def _take(data: Sample, indices: torch.Tensor) -> Sample:
    if isinstance(data, torch.Tensor):
        return data[indices.to(data.device)]
    return type(data)(tensor[indices.to(tensor.device)] for tensor in data)


class FuncDataConverter(Operation):
    """Применяет функцию из входа `transform` к данным из входа `data`."""

    @override
    def run(self, inputs: OpArg) -> OpArg:
        source = inputs["data"]
        transform = inputs["transform"].value
        return {"data": Artifact(value=transform(source.value), meta=source.meta | self.getMeta())}

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__}


class RandomDataConverter(Operation):
    """Случайно отбирает строки по оси 0 из тензора, кортежа или списка тензоров.

    Для связанных данных вроде `(points, values)` берёт строки с одними и теми же
    индексами из каждого тензора.
    """

    def __init__(self, generator: torch.Generator):
        self._generator = generator

    @abstractmethod
    def _sampleSize(self, length: int) -> int: ...

    @override
    def run(self, inputs: OpArg) -> OpArg:
        source = inputs["data"]
        return {
            "data": Artifact(value=self._select(source.value), meta=source.meta | self.getMeta())
        }

    def _select(self, data: Sample) -> Sample:
        length = _length(data)
        count = self._sampleSize(length)
        if count >= length:
            return data
        indices = torch.randperm(length, generator=self._generator, device=self._generator.device)
        return _take(data, indices[:count].sort().values)

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__}


class RandomCountDataConverter(RandomDataConverter):
    def __init__(self, generator: torch.Generator, count: int):
        super().__init__(generator)
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        self._count = count

    @override
    def _sampleSize(self, length: int) -> int:
        if self._count > length:
            log.warning(
                f"{type(self).__name__}: count={self._count} > len(data)={length}, "
                "taking the whole data"
            )
        return self._count

    @override
    def getMeta(self) -> dict[str, Any]:
        return super().getMeta() | {"count": self._count}


class RandomFractionDataConverter(RandomDataConverter):
    def __init__(self, generator: torch.Generator, fraction: float):
        super().__init__(generator)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must lie in [0, 1], got {fraction}")
        self._fraction = fraction

    @override
    def _sampleSize(self, length: int) -> int:
        return round(self._fraction * length)

    @override
    def getMeta(self) -> dict[str, Any]:
        return super().getMeta() | {"fraction": self._fraction}
