from abc import ABC, abstractmethod

import torch
from torch import nn, optim


class NeuralNetwork(nn.Module, ABC):
    """Базовый класс нейронных сетей для обучения.

    Помимо `forward`, объявляет два хука, которые тренер вызывает до и после
    каждого шага оптимизации. Нумерация шагов `step` начинается с 1.

        model.beforeOptim(step, optimizer)
        optimizer.zero_grad(); loss.backward(); optimizer.step()  # или step(closure)
        model.afterOptim(step, optimizer)

    По умолчанию хуки пусты. Модель может переопределить их, например для
    обновления сетки KAN во время обучения.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def beforeOptim(self, step: int, optimizer: optim.Optimizer) -> None:  # noqa: B027
        """Вызывается перед шагом оптимизатора `step`."""

    def afterOptim(self, step: int, optimizer: optim.Optimizer) -> None:  # noqa: B027
        """Вызывается после шага оптимизатора `step`."""
