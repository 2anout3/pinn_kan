import logging
from typing import override

import torch
from torch import nn, optim

from pinn_networks.kan.bsplines.layer import BsplineKANLayer
from pinn_networks.network import NeuralNetwork

log = logging.getLogger(__name__)


def _evenlySpaced(points: torch.Tensor, count: int) -> torch.Tensor:
    if count >= len(points):
        return points
    indices = torch.linspace(0, len(points) - 1, count, device=points.device).round().long()
    return points[indices]


class KANModel(NeuralNetwork):
    """KAN на B-сплайновых слоях с обновлением сеток во время обучения.

    Сетки обновляются в `afterOptim` на шагах `gridUpdateEvery`,
    `2 * gridUpdateEvery` и так далее до `gridUpdateUntil`. Значение
    `gridUpdateEvery=0` отключает обновления. Модель ведёт общий счёт шагов по
    всем стадиям (warmup, Adam, L-BFGS и другим) и использует его в хуках.
    Счётчик хранится в буфере и входит в `state_dict`, поэтому при возобновлении
    обучения из чекпойнта расписание обновлений сохраняется.

    На шаге обновления модель собирает точки всех вызовов между `beforeOptim`
    и `afterOptim`. Она равномерно прореживает их до `gridPointsCount` и строит
    новую сетку. Помимо точек вычисления потерь учитываются и другие вызовы
    `model(x)` внутри шага, например для логирования.
    """

    def __init__(
        self,
        *layers: BsplineKANLayer,
        gridMaxRelativeError: float = 2e-2,
        gridMaxPointwiseError: float = 1e-1,
        gridMaxJacobianRelativeError: float = 5e-2,
        gridMaxJacobianPointwiseError: float = 2e-1,
        gridUpdateEvery: int = 0,
        gridUpdateUntil: int = 0,
        gridPointsCount: int = 500,
    ) -> None:
        super().__init__()
        if gridUpdateEvery < 0 or gridUpdateUntil < 0:
            raise ValueError("gridUpdateEvery and gridUpdateUntil must be non-negative")
        if gridPointsCount <= 0:
            raise ValueError(f"gridPointsCount must be positive, got {gridPointsCount}")
        self._layers = nn.ModuleList(layers)
        self._gridMaxRelativeError = gridMaxRelativeError
        self._gridMaxPointwiseError = gridMaxPointwiseError
        self._gridMaxJacobianRelativeError = gridMaxJacobianRelativeError
        self._gridMaxJacobianPointwiseError = gridMaxJacobianPointwiseError
        self._lastGridUpdateMetrics: dict[str, float | str] = {"status": "not_run"}
        self._gridUpdateEvery = gridUpdateEvery
        self._gridUpdateUntil = gridUpdateUntil
        self._gridPointsCount = gridPointsCount
        self.register_buffer("_optimStep", torch.zeros((), dtype=torch.long))
        # Inputs of forward during the current update step; None when not recording.
        self._recordedInputs: list[torch.Tensor] | None = None

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._recordedInputs is not None and not any(
            x is seen for seen in self._recordedInputs
        ):
            # Identity check: LBFGS calls the closure, and so forward, many times per step.
            self._recordedInputs.append(x)
        for layer in self._layers:
            x = layer(x)
        return x

    @property
    def optimStep(self) -> int:
        """Число шагов оптимизации модели по всем тренерам."""
        return int(self._optimStep.item())

    def _isGridUpdateStep(self, step: int) -> bool:
        return (
            self._gridUpdateEvery > 0
            and step <= self._gridUpdateUntil
            and step % self._gridUpdateEvery == 0
        )

    @override
    def beforeOptim(self, step: int, optimizer: optim.Optimizer) -> None:
        self._optimStep.add_(1)
        if self.training and self._isGridUpdateStep(self.optimStep):
            self._recordedInputs = []

    @override
    def afterOptim(self, step: int, optimizer: optim.Optimizer) -> None:
        if self._recordedInputs is None:
            return
        recorded, self._recordedInputs = self._recordedInputs, None
        if not recorded:
            log.warning(f"[KAN][grid] step={self.optimStep}: no inputs recorded, skipping")
            return
        points = torch.cat([inputs.detach() for inputs in recorded])
        updated = self.updateGrid(_evenlySpaced(points, self._gridPointsCount))
        for parameter in updated:
            # Adam moments belong to the coefficients on the old grid.
            optimizer.state.pop(parameter, None)
        self._logGridUpdate(len(updated))

    def _logGridUpdate(self, updatedCount: int) -> None:
        metrics = self._lastGridUpdateMetrics
        detail = "".join(
            f" {name}={metrics[name]:.6e}"
            for name in ("relativeL2", "relativeMax", "jacobianRelativeL2", "jacobianRelativeMax")
            if name in metrics
        )
        log.info(
            f"[KAN][grid] step={self.optimStep:5d} status={metrics['status']} "
            f"updated_layers={updatedCount}{detail}"
        )

    def valueAndJacobian(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = x
        jacobian = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device).expand(x.shape[0], -1, -1)
        for layer in self._layers:
            values, localJacobian = layer.forward_with_jacobian(values)
            jacobian = torch.bmm(localJacobian, jacobian)
        return values, jacobian

    @property
    def lastGridUpdateMetrics(self) -> dict[str, float | str]:
        return self._lastGridUpdateMetrics

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, index: int) -> nn.Module:
        return self._layers[index]

    def updateGrid(self, points: torch.Tensor, *, debug: bool = False) -> tuple[nn.Parameter, ...]:
        updatedParameters, _ = self.updateGridDetailed(points, debug=debug)
        return updatedParameters

    def updateGridDetailed(
        self, points: torch.Tensor, *, debug: bool = False
    ) -> tuple[tuple[nn.Parameter, ...], tuple[str, ...]]:
        updatedParameters: list[nn.Parameter] = []
        statuses: list[str] = []
        self._lastGridUpdateMetrics = {"status": "unchanged"}
        with torch.no_grad():
            predictionBefore, jacobianBefore = self.valueAndJacobian(points)
            snapshots: list[tuple[BsplineKANLayer, torch.Tensor, torch.Tensor]] = []
            values = points
            for index, layer in enumerate(self._layers, start=1):
                oldGrid = layer._grid.clone()
                oldCoef = layer._coef.clone()
                if layer.update_grid(values, debug_label=str(index), debug=debug):
                    updatedParameters.append(layer.coefficient_parameter)
                    snapshots.append((layer, oldGrid, oldCoef))
                statuses.append(layer.last_grid_update_status)
                values = layer(values)

            if updatedParameters:
                predictionAfter, jacobianAfter = self.valueAndJacobian(points)
                difference = predictionAfter - predictionBefore
                eps = torch.finfo(predictionBefore.dtype).eps
                relativeL2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
                    predictionBefore
                ).clamp_min(eps)
                relativeMax = difference.abs().max() / predictionBefore.abs().max().clamp_min(eps)
                jacobianDifference = jacobianAfter - jacobianBefore
                jacobianRelativeL2 = torch.linalg.vector_norm(
                    jacobianDifference
                ) / torch.linalg.vector_norm(jacobianBefore).clamp_min(eps)
                jacobianRelativeMax = (
                    jacobianDifference.abs().max() / jacobianBefore.abs().max().clamp_min(eps)
                )
                jacobianRejected = bool(
                    not bool(torch.isfinite(jacobianAfter).all())
                    or jacobianRelativeL2.item() > self._gridMaxJacobianRelativeError
                    or jacobianRelativeMax.item() > self._gridMaxJacobianPointwiseError
                )
                modelRejected = bool(
                    not bool(torch.isfinite(predictionAfter).all())
                    or relativeL2.item() > self._gridMaxRelativeError
                    or relativeMax.item() > self._gridMaxPointwiseError
                    or jacobianRejected
                )
                self._lastGridUpdateMetrics = {
                    "status": "model_rejected" if modelRejected else "accepted",
                    "relativeL2": relativeL2.item(),
                    "relativeMax": relativeMax.item(),
                    "jacobianRelativeL2": jacobianRelativeL2.item(),
                    "jacobianRelativeMax": jacobianRelativeMax.item(),
                }
                if modelRejected:
                    for layer, oldGrid, oldCoef in reversed(snapshots):
                        layer._grid.copy_(oldGrid)
                        layer._coef.copy_(oldCoef)
                        layer._rebuild_bspline()
                        layer._last_grid_update_status = "model_rejected"
                    statuses = [
                        "model_rejected" if status == "accepted" else status for status in statuses
                    ]
                    updatedParameters.clear()
                    if debug:
                        log.info(
                            "[KAN][grid] status=model_rejected "
                            f"relative_l2={relativeL2.item():.6e} "
                            f"relative_max={relativeMax.item():.6e} "
                            f"jacobian_relative_l2={jacobianRelativeL2.item():.6e} "
                            f"jacobian_relative_max={jacobianRelativeMax.item():.6e}"
                        )
        return tuple(updatedParameters), tuple(statuses)
