import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, override

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure

from pinn_platform.operation._operation import OpArg, Operation

log = logging.getLogger(__name__)


def _save(figure: Figure, title: str, outputPath: Path | None) -> None:
    if outputPath is not None:
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(outputPath)
        log.info(f"[plot] {title}: {outputPath.resolve()}")


def _finish(figure: Figure, title: str, show: bool, outputPath: Path | None) -> None:
    _save(figure, title, outputPath)
    if show:
        plt.show()
    plt.close(figure)


def _readSeries(inputs: OpArg) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Возвращает на CPU вход `domain` и значения из остальных входов."""
    domain = inputs["domain"].value.detach().cpu()
    series = {
        name: artifact.value.detach().cpu() for name, artifact in inputs.items() if name != "domain"
    }
    return domain, series


def _checkSeries(owner: str, domain: torch.Tensor, series: dict[str, torch.Tensor]) -> int:
    """Проверяет значения на соответствие точкам области и возвращает число выходов модели."""
    if not series:
        raise ValueError(f"{owner}: no values to draw, only a domain")
    for name, values in series.items():
        if values.shape[:-1] != domain.shape[:-1]:
            raise ValueError(
                f"{owner}: {name!r} has shape {tuple(values.shape)}, "
                f"domain points are {tuple(domain.shape[:-1])}"
            )
    outputCounts = {values.shape[-1] for values in series.values()}
    if len(outputCounts) != 1:
        raise ValueError(f"{owner}: inputs have different output counts {outputCounts}")
    return outputCounts.pop()


def _names(
    owner: str,
    axisNames: tuple[str, ...] | None,
    outputNames: tuple[str, ...] | None,
    axisCount: int,
    outputCount: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Проверяет число имён и возвращает заданные имена или `x0, x1, ...` и `u0, u1, ...`."""
    axes = axisNames or tuple(f"x{i}" for i in range(axisCount))
    outputs = outputNames or tuple(f"u{k}" for k in range(outputCount))
    if len(axes) != axisCount or len(outputs) != outputCount:
        raise ValueError(
            f"{owner}: got axisNames {axes} for {axisCount} coordinates "
            f"and outputNames {outputs} for {outputCount} outputs"
        )
    return axes, outputs


class LossVisualizer(Operation):
    """Строит графики истории потерь, обычно из выхода `history` тренера `ModelTrainer`.

    Каждый вход содержит историю в виде списка потерь. Несколько историй,
    например для стадий `warmup → adam → lbfgs`, рисуются в порядке входов.
    Каждая кривая начинается там, где закончилась предыдущая, и подписывается
    именем соответствующего входа.
    """

    def __init__(
        self,
        title: str = "loss",
        show: bool = True,
        outputPath: str | Path | None = None,
        logScale: bool = True,
    ):
        self._title = title
        self._show = show
        self._outputPath = Path(outputPath) if outputPath is not None else None
        self._logScale = logScale

    @override
    def run(self, inputs: OpArg) -> OpArg:
        histories = {name: list(artifact.value) for name, artifact in inputs.items()}
        if not any(histories.values()):
            log.warning(f"[{type(self).__name__}] {self._title}: all histories are empty")
            return {}

        figure, axis = plt.subplots(figsize=(8, 4.5))
        start = 0
        for name, history in histories.items():
            axis.plot(range(start + 1, start + len(history) + 1), history, label=name)
            start += len(history)
        if self._logScale:
            # PINN losses fall by orders of magnitude; a linear axis would show only the start.
            axis.set_yscale("log")
        axis.set(xlabel="step", ylabel="loss", title=self._title)
        axis.grid(True, which="both", alpha=0.3)
        if len(histories) > 1:
            axis.legend()
        figure.tight_layout()
        _finish(figure, self._title, self._show, self._outputPath)
        return {}

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "title": self._title}


class SolutionVisualizer(Operation):
    """Визуализирует значения на области.

    Вход `domain` хранит точки с координатами по последней оси. Остальные
    входы хранят значения в этих точках с выходами модели по последней оси,
    например `values` из `ModelEvaluator`. На одной фигуре можно сравнить
    несколько моделей. Подписи соответствуют именам входов.

    - Одномерная область. Отдельный график для каждого выхода и кривая для каждого входа.
    - Двумерная сетка формы `(n, m, 2)`. Тепловые карты по строкам для входов
      и по столбцам для выходов. У каждого выхода своя общая цветовая шкала.
    - Набор двумерных точек формы `(N, 2)`. Та же раскладка с диаграммами рассеяния.
    """

    def __init__(
        self,
        title: str = "solution",
        axisNames: Sequence[str] | None = None,
        outputNames: Sequence[str] | None = None,
        show: bool = True,
        outputPath: str | Path | None = None,
    ):
        self._title = title
        self._axisNames = tuple(axisNames) if axisNames is not None else None
        self._outputNames = tuple(outputNames) if outputNames is not None else None
        self._show = show
        self._outputPath = Path(outputPath) if outputPath is not None else None

    @override
    def run(self, inputs: OpArg) -> OpArg:
        owner = type(self).__name__
        domain, series = _readSeries(inputs)
        outputCount = _checkSeries(owner, domain, series)
        if domain.shape[-1] not in (1, 2):
            raise ValueError(f"{owner}: only 1D and 2D domains, got {domain.shape[-1]}D")
        axisNames, outputNames = _names(
            owner, self._axisNames, self._outputNames, domain.shape[-1], outputCount
        )

        if domain.shape[-1] == 1:
            figure = self._plotCurves(domain, series, axisNames, outputNames)
        else:
            figure = self._plotMaps(domain, series, axisNames, outputNames)
        figure.suptitle(self._title)
        _finish(figure, self._title, self._show, self._outputPath)
        return {}

    @staticmethod
    def _plotCurves(
        domain: torch.Tensor,
        series: dict[str, torch.Tensor],
        axisNames: tuple[str, ...],
        outputNames: tuple[str, ...],
    ) -> Figure:
        x = domain.reshape(-1)
        order = x.argsort()
        figure, axes = plt.subplots(
            1,
            len(outputNames),
            figsize=(5 * len(outputNames), 4),
            squeeze=False,
            layout="constrained",  # unlike tight_layout, leaves room for the suptitle
        )
        for k, outputName in enumerate(outputNames):
            axis = axes[0, k]
            for name, values in series.items():
                axis.plot(x[order], values.reshape(-1, len(outputNames))[order, k], label=name)
            axis.set(xlabel=axisNames[0], ylabel=outputName)
            axis.grid(True, alpha=0.3)
            if len(series) > 1:
                axis.legend()
        return figure

    @staticmethod
    def _plotMaps(
        domain: torch.Tensor,
        series: dict[str, torch.Tensor],
        axisNames: tuple[str, ...],
        outputNames: tuple[str, ...],
    ) -> Figure:
        isGrid = domain.dim() == 3
        figure, axes = plt.subplots(
            len(series),
            len(outputNames),
            figsize=(5 * len(outputNames), 4 * len(series)),
            squeeze=False,
            layout="constrained",
        )
        for k, outputName in enumerate(outputNames):
            # One color scale per output, so the inputs (models) can be compared by color.
            low = min(values[..., k].min().item() for values in series.values())
            high = max(values[..., k].max().item() for values in series.values())
            for row, (name, values) in enumerate(series.items()):
                axis = axes[row, k]
                if isGrid:
                    mappable = axis.pcolormesh(
                        domain[..., 0],
                        domain[..., 1],
                        values[..., k],
                        shading="auto",
                        vmin=low,
                        vmax=high,
                    )
                else:
                    mappable = axis.scatter(
                        domain[..., 0],
                        domain[..., 1],
                        c=values[..., k],
                        s=8,
                        vmin=low,
                        vmax=high,
                    )
                axis.set(
                    xlabel=axisNames[0],
                    ylabel=axisNames[1],
                    title=outputName if len(series) == 1 else f"{name}: {outputName}",
                )
            figure.colorbar(mappable, ax=axes[:, k], label=outputName)
        return figure

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "title": self._title}


class TimeSliceVisualizer(Operation):
    """Строит срезы `u(x)` двумерной сетки в выбранные моменты времени.

    Входы совпадают с `SolutionVisualizer`. Вход `domain` должен содержать сетку
    формы `(nt, nx, 2)`, построенную как в `DomainRectProducer` (вызов `meshgrid`
    с `indexing="ij"`). Временная координата `timeAxis` меняется вдоль одноимённой
    оси сетки. Для каждого запрошенного времени выбирается ближайшая линия сетки,
    а в заголовке подграфика указывается её время.

    Для каждого выхода создаётся отдельная фигура. На ней каждому времени
    соответствует подграфик с кривой для каждого входа. При нескольких выходах
    имя выхода добавляется к `outputPath`, например `slices.png → slices_h.png`.
    """

    def __init__(
        self,
        times: Sequence[float],
        title: str = "slices",
        timeAxis: int = 0,
        axisNames: Sequence[str] | None = None,
        outputNames: Sequence[str] | None = None,
        columns: int = 3,
        show: bool = True,
        outputPath: str | Path | None = None,
    ):
        if not times:
            raise ValueError("times must not be empty")
        if timeAxis not in (0, 1):
            raise ValueError(f"timeAxis must be 0 or 1, got {timeAxis}")
        if columns <= 0:
            raise ValueError(f"columns must be positive, got {columns}")
        self._times = tuple(times)
        self._title = title
        self._timeAxis = timeAxis
        self._axisNames = tuple(axisNames) if axisNames is not None else None
        self._outputNames = tuple(outputNames) if outputNames is not None else None
        self._columns = columns
        self._show = show
        self._outputPath = Path(outputPath) if outputPath is not None else None

    @override
    def run(self, inputs: OpArg) -> OpArg:
        owner = type(self).__name__
        domain, series = _readSeries(inputs)
        outputCount = _checkSeries(owner, domain, series)
        if domain.dim() != 3 or domain.shape[-1] != 2:
            raise ValueError(
                f"{owner}: domain must be a grid (nt, nx, 2), got {tuple(domain.shape)}"
            )
        axisNames, outputNames = _names(owner, self._axisNames, self._outputNames, 2, outputCount)
        timeName, spaceName = axisNames[self._timeAxis], axisNames[1 - self._timeAxis]

        if self._timeAxis == 1:
            # Bring time to grid axis 0 and coordinate 0.
            domain = domain.transpose(0, 1)[..., [1, 0]]
            series = {name: values.transpose(0, 1) for name, values in series.items()}
        gridTimes, x = self._gridAxes(owner, domain)
        indices = [self._nearest(owner, gridTimes, time) for time in self._times]

        figures = [
            self._plot(k, outputName, indices, gridTimes, x, series, timeName, spaceName)
            for k, outputName in enumerate(outputNames)
        ]
        for figure, outputName in zip(figures, outputNames, strict=True):
            _save(figure, self._title, self._pathFor(outputName, outputCount))
        if self._show:
            plt.show()
        for figure in figures:
            plt.close(figure)
        return {}

    @staticmethod
    def _gridAxes(owner: str, domain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gridTimes, x = domain[:, 0, 0], domain[0, :, 1]
        if not (
            torch.equal(domain[..., 0], gridTimes[:, None].expand_as(domain[..., 0]))
            and torch.equal(domain[..., 1], x[None, :].expand_as(domain[..., 1]))
        ):
            raise ValueError(
                f"{owner}: domain is not a time x space grid; build it with "
                "torch.meshgrid(..., indexing='ij') as DomainRectProducer does"
            )
        return gridTimes, x

    @staticmethod
    def _nearest(owner: str, gridTimes: torch.Tensor, time: float) -> int:
        low, high = gridTimes.min().item(), gridTimes.max().item()
        if not low <= time <= high:
            raise ValueError(f"{owner}: time {time} is outside the domain [{low}, {high}]")
        return int((gridTimes - time).abs().argmin().item())

    def _plot(
        self,
        k: int,
        outputName: str,
        indices: list[int],
        gridTimes: torch.Tensor,
        x: torch.Tensor,
        series: dict[str, torch.Tensor],
        timeName: str,
        spaceName: str,
    ) -> Figure:
        columns = min(self._columns, len(indices))
        rows = (len(indices) + columns - 1) // columns
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4 * columns, 3 * rows),
            squeeze=False,
            sharex=True,
            sharey=True,  # a common scale shows how the solution changes in time
            layout="constrained",
        )
        for axis, index in zip(axes.flat, indices, strict=False):
            for name, values in series.items():
                axis.plot(x, values[index, :, k], label=name)
            axis.set(title=f"{timeName}={gridTimes[index].item():g}", xlabel=spaceName)
            axis.grid(True, alpha=0.3)
        for axis in axes[:, 0]:
            axis.set_ylabel(outputName)
        for axis in list(axes.flat)[len(indices) :]:
            axis.set_visible(False)
        if len(series) > 1:
            axes[0, 0].legend()
        figure.suptitle(f"{self._title}: {outputName}")
        return figure

    def _pathFor(self, outputName: str, outputCount: int) -> Path | None:
        if self._outputPath is None or outputCount == 1:
            return self._outputPath
        path = self._outputPath
        return path.with_name(f"{path.stem}_{outputName}{path.suffix}")

    @override
    def getMeta(self) -> dict[str, Any]:
        return {"name": type(self).__name__, "title": self._title, "times": self._times}
