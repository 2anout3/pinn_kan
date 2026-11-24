"""Фабрики операций для узлов пайплайна. Каждая создаёт операцию по своей конфигурации.

Декоратор `@hydra_config(name=..., group=...)` строит структурированную схему
по аргументам функции. Hydra проверяет по ней опцию `<group>/<name>.yaml`
(см. `write_config_tree`). Вызов `hydra.utils.instantiate(cfg)` выбирает
фабрику по конфигурации, и один сборщик поддерживает все опции группы.

Фабрики возвращают операции. Имена узлов и привязки входов задают структуру
графа эксперимента. Функция `build.place` создаёт узел из операции, имени и привязок.

Для фиксированных частей графа сборщик напрямую вызывает фабрики без декоратора.
К таким частям относятся разбиение области, источники выборок, PDE-условие и чекпойнты.

Параметры `dtype` и `device` области и моделей исключены из их конфигов через
`exclude`. Сборщик передаёт общие значения, чтобы данные и модели использовали
одно устройство и одну точность.

Опции каждой группы взаимозаменяемы по входам и выходам. Фиксированные фабрики
описывают свои входы и выходы отдельно.

    domain     → "domain"                      (+ dtype, device от сборщика)
    pde        → "pde"
    initial    "domain" (источник точек), "loss" → "condition"
    boundary   "domain" (источник пар границ), "loss" → "condition"
    sample     "data" → "data"
    loss       → "loss"
    condition  условия по именам → "condition"
    model      → "model"                       (+ dtype, device от сборщика)
    trainer    "model", "condition" → "model", "history", "optimizer"
    evaluator  "domain", "model" → "values"
    plot       см. операции визуализации
"""

import functools
from typing import Literal

import torch

from experiments.pinn.ops import (
    DomainParts,
    FactoryModelProducer,
    PeriodicEdgesConditionProducer,
    SampleSource,
    TargetConditionProducer,
)
from pinn_networks.kan.bsplines.layer import BsplineKANLayer
from pinn_networks.kan.cheby import ChebyKANModel
from pinn_networks.kan.initial.initialization import KANInit
from pinn_networks.kan.model import KANModel
from pinn_networks.mlp import MLP
from pinn_platform.conf import hydra_config
from pinn_platform.core import Operation
from pinn_platform.operation import (
    AdamModelTrainer,
    BurgersPDEProducer,
    DomainRectProducer,
    LBFGSModelTrainer,
    LoadModelCheckpoint,
    LossVisualizer,
    ModelEvaluator,
    MSELossProducer,
    PCPWPDEProducer,
    PDEConditionProducer,
    RandomCountDataConverter,
    RandomFractionDataConverter,
    SaveModelCheckpoint,
    ShallowWaterPDEProducer,
    SolutionVisualizer,
    TimeSliceVisualizer,
    TransportPDEProducer,
    WeightedAccConditionProducer,
)

type InitScheme = Literal["default", "baseline", "lecun", "glorot", "power_law"]


# --------------------------------------------------------------------------- #
# domain
# --------------------------------------------------------------------------- #
@hydra_config(name="rect", group="domain", exclude=("dtype", "device"))
def rectDomain(
    start: list[float],
    end: list[float],
    steps: list[int],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Operation:
    """Сетка на отрезках `[start, end]` с числом точек `steps` по осям `(t, x)`."""
    if not len(start) == len(end) == len(steps):
        raise ValueError(f"start, end, steps differ in length: {start}, {end}, {steps}")
    return DomainRectProducer(start, end, steps, dtype, device)


def domainParts() -> Operation:
    """Разбивает вход `domain` на `initial`, `edges` и `inner` (см. `DomainParts`)."""
    return DomainParts()


def sampleSource() -> Operation:
    """Преобразует вход `data` в вызываемый источник точек `source`."""
    return SampleSource()


# --------------------------------------------------------------------------- #
# pde
# --------------------------------------------------------------------------- #
@hydra_config(name="burgers", group="pde")
def burgersPDE() -> Operation:
    return BurgersPDEProducer()


@hydra_config(name="transport", group="pde")
def transportPDE() -> Operation:
    return TransportPDEProducer()


@hydra_config(name="shallow_water", group="pde")
def shallowWaterPDE(gravity: float = 10.0) -> Operation:
    return ShallowWaterPDEProducer(gravity)


@hydra_config(name="pcpw", group="pde")
def pcpwPDE(
    fluxCoefficient: float = 1.0,
    momentumCoefficient: float = 1.0,
    heightCoefficient: float = 5.0,
    heightFloor: float = 1e-3,
) -> Operation:
    return PCPWPDEProducer(fluxCoefficient, momentumCoefficient, heightCoefficient, heightFloor)


def pdeCondition() -> Operation:
    """Создаёт `condition` из `domain` (источника точек коллокации), `pde` и `loss`."""
    return PDEConditionProducer()


# --------------------------------------------------------------------------- #
# initial / boundary targets: points (n, 2) with columns (t, x) -> values (n, outputs)
# --------------------------------------------------------------------------- #
def _x(points: torch.Tensor) -> torch.Tensor:
    return points[:, 1:2]


def _sine(points: torch.Tensor) -> torch.Tensor:
    return -torch.sin(torch.pi * _x(points))


def _cos(points: torch.Tensor) -> torch.Tensor:
    return -torch.cos(torch.pi * _x(points))


def _hat(points: torch.Tensor) -> torch.Tensor:
    x = _x(points)
    return torch.relu(torch.minimum(x - 1.0, 5.0 - x))


def _step(points: torch.Tensor) -> torch.Tensor:
    x = _x(points)
    return torch.where((x >= 1.0) & (x <= 2.0), 2.0, 0.5).to(x)


def _constant(points: torch.Tensor, value: float) -> torch.Tensor:
    return torch.full_like(_x(points), value)


def _shallowWave(
    points: torch.Tensor, gravity: float, spaceStart: float, spaceEnd: float
) -> torch.Tensor:
    """Высота и расход гладкой периодической волны. `u = 2 sin(2 pi x / L + pi / 4)`,
    `h = (u + 10)^2 / 4g`.
    """
    length = spaceEnd - spaceStart
    u = 2.0 * torch.sin(2.0 * torch.pi * _x(points) / length + torch.pi / 4.0)
    h = (u + 10.0).square() / (4.0 * gravity)
    return torch.cat([h, u * h], dim=-1)


def _damBreak(
    points: torch.Tensor, position: float, heightLeft: float, heightRight: float
) -> torch.Tensor:
    """Высота `h` и расход `q` воды в покое со скачком высоты в точке `position`."""
    x = _x(points)
    h = torch.where(x < position, heightLeft, heightRight).to(x)
    return torch.cat([h, torch.zeros_like(h)], dim=-1)


def _hydraulicJump(
    points: torch.Tensor, gravity: float, position: float, heightLeft: float, heightRight: float
) -> torch.Tensor:
    x = _x(points)
    h = torch.where(x < position, heightLeft, heightRight).to(x)
    q_value = gravity * heightLeft * heightRight * (heightLeft + heightRight) / 2
    q_value = q_value ** (1 / 2)
    q = torch.full_like(h, q_value)
    return torch.cat([h, q], dim=-1)


def _target(target, label: str) -> Operation:
    return TargetConditionProducer(target, label)


@hydra_config(name="sine", group="initial")
def sineInitial() -> Operation:
    """Синусоидальное начальное условие `u(0, x) = -sin(pi x)`."""
    return _target(_sine, "sine")


@hydra_config(name="cos", group="initial")
def cosInitial() -> Operation:
    return _target(_cos, "cos")


@hydra_config(name="hat", group="initial")
def hatInitial() -> Operation:
    """Треугольный начальный профиль `u(0, x) = max(0, min(x - 1, 5 - x))`."""
    return _target(_hat, "hat")


@hydra_config(name="step", group="initial")
def stepInitial() -> Operation:
    """Ступенчатый начальный профиль. `u(0, x) = 2` на `[1, 2]` и `0.5` вне отрезка."""
    return _target(_step, "step")


@hydra_config(name="shallow_wave", group="initial")
def shallowWaveInitial(
    gravity: float = 10.0, spaceStart: float = 0.0, spaceEnd: float = 10.0
) -> Operation:
    target = functools.partial(
        _shallowWave, gravity=gravity, spaceStart=spaceStart, spaceEnd=spaceEnd
    )
    return _target(target, "shallow_wave")


@hydra_config(name="dam_break", group="initial")
def damBreakInitial(
    position: float = 1.0, heightLeft: float = 1.5, heightRight: float = 0.5
) -> Operation:
    target = functools.partial(
        _damBreak, position=position, heightLeft=heightLeft, heightRight=heightRight
    )
    return _target(target, "dam_break")


@hydra_config(name="hydraulic_jump", group="initial")
def hydraulicJumpInitial(
    gravity: float = 10, position: float = 0.0, heightLeft: float = 2.0, heightRight: float = 1.0
) -> Operation:
    target = functools.partial(
        _hydraulicJump,
        gravity=gravity,
        position=position,
        heightLeft=heightLeft,
        heightRight=heightRight,
    )
    return _target(target, "hydralic_jump")


@hydra_config(name="constant", group="boundary")
def constantBoundary(value: float = 0.0) -> Operation:
    """Условие Дирихле `u = value` на обеих пространственных границах."""
    return _target(functools.partial(_constant, value=value), f"constant({value})")


@hydra_config(name="dam_break", group="boundary")
def damBreakBoundary(
    position: float = 1.0, heightLeft: float = 1.5, heightRight: float = 0.5
) -> Operation:
    """Условие Дирихле сохраняет начальные значения задачи разрушения плотины на обоих краях."""
    target = functools.partial(
        _damBreak, position=position, heightLeft=heightLeft, heightRight=heightRight
    )
    return _target(target, "dam_break")


@hydra_config(name="hydraulic_jump", group="boundary")
def hydraulicJumpBoundary(
    gravity: float = 10, position: float = 0.0, heightLeft: float = 2.0, heightRight: float = 1.0
) -> Operation:
    target = functools.partial(
        _hydraulicJump,
        gravity=gravity,
        position=position,
        heightLeft=heightLeft,
        heightRight=heightRight,
    )
    return _target(target, "hydralic_jump")


@hydra_config(name="periodic", group="boundary")
def periodicBoundary() -> Operation:
    """Периодическое условие `u(t, left) = u(t, right)`."""
    return PeriodicEdgesConditionProducer()


# --------------------------------------------------------------------------- #
# sampling, loss, conditions
# --------------------------------------------------------------------------- #
@hydra_config(name="random", group="sample")
def randomSample(count: int | None = None, seed: int = 0) -> Operation:
    """Выбирает `count` случайных строк данных. При `None` берёт все строки."""
    generator = torch.Generator().manual_seed(seed)
    if count is None:
        return RandomFractionDataConverter(generator, 1.0)
    return RandomCountDataConverter(generator, count)


@hydra_config(name="validate_random", group="sample")
def validateSample(count: int | None = None, seed: int = 0) -> Operation:
    """Выбирает случайные строки с отдельным начальным значением генератора для проверки."""
    return randomSample(count, seed)


@hydra_config(name="mse", group="loss")
def mseLoss() -> Operation:
    return MSELossProducer()


@hydra_config(name="weighted", group="condition")
def weightedCondition(weights: dict[str, float]) -> Operation:
    """Взвешенная сумма условий из `weights` (`initial`, `boundary`, `pde`)."""
    return WeightedAccConditionProducer(weights)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def _boundary(
    inputBoundary: list[list[float]] | None, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    """Преобразует `[[mins...], [maxs...]]` в тензор формы `(2, inputs)` для слоёв KAN."""
    if inputBoundary is None:
        return None
    boundary = torch.tensor(inputBoundary, dtype=dtype, device=device)
    if boundary.dim() != 2 or boundary.shape[0] != 2 or (boundary[0] >= boundary[1]).any():
        raise ValueError(f"inputBoundary must be [[mins...], [maxs...]], got {inputBoundary}")
    return boundary


def _bsplineKAN(
    features: list[int],
    intervalsCount: int,
    degree: int,
    gridUpdateRegularization: float,
    gridUpdateMaxRelativeError: float,
    gridUpdateMaxPointwiseError: float,
    modelGridMaxRelativeError: float,
    modelGridMaxPointwiseError: float,
    modelGridMaxJacobianRelativeError: float,
    modelGridMaxJacobianPointwiseError: float,
    init: KANInit,
    inputBoundary: torch.Tensor | None,
    gridUpdateEvery: int,
    gridUpdateUntil: int,
    gridPointsCount: int,
    dtype: torch.dtype,
    device: torch.device,
) -> KANModel:
    layers = [
        BsplineKANLayer(
            features[index],
            features[index + 1],
            intervalsCount,
            degree,
            initial_boundary=inputBoundary if index == 0 else None,
            grid_update_regularization=gridUpdateRegularization,
            grid_update_max_relative_error=gridUpdateMaxRelativeError,
            grid_update_max_pointwise_error=gridUpdateMaxPointwiseError,
            init=init,
            dtype=dtype,
            device=device,
        )
        for index in range(len(features) - 1)
    ]
    return KANModel(
        *layers,
        gridMaxRelativeError=modelGridMaxRelativeError,
        gridMaxPointwiseError=modelGridMaxPointwiseError,
        gridMaxJacobianRelativeError=modelGridMaxJacobianRelativeError,
        gridMaxJacobianPointwiseError=modelGridMaxJacobianPointwiseError,
        gridUpdateEvery=gridUpdateEvery,
        gridUpdateUntil=gridUpdateUntil,
        gridPointsCount=gridPointsCount,
    )


@hydra_config(name="kan_bsplines", group="model", exclude=("dtype", "device"))
def bsplineKANModel(
    features: list[int],
    intervalsCount: int = 10,
    degree: int = 7,
    gridUpdateRegularization: float = 1e-3,
    gridUpdateMaxRelativeError: float = 2e-1,
    gridUpdateMaxPointwiseError: float = 5e-1,
    modelGridMaxRelativeError: float = 2e-1,
    modelGridMaxPointwiseError: float = 1e-1,
    modelGridMaxJacobianRelativeError: float = 5e-1,
    modelGridMaxJacobianPointwiseError: float = 2e-1,
    initScheme: InitScheme = "power_law",
    initAlpha: float = 0.25,
    initBeta: float = 1.0,
    inputBoundary: list[list[float]] | None = None,
    gridUpdateEvery: int = 0,
    gridUpdateUntil: int = 0,
    gridPointsCount: int = 500,
    seed: int = 0,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Operation:
    """KAN на B-сплайнах обновляет сетки по данным каждые `gridUpdateEvery` шагов
    до шага `gridUpdateUntil`. Шаги считаются по всем тренерам."""
    factory = functools.partial(
        _bsplineKAN,
        features,
        intervalsCount,
        degree,
        gridUpdateRegularization,
        gridUpdateMaxRelativeError,
        gridUpdateMaxPointwiseError,
        modelGridMaxRelativeError,
        modelGridMaxPointwiseError,
        modelGridMaxJacobianRelativeError,
        modelGridMaxJacobianPointwiseError,
        KANInit(initScheme, initAlpha, initBeta),
        _boundary(inputBoundary, dtype, device),
        gridUpdateEvery,
        gridUpdateUntil,
        gridPointsCount,
        dtype,
        device,
    )
    return FactoryModelProducer(factory, seed, device, {"kind": "kan_bsplines"})


@hydra_config(name="chebykan", group="model", exclude=("dtype", "device"))
def chebyKANModel(
    features: list[int],
    degree: int = 5,
    initScheme: InitScheme = "power_law",
    initAlpha: float = 0.25,
    initBeta: float = 1.25,
    inputBoundary: list[list[float]] | None = None,
    seed: int = 0,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Operation:
    factory = functools.partial(
        ChebyKANModel,
        features,
        degree,
        init=KANInit(initScheme, initAlpha, initBeta),
        inputBoundary=_boundary(inputBoundary, dtype, device),
        dtype=dtype,
        device=device,
    )
    return FactoryModelProducer(factory, seed, device, {"kind": "chebykan"})


@hydra_config(name="mlp", group="model", exclude=("dtype", "device"))
def mlpModel(
    features: list[int],
    seed: int = 0,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Operation:
    if len(features) < 3:
        raise ValueError(f"features must have an input, hidden and output width: {features}")
    factory = functools.partial(
        MLP, features[0], features[-1], hidden_sizes=features[1:-1], dtype=dtype, device=device
    )
    return FactoryModelProducer(factory, seed, device, {"kind": "mlp"})


# --------------------------------------------------------------------------- #
# trainers
# --------------------------------------------------------------------------- #
@hydra_config(name="adam", group="trainer", exclude=("modelName",))
def adamTrainer(
    epochs: int,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weightDecay: float = 0.0,
    amsgrad: bool = False,
    lossThreshold: float | None = None,
    logInterval: int = 100,
    modelName: str | None = None,
) -> Operation:
    return AdamModelTrainer(
        epochs,
        lr=lr,
        betas=betas,
        eps=eps,
        weightDecay=weightDecay,
        amsgrad=amsgrad,
        lossThreshold=lossThreshold,
        logInterval=logInterval,
        modelName=modelName,
    )


@hydra_config(name="lbfgs", group="trainer", exclude=("modelName",))
def lbfgsTrainer(
    maxIter: int,
    lr: float = 1.0,
    maxEval: int | None = None,
    toleranceGrad: float = 1e-9,
    toleranceChange: float = 1e-12,
    historySize: int = 100,
    lineSearchFn: Literal["strong_wolfe"] | None = "strong_wolfe",
    lossThreshold: float | None = None,
    logInterval: int = 100,
    modelName: str | None = None,
) -> Operation:
    return LBFGSModelTrainer(
        lr=lr,
        maxIter=maxIter,
        maxEval=maxEval,
        toleranceGrad=toleranceGrad,
        toleranceChange=toleranceChange,
        historySize=historySize,
        lineSearchFn=lineSearchFn,
        lossThreshold=lossThreshold,
        logInterval=logInterval,
        modelName=modelName,
    )


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def saveCheckpoint(path: str) -> Operation:
    """Передаёт входную модель `model` на одноимённый выход после записи `state_dict` в `path`."""
    return SaveModelCheckpoint(path)


def loadCheckpoint(path: str, strict: bool = True) -> Operation:
    """Передаёт модель `model` на одноимённый выход после загрузки весов из `path`."""
    return LoadModelCheckpoint(path, strict)


# --------------------------------------------------------------------------- #
# evaluation and plots
# --------------------------------------------------------------------------- #
@hydra_config(name="grid", group="evaluator")
def gridEvaluator(batchSize: int | None = 65536) -> Operation:
    return ModelEvaluator(batchSize)


@hydra_config(name="loss", group="plot")
def lossPlot(
    title: str = "loss",
    show: bool = False,
    outputPath: str | None = None,
    logScale: bool = True,
) -> Operation:
    return LossVisualizer(title, show, outputPath, logScale)


@hydra_config(name="solution", group="plot")
def solutionPlot(
    title: str = "solution",
    axisNames: list[str] | None = None,
    outputNames: list[str] | None = None,
    show: bool = False,
    outputPath: str | None = None,
) -> Operation:
    return SolutionVisualizer(title, axisNames, outputNames, show, outputPath)


@hydra_config(name="slices", group="plot")
def slicesPlot(
    times: list[float],
    title: str = "slices",
    axisNames: list[str] | None = None,
    outputNames: list[str] | None = None,
    columns: int = 3,
    show: bool = False,
    outputPath: str | None = None,
) -> Operation:
    return TimeSliceVisualizer(
        times,
        title=title,
        axisNames=axisNames,
        outputNames=outputNames,
        columns=columns,
        show=show,
        outputPath=outputPath,
    )
