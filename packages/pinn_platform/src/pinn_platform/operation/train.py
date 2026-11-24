import logging
from abc import abstractmethod
from typing import Any, Literal, override

import torch
from torch import nn, optim

from pinn_networks import NeuralNetwork
from pinn_platform.core.artifact import Artifact
from pinn_platform.operation._operation import OpArg, Operation
from pinn_platform.operation.condition import Condition, requireCondition

log = logging.getLogger(__name__)


class ModelTrainer(Operation):
    """Обучает входную модель `model` на входном условии `condition`.

    Условие имеет тип `Condition`. Его может создать, например,
    `WeightedAccConditionProducer`. Тренер создаёт оптимизатор в `run`
    по параметрам входной модели и возвращает его на выходе `optimizer`.
    Состояние оптимизатора можно проверить или сохранить.
    """

    def __init__(self, modelName: str | None = None) -> None:
        self._modelName = modelName

    @property
    def logOwner(self) -> str:
        trainer = type(self).__name__
        return trainer if self._modelName is None else f"{self._modelName}/{trainer}"

    @abstractmethod
    def createOptimizer(self, model: nn.Module) -> optim.Optimizer: ...

    @abstractmethod
    def train(
        self, model: nn.Module, optimizer: optim.Optimizer, condition: Condition
    ) -> list[float]:
        """Выполняет оптимизацию и возвращает историю потерь."""

    @override
    def run(self, inputs: OpArg) -> OpArg:
        model = inputs["model"].value
        condition = requireCondition(inputs["condition"].value, type(self).__name__, "condition")
        optimizer = self.createOptimizer(model)

        model.train()
        history = self.train(model, optimizer, condition)
        model.eval()

        meta = self.getMeta()
        return {
            "model": Artifact(value=model, meta=inputs["model"].meta | meta),
            "history": Artifact(value=history, meta={"kind": "history"} | meta),
            "optimizer": Artifact(value=optimizer, meta={"kind": "optimizer"} | meta),
        }

    @override
    def getMeta(self) -> dict[str, Any]:
        meta = {"name": type(self).__name__}
        if self._modelName is not None:
            meta["modelName"] = self._modelName
        return meta


def _beforeOptim(model: nn.Module, step: int, optimizer: optim.Optimizer) -> None:
    if isinstance(model, NeuralNetwork):
        model.beforeOptim(step, optimizer)


def _afterOptim(model: nn.Module, step: int, optimizer: optim.Optimizer) -> None:
    if isinstance(model, NeuralNetwork):
        model.afterOptim(step, optimizer)


def _checkStopping(lossThreshold: float | None, logInterval: int) -> None:
    if lossThreshold is not None and lossThreshold <= 0:
        raise ValueError(f"lossThreshold must be positive or None, got {lossThreshold}")
    if logInterval < 0:
        raise ValueError(f"logInterval must be non-negative, got {logInterval}")


def _reached(loss: float, lossThreshold: float | None) -> bool:
    return lossThreshold is not None and loss <= lossThreshold


def _logStep(
    owner: str, step: int, loss: float, losses: dict[str, torch.Tensor], logInterval: int
) -> None:
    """Записывает в лог общие потери и взвешенные потери по каждому условию."""
    if logInterval <= 0 or step % logInterval != 0:
        return
    # .item() only here: converting every condition on every step would sync the GPU.
    parts = "".join(f" {name}={value.item():.6e}" for name, value in losses.items())
    log.info(f"[{owner}] step={step:5d} loss={loss:.6e}{parts}")


class StepModelTrainer(ModelTrainer):
    """Выполняет `epochs` шагов оптимизации без `closure`.

    Подклассы выбирают оптимизатор (Adam, AdamW, SGD и другие) в
    `createOptimizer`. Обучение завершается досрочно, когда потери
    на шаге достигают `lossThreshold` или опускаются ниже.
    """

    def __init__(
        self,
        epochs: int,
        lossThreshold: float | None = None,
        logInterval: int = 100,
        modelName: str | None = None,
    ):
        super().__init__(modelName)
        if epochs < 0:
            raise ValueError(f"epochs must be non-negative, got {epochs}")
        _checkStopping(lossThreshold, logInterval)
        self._epochs = epochs
        self._lossThreshold = lossThreshold
        self._logInterval = logInterval

    @override
    def train(
        self, model: nn.Module, optimizer: optim.Optimizer, condition: Condition
    ) -> list[float]:
        owner = self.logOwner
        history: list[float] = []
        for step in range(1, self._epochs + 1):
            _beforeOptim(model, step, optimizer)
            optimizer.zero_grad()
            loss, losses = condition.evaluate(model)
            loss.backward()
            optimizer.step()
            _afterOptim(model, step, optimizer)

            history.append(loss.item())
            _logStep(owner, step, history[-1], losses, self._logInterval)
            if _reached(history[-1], self._lossThreshold):
                log.info(f"[{owner}] step={step:5d} reached threshold={self._lossThreshold:.6e}")
                break
        return history

    @override
    def getMeta(self) -> dict[str, Any]:
        return super().getMeta() | {"epochs": self._epochs, "lossThreshold": self._lossThreshold}


class AdamModelTrainer(StepModelTrainer):
    def __init__(
        self,
        epochs: int,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weightDecay: float = 0.0,
        amsgrad: bool = False,
        decoupledWeightDecay: bool = False,
        lossThreshold: float | None = None,
        logInterval: int = 100,
        modelName: str | None = None,
    ):
        super().__init__(epochs, lossThreshold, logInterval, modelName)
        self._params = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weightDecay,
            "amsgrad": amsgrad,
            "decoupled_weight_decay": decoupledWeightDecay,
        }

    @override
    def createOptimizer(self, model: nn.Module) -> optim.Adam:
        return optim.Adam(model.parameters(), **self._params)

    @override
    def getMeta(self) -> dict[str, Any]:
        return super().getMeta() | {"optimizer": {"name": "Adam"} | self._params}


class _ThresholdReached(Exception):
    pass


class LBFGSModelTrainer(ModelTrainer):
    """Выполняет один вызов `optimizer.step(closure)` для L-BFGS с числом итераций `maxIter`.

    История содержит потери каждого вызова `closure`, включая вычисления при
    линейном поиске. Обучение завершается досрочно, когда потери одного вызова
    достигают `lossThreshold` или опускаются ниже.
    """

    def __init__(
        self,
        lr: float = 1.0,
        maxIter: int = 20,
        maxEval: int | None = None,
        toleranceGrad: float = 1e-7,
        toleranceChange: float = 1e-9,
        historySize: int = 100,
        lineSearchFn: Literal["strong_wolfe"] | None = None,
        lossThreshold: float | None = None,
        logInterval: int = 100,
        modelName: str | None = None,
    ):
        super().__init__(modelName)
        _checkStopping(lossThreshold, logInterval)
        self._lossThreshold = lossThreshold
        self._logInterval = logInterval
        self._params = {
            "lr": lr,
            "max_iter": maxIter,
            "max_eval": maxEval,
            "tolerance_grad": toleranceGrad,
            "tolerance_change": toleranceChange,
            "history_size": historySize,
            "line_search_fn": lineSearchFn,
        }

    @override
    def createOptimizer(self, model: nn.Module) -> optim.LBFGS:
        return optim.LBFGS(model.parameters(), **self._params)

    @override
    def train(
        self, model: nn.Module, optimizer: optim.Optimizer, condition: Condition
    ) -> list[float]:
        owner = self.logOwner
        history: list[float] = []

        def closure() -> float:
            optimizer.zero_grad()
            loss, losses = condition.evaluate(model)
            loss.backward()
            history.append(loss.item())
            _logStep(owner, len(history), history[-1], losses, self._logInterval)
            if _reached(history[-1], self._lossThreshold):
                raise _ThresholdReached
            # LBFGS only needs the value: it converts the result with float() anyway.
            return history[-1]

        _beforeOptim(model, 1, optimizer)
        try:
            optimizer.step(closure)
        except _ThresholdReached:
            log.info(
                f"[{owner}] evaluation={len(history):5d} "
                f"reached threshold={self._lossThreshold:.6e}"
            )
        _afterOptim(model, 1, optimizer)
        return history

    @override
    def getMeta(self) -> dict[str, Any]:
        return super().getMeta() | {
            "lossThreshold": self._lossThreshold,
            "optimizer": {"name": "LBFGS"} | self._params,
        }
