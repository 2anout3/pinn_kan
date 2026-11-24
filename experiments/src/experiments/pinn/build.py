"""Строит пайплайн PINN из итоговой конфигурации Hydra.

    root/data/       область → её части → выборки → источники тензоров
    root/problem/    PDE, функция потерь, начальное, граничное и PDE-условия
    root/stages/     по одному взвешенному условию на стадию обучения
    root/models/<m>/ модель → тренеры стадий → сохранение → оценка, график потерь
                     модель → загрузка → оценка для моделей из checkpoint.load
    root/plots/      общие фигуры решения и срезов для всех моделей

Операции для узлов создают фабрики из `experiments.pinn.nodes`.
Для опций группы, например `model: mlp`, сборщик вызывает `instantiate`.
Для секций с известной фабрикой сборщик вызывает её `from_config`.
Функция `place` создаёт узел из операции.
"""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from experiments.pinn import nodes
from pinn_platform.core import NodeInput, Operation, Path, Pipeline, PipelineNode

log = logging.getLogger(__name__)

ROOT = Path.parse("root")


def place(pipe: Pipeline, name: str, operation: Operation, **inputs: NodeInput) -> PipelineNode:
    """Создаёт узел `name` из `operation` в текущей группе `pipe` и подключает его входы.

    Фабрики возвращают операции. Здесь узел получает имя и привязки входов
    в формате `{локальное имя: (узел-источник, имя выхода)}`. Каждая привязка
    задаёт зависимость узла от источника.
    """
    node = PipelineNode(operation, dict(inputs), name=name)
    pipe.add([node])
    for source in dict.fromkeys(source for source, _ in inputs.values()):
        pipe.connect(source, [node])
    return node


def make(cfg: DictConfig, **runtime: Any) -> Operation:
    """Создаёт операцию из конфига. `runtime` задаёт параметры вне конфига (`dtype`, `device`)."""
    operation = instantiate(cfg, **runtime)
    if not isinstance(operation, Operation):
        raise TypeError(
            f"config {cfg.get('_target_')} created {type(operation).__name__}, not an Operation"
        )
    return operation


@dataclass
class Built:
    """Пайплайн и узлы, чьи результаты вызывающий код читает после запуска."""

    env: dict[str, Any]
    pipe: Pipeline
    models: dict[str, PipelineNode] = field(default_factory=dict)
    trainers: dict[str, dict[str, PipelineNode]] = field(default_factory=dict)
    loaded: dict[str, str] = field(default_factory=dict)  # model label -> checkpoint path


def replace(built: Built, oldNode: PipelineNode, operation: Operation) -> PipelineNode:
    """Меняет операцию и обновляет ссылки на узел. Перед запуском нужен `commit()`."""
    node = PipelineNode(operation, oldNode.inputs.copy())
    built.pipe.replaceNode(oldNode, node, keepIncomingEdges=True)
    for registry in (built.models, *built.trainers.values()):
        for name, registered in registry.items():
            if registered is oldNode:
                registry[name] = node
    return node


def environment(cfg: DictConfig) -> dict[str, Any]:
    """Возвращает общие `dtype` и `device` для области и моделей эксперимента."""
    dtype = getattr(torch, cfg.dtype, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown dtype {cfg.dtype!r}, expected e.g. float32 or float64")
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device is cuda, but CUDA is not available; use device=cpu")
    return {"dtype": dtype, "device": device}


def setup(cfg: DictConfig) -> None:
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.manual_seed(cfg.seed)


def seedTraining(cfg: DictConfig, label: str, seed: int) -> None:
    """Задаёт глобальный сид и выводит из него сиды модели и обучающих выборок."""
    cfg.seed = seed
    generator = random.Random(seed)
    for section in (cfg.models[label], *cfg.samples.values()):
        if "seed" in section:
            section.seed = generator.getrandbits(32)


def _plotPath(cfg: DictConfig, name: str) -> str | None:
    return None if cfg.plotDir is None else os.path.join(cfg.plotDir, f"{name}.png")


def _checkpointPath(cfg: DictConfig, label: str) -> str:
    return os.path.join(cfg.checkpoint.dir, f"{label}.pt")


def _validate(cfg: DictConfig) -> None:
    """Проверяет ссылки между разделами конфигурации перед сборкой."""
    unknown = [stage for stage in cfg.schedule if stage not in cfg.stages]
    if unknown:
        raise ValueError(f"schedule refers to stages {unknown}; defined: {list(cfg.stages)}")
    if not cfg.models:
        raise ValueError("no models: select them with model@models.<label>=<option>")
    unknown = [label for label in cfg.checkpoint.load if label not in cfg.models]
    if unknown:
        raise ValueError(f"checkpoint.load names unknown models {unknown}: {list(cfg.models)}")


def _addData(
    pipe: Pipeline, cfg: DictConfig, env: dict[str, Any]
) -> tuple[PipelineNode, dict[str, PipelineNode]]:
    """Создаёт сетку области, выборки и источники тензоров в `root/data`.

    Возвращает узел области и узлы источников по условиям `initial`, `boundary` и `pde`.
    """
    with pipe.group(ROOT / "data"):
        domain = place(pipe, "domain", make(cfg.domain, **env))
        parts = place(pipe, "parts", nodes.domainParts(), domain=(domain, "domain"))
        samplers = {name: make(cfg.samples[name]) for name in ("initial", "boundary", "pde")}
        sources = placeSampleSources(pipe, parts, samplers)
    return domain, sources


def placeSampleSources(
    pipe: Pipeline, parts: PipelineNode, samplers: dict[str, Operation]
) -> dict[str, PipelineNode]:
    """Создаёт выборки из трёх частей области и их вызываемые источники."""
    samples = {
        name: place(pipe, name, samplers[name], data=(parts, part))
        for name, part in (("initial", "initial"), ("boundary", "edges"), ("pde", "inner"))
    }
    return {
        name: place(pipe, f"{name}_source", nodes.sampleSource(), data=(sample, "data"))
        for name, sample in samples.items()
    }


def _addConditions(
    pipe: Pipeline, cfg: DictConfig, sources: dict[str, PipelineNode]
) -> dict[str, PipelineNode]:
    """Создаёт PDE, функцию потерь и условия `initial`, `boundary` и `pde` в `root/problem`."""
    with pipe.group(ROOT / "problem"):
        pde = place(pipe, "pde", make(cfg.pde))
        loss = place(pipe, "loss", make(cfg.loss))
        return placeConditions(pipe, cfg, sources, pde, loss)


def placeConditions(
    pipe: Pipeline,
    cfg: DictConfig,
    sources: dict[str, PipelineNode],
    pde: PipelineNode,
    loss: PipelineNode,
) -> dict[str, PipelineNode]:
    """Подключает условия задачи к переданным источникам точек."""
    return {
        "initial": place(
            pipe,
            "initial",
            make(cfg.initial),
            domain=(sources["initial"], "source"),
            loss=(loss, "loss"),
        ),
        "boundary": place(
            pipe,
            "boundary",
            make(cfg.boundary),
            domain=(sources["boundary"], "source"),
            loss=(loss, "loss"),
        ),
        "pde": place(
            pipe,
            "pde_condition",
            nodes.pdeCondition(),
            domain=(sources["pde"], "source"),
            pde=(pde, "pde"),
            loss=(loss, "loss"),
        ),
    }


def _addStages(
    pipe: Pipeline, cfg: DictConfig, conditions: dict[str, PipelineNode]
) -> dict[str, PipelineNode]:
    """Создаёт взвешенное условие для каждой стадии в `root/stages`.

    В него входят условия, перечисленные в весах стадии. Все модели используют общие стадии."""
    stages: dict[str, PipelineNode] = {}
    with pipe.group(ROOT / "stages"):
        for stage in cfg.schedule:
            weights = cfg.stages[stage].condition.weights
            missing = [name for name in weights if name not in conditions]
            if missing:
                raise ValueError(
                    f"stage {stage!r} weights unknown conditions {missing}; "
                    f"known: {list(conditions)}"
                )
            stages[stage] = place(
                pipe,
                stage,
                make(cfg.stages[stage].condition),
                **{name: (conditions[name], "condition") for name in weights},
            )
    return stages


def _addTraining(
    pipe: Pipeline,
    cfg: DictConfig,
    label: str,
    model: NodeInput,
    stages: dict[str, PipelineNode],
) -> tuple[NodeInput, dict[str, PipelineNode]]:
    """Создаёт цепочку тренеров по расписанию и график их потерь.

    Каждый тренер получает модель предыдущей стадии. Возвращает ссылку на
    обученную модель и узлы тренеров по стадиям. Размещает их в группе текущей модели.
    """
    trainers: dict[str, PipelineNode] = {}
    for stage in cfg.schedule:
        trainers[stage] = place(
            pipe,
            stage,
            make(cfg.stages[stage].trainer, modelName=label),
            model=model,
            condition=(stages[stage], "condition"),
        )
        model = (trainers[stage], "model")

    if trainers:
        place(
            pipe,
            "loss_plot",
            nodes.lossPlot.from_config(
                cfg.plots.loss, title=f"{label}: loss", outputPath=_plotPath(cfg, f"{label}_loss")
            ),
            **{stage: (trainer, "history") for stage, trainer in trainers.items()},
        )
    return model, trainers


def _addModel(
    built: Built,
    cfg: DictConfig,
    label: str,
    env: dict[str, Any],
    domain: PipelineNode,
    stages: dict[str, PipelineNode],
) -> NodeInput:
    """Создаёт ветку модели в `root/models/<label>` и возвращает ссылку на значения оценки.

    После создания модели загружает веса или обучает модель и сохраняет веса
    при включённом сохранении. Затем вычисляет решение на сетке области."""
    pipe = built.pipe
    path = _checkpointPath(cfg, label)
    with pipe.group(ROOT / "models" / label):
        model = place(pipe, "model", make(cfg.models[label], **env))
        built.models[label] = model
        built.trainers[label] = {}

        if label in cfg.checkpoint.load:
            # the model is created by its config and gets the saved weights; no training
            load = nodes.loadCheckpoint(path, cfg.checkpoint.strict)
            current: NodeInput = (place(pipe, "load", load, model=(model, "model")), "model")
            built.loaded[label] = path
        else:
            current, built.trainers[label] = _addTraining(
                pipe, cfg, label, (model, "model"), stages
            )
            if cfg.checkpoint.save:
                current = (place(pipe, "save", nodes.saveCheckpoint(path), model=current), "model")

        evaluation = place(
            pipe, "evaluation", make(cfg.evaluator), domain=(domain, "domain"), model=current
        )
    return (evaluation, "values")


def _addPlots(
    pipe: Pipeline, cfg: DictConfig, domain: PipelineNode, values: dict[str, NodeInput]
) -> None:
    """Создаёт в `root/plots` общие фигуры решения и временных срезов для всех моделей."""
    with pipe.group(ROOT / "plots"):
        for name, factory in (("solution", nodes.solutionPlot), ("slices", nodes.slicesPlot)):
            place(
                pipe,
                name,
                factory.from_config(cfg.plots[name], outputPath=_plotPath(cfg, name)),
                domain=(domain, "domain"),
                **values,
            )


def buildPipeline(cfg: DictConfig) -> Built:
    _validate(cfg)
    env = environment(cfg)
    built = Built(env, Pipeline())

    domain, sources = _addData(built.pipe, cfg, env)
    conditions = _addConditions(built.pipe, cfg, sources)
    stages = _addStages(built.pipe, cfg, conditions)
    values = {label: _addModel(built, cfg, label, env, domain, stages) for label in cfg.models}
    _addPlots(built.pipe, cfg, domain, values)
    return built


def report(built: Built) -> None:
    for label, model in built.models.items():
        network = model.results["model"].value
        count = sum(p.numel() for p in network.parameters() if p.requires_grad)
        log.info(f"{label}: parameters={count}")
        if label in built.loaded:
            log.info(f"{label}: loaded from {built.loaded[label]}")
        for stage, trainer in built.trainers[label].items():
            history = trainer.results["history"].value
            final = f"{history[-1]:.6e}" if history else "-"
            log.info(f"{label}/{stage}: steps={len(history)} loss={final}")
