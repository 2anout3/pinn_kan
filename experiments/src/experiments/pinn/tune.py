"""Подбирает архитектуру PINN, сид и бюджеты обучения по проверочным потерям."""

import copy
import logging
from typing import override

import hydra
import optuna
import torch
from omegaconf import DictConfig, open_dict

from experiments.pinn import nodes
from experiments.pinn.__main__ import pickDevice as pickDevice
from experiments.pinn.build import (
    ROOT,
    Built,
    buildPipeline,
    make,
    place,
    placeConditions,
    placeSampleSources,
    replace,
    seedTraining,
    setup,
)
from pinn_platform.core import Artifact, OpArg, Operation, PipelineNode

log = logging.getLogger(__name__)


class ValidationScore(Operation):
    """Вычисляет общие проверочные потери с градиентами по координатам для PDE."""

    @override
    def run(self, inputs: OpArg) -> OpArg:
        condition = inputs["condition"].value
        with torch.enable_grad():
            score = condition(inputs["model"].value).item()
        return {"score": Artifact(value=score, meta={"kind": "validation_loss"})}


def _addValidation(built: Built, cfg: DictConfig, lastTrainer: PipelineNode) -> PipelineNode:
    pipe = built.pipe
    parts = pipe.getNode(ROOT / "data" / "parts")
    pde = pipe.getNode(ROOT / "problem" / "pde")
    loss = pipe.getNode(ROOT / "problem" / "loss")
    settings = cfg.tune_config.validation
    counts = {
        "initial": settings.get("initial_count"),
        "boundary": settings.get("boundary_count"),
        "pde": settings.count,
    }
    samplers = {
        name: nodes.validateSample(count=count, seed=settings.seed)
        for name, count in counts.items()
    }
    validationRoot = ROOT / "validation"
    with pipe.group(validationRoot):
        with pipe.group(validationRoot / "data"):
            sources = placeSampleSources(pipe, parts, samplers)
        conditions = placeConditions(pipe, cfg, sources, pde, loss)
        combined = place(
            pipe,
            "condition",
            nodes.weightedCondition(dict.fromkeys(conditions, 1.0)),
            **{name: (node, "condition") for name, node in conditions.items()},
        )
        return place(
            pipe,
            "score",
            ValidationScore(),
            condition=(combined, "condition"),
            model=(lastTrainer, "model"),
        )


def prepareStudyConfig(cfg: DictConfig, label: str) -> DictConfig:
    """Копирует конфигурацию и оставляет одну модель для исследования."""
    cfg = copy.deepcopy(cfg)
    if label not in cfg.models:
        raise ValueError(f"study model {label!r} is not selected: {list(cfg.models)}")
    if label in cfg.checkpoint.load:
        raise ValueError(f"study requires training {label!r}; remove it from checkpoint.load")

    with open_dict(cfg.models):
        for name in list(cfg.models):
            if name != label:
                del cfg.models[name]
    cfg.checkpoint.load = []
    cfg.checkpoint.save = False
    cfg.plotDir = None
    for plot in cfg.plots.values():
        plot.show = False
    return cfg


def _replaceTraining(trial: optuna.Trial, built: Built, cfg: DictConfig, label: str) -> None:
    """Применяет пробный сид и заменяет узлы со случайным отбором и тренерами."""
    settings = cfg.tune_config
    seed = trial.suggest_int("seed", settings.seed.low, settings.seed.high)
    seedTraining(cfg, label, seed)
    setup(cfg)

    for name in ("initial", "boundary", "pde"):
        current = built.pipe.getNode(ROOT / "data" / name)
        replace(built, current, make(cfg.samples[name]))

    trainers = built.trainers[label]
    for stage, field, name in (
        ("adam", "epochs", "adam_epochs"),
        ("lbfgs", "maxIter", "lbfgs_maxiter"),
    ):
        if stage not in trainers:
            continue
        trainer = cfg.stages[stage].trainer
        if field in trainer:
            bounds = settings[name]
            trainer[field] = trial.suggest_int(name, bounds.low, bounds.high)
            replace(built, trainers[stage], make(trainer, modelName=label))


def tune_config(cfg: DictConfig) -> optuna.Study:
    label = cfg.tune_config.model
    cfg = prepareStudyConfig(cfg, label)
    settings = cfg.tune_config
    built = buildPipeline(cfg)
    lastTrainer = built.trainers[label][cfg.schedule[-1]]
    validation = _addValidation(built, cfg, lastTrainer)

    def objective(trial: optuna.Trial) -> float:
        trialCfg = copy.deepcopy(cfg)
        modelCfg = trialCfg.models[label]
        features = modelCfg.features

        nFeatures = [features[0]]
        countLayers = trial.suggest_int(
            "countLayers", settings.count_layers.min, settings.count_layers.max
        )

        for i in range(countLayers):
            width = trial.suggest_int(f"width_{i}", settings.width.min, settings.width.max)
            nFeatures.append(width)
        nFeatures.append(features[-1])
        modelCfg.features = nFeatures

        _replaceTraining(trial, built, trialCfg, label)
        modelProducer = make(modelCfg, **built.env)
        replace(built, built.models[label], modelProducer)
        built.pipe.commit()

        built.pipe.run()
        return validation.results["score"].value

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=settings.n_trials)
    return study


@hydra.main(version_base=None, config_path="conf", config_name="transport")
def main(cfg: DictConfig) -> None:
    study = tune_config(cfg)

    if any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        log.info("Best validation loss %.6e with %s", study.best_value, study.best_params)
    else:
        log.warning("No tuning trial completed")


if __name__ == "__main__":
    main()
