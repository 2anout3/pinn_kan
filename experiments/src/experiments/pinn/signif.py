"""Оценивает вклад сида, архитектуры и настроек обучения в проверочные потери."""

import copy
import logging
import math

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf
from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances

from experiments.pinn.__main__ import pickDevice as pickDevice
from experiments.pinn.build import buildPipeline, seedTraining, setup
from experiments.pinn.tune import _addValidation, prepareStudyConfig

log = logging.getLogger(__name__)


def _suggestTrainerParams(trial: optuna.Trial, cfg: DictConfig, ranges: DictConfig) -> None:
    """Подбирает числовые параметры тренеров активных стадий."""
    for stage in cfg.schedule:
        trainer = cfg.stages[stage].trainer
        for name, bounds in ranges.get(stage, {}).items():
            parameter = f"{stage}.{name}"
            current = trainer[name]
            if type(current) is int:
                suggest = trial.suggest_int
            elif type(current) is float:
                suggest = trial.suggest_float
            else:
                raise ValueError(f"{parameter}: expected an int or float trainer parameter")
            trainer[name] = suggest(
                parameter, bounds.low, bounds.high, log=bounds.get("log", False)
            )


def signif_config(cfg: DictConfig) -> optuna.Study:
    """Выполняет испытания с независимым случайным выбором параметров."""
    label = cfg.signif_config.model
    cfg = prepareStudyConfig(cfg, label)
    settings = cfg.signif_config
    validation = cfg.tune_config.validation
    cfg.tune_config.validation = OmegaConf.to_container(validation, resolve=True)

    def objective(trial: optuna.Trial) -> float:
        trialCfg = copy.deepcopy(cfg)
        seed = trial.suggest_categorical("seed", list(settings.seeds))
        seedTraining(trialCfg, label, seed)
        architecture = trial.suggest_categorical("architecture", list(settings.architectures))
        model = trialCfg.models[label]
        hiddenWidths = settings.architectures[architecture]
        model.features = [model.features[0], *hiddenWidths, model.features[-1]]
        _suggestTrainerParams(trial, trialCfg, settings.trainers)

        setup(trialCfg)
        built = buildPipeline(trialCfg)
        lastTrainer = built.trainers[label][trialCfg.schedule[-1]]
        score = _addValidation(built, trialCfg, lastTrainer)
        built.pipe.run()
        return score.results["score"].value

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.RandomSampler(seed=settings.sampler_seed)
    )
    study.optimize(objective, n_trials=settings.n_trials)
    return study


def parameter_importances(study: optuna.Study, target_quantile: float) -> dict[str, float]:
    """Возвращает нормированные оценки важности для лучших завершённых испытаний."""
    scores = {
        trial.value
        for trial in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if trial.value is not None and math.isfinite(trial.value)
    }
    if len(scores) < 2:
        return {}
    # Нулевая сумма оценок оставляет рейтинг пустым.
    raw = get_param_importances(
        study,
        evaluator=PedAnovaImportanceEvaluator(target_quantile=target_quantile),
        normalize=False,
    )
    total = sum(raw.values())
    return {name: importance / total for name, importance in raw.items()} if total > 0 else {}


@hydra.main(version_base=None, config_path="conf", config_name="transport")
def main(cfg: DictConfig) -> None:
    study = signif_config(cfg)
    results = parameter_importances(study, cfg.signif_config.target_quantile)
    if not results:
        log.warning("Parameter importances unavailable for the completed trials")
        return

    log.info("Parameter importances (PED-ANOVA)")
    for name, importance in results.items():
        log.info("%-18s %6.2f%%", name, 100 * importance)


if __name__ == "__main__":
    main()
