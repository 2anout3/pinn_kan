"""Структурированные конфиги Hydra из сигнатуры функции или __init__ класса.

    @hydra_config
    class Trainer:
        def __init__(self, layers: list[int], dtype: torch.dtype = torch.float32): ...

    Trainer.Config                   # сгенерированный dataclass (схема для Hydra/OmegaConf)
    Trainer.from_config(cfg)         # cfg (DictConfig / dict / Config) -> Trainer(...)
    hydra.utils.instantiate(cfg)     # тоже работает: в Config есть поле _target_

Конфиги, объявленные с name (и group), попадают в ConfigStore как схема
<group>/<name>_schema, а write_config_tree() раскладывает по файлам
<root>/<group>/<name>.yaml со значениями по умолчанию; файл наследует схему
через defaults, так что Hydra проверяет типы.

Типы, которых нет в YAML (torch.dtype, torch.device, ...), регистрируются через
register_type(): задаётся YAML-тип и пара функций encode (python -> yaml) и
decode (yaml -> python). В dataclass попадает YAML-тип, при создании объекта
значение переводится обратно.
"""

from __future__ import annotations

import collections.abc as cabc
import copy
import dataclasses
import enum
import functools
import inspect
import logging
import pathlib
import types
from collections.abc import Callable
from dataclasses import dataclass, field, make_dataclass
from typing import (
    Any,
    Literal,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import torch
from omegaconf import MISSING, OmegaConf

__all__ = ["hydra_config", "register_type", "write_config_tree", "Codec"]

log = logging.getLogger(__name__)

# Схемы, объявленные с name: (group, name) -> Config. Их раскладывает write_config_tree().
_NAMED_CONFIGS: dict[tuple[str | None, str], type] = {}


def _identity(x: Any) -> Any:
    return x


@dataclass(frozen=True)
class Codec:
    """Как тип представлен в конфиге и как переводить значения туда и обратно."""

    yaml_type: Any
    encode: Callable[[Any], Any] = _identity  # python -> yaml (для значений по умолчанию)
    decode: Callable[[Any], Any] = _identity  # yaml -> python (при создании объекта)


# --------------------------------------------------------------------------- #
# Реестр типов, которые нельзя положить в YAML как есть
# --------------------------------------------------------------------------- #
_REGISTRY: dict[type, Codec] = {}


def register_type(
    py_type: type, yaml_type: Any, *, encode: Callable[[Any], Any], decode: Callable[[Any], Any]
) -> None:
    """Зарегистрировать перевод py_type <-> yaml_type.

    register_type(torch.dtype, str,
                  encode=lambda d: str(d).removeprefix("torch."),
                  decode=lambda s: getattr(torch, s))
    """
    _REGISTRY[py_type] = Codec(yaml_type, encode, decode)


def _registered_codec(tp: type) -> Codec | None:
    py_type = (
        tp if tp in _REGISTRY else next((base for base in _REGISTRY if issubclass(tp, base)), None)
    )
    if py_type is None:
        return None
    raw = _REGISTRY[py_type]
    # Значение может прийти уже готовым объектом (например, из Python-кода
    # или после hydra.utils.instantiate) — тогда его не трогаем.
    return Codec(
        raw.yaml_type,
        encode=lambda v: raw.encode(v) if isinstance(v, py_type) else v,
        decode=lambda v: v if isinstance(v, py_type) else raw.decode(v),
    )


def _register_bultin_types() -> None:
    register_type(
        torch.dtype,
        str,
        encode=lambda d: str(d).removeprefix("torch."),
        decode=lambda s: getattr(torch, s),
    )
    register_type(torch.device, str, encode=lambda d: d.type, decode=torch.device)


_register_bultin_types()
# --------------------------------------------------------------------------- #
# Аннотация python -> Codec (тип для dataclass + перевод значений)
# --------------------------------------------------------------------------- #
# То, что OmegaConf умеет хранить без преобразований
_NATIVE = (int, float, bool, str, bytes, pathlib.Path)

_SEQ_CONTAINERS = {
    list: list,
    tuple: tuple,
    set: set,
    frozenset: frozenset,
    cabc.Sequence: list,
    cabc.MutableSequence: list,
    cabc.Set: set,
    cabc.MutableSet: set,
    cabc.Iterable: list,
}
_MAP_CONTAINERS = (dict, cabc.Mapping, cabc.MutableMapping)


def make_codec(tp: Any) -> Codec:
    if tp is Any:
        return Codec(Any)

    if isinstance(tp, TypeAliasType):  # type Scheme = Literal[...] -> Literal[...]
        return make_codec(tp.__value__)

    if inspect.isclass(tp):
        if (codec := _registered_codec(tp)) is not None:
            return codec
        if tp in _NATIVE or issubclass(tp, enum.Enum):
            return Codec(tp)
        if "__hydra_config__" in tp.__dict__:  # вложенный @hydra_config-класс
            return Codec(
                tp.__hydra_config__,
                encode=_cannot_encode,
                decode=lambda v: v if isinstance(v, tp) else tp.from_config(v),
            )

    origin, args = get_origin(tp), get_args(tp)

    # Optional[X] / X | None. Прочие Union OmegaConf не поддерживает -> Any.
    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = make_codec(non_none[0])
            return Codec(
                inner.yaml_type | None,
                encode=lambda v: None if v is None else inner.encode(v),
                decode=lambda v: None if v is None else inner.decode(v),
            )
        return Codec(Any)

    # Literal OmegaConf тоже не знает: храним базовый тип, проверяем при decode
    if origin is Literal:
        base_types = {type(a) for a in args}
        base = base_types.pop() if len(base_types) == 1 else Any

        def check_literal(v: Any) -> Any:
            if v not in args:
                raise ValueError(f"{v!r} не входит в {list(args)}")
            return v

        return Codec(base, decode=check_literal)

    # Последовательности -> List[...]; при decode возвращаем исходный контейнер
    container = origin or tp
    if container in _SEQ_CONTAINERS:
        py_container = _SEQ_CONTAINERS[container]
        if container is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):
            items = [make_codec(a) for a in args]  # tuple[int, str] и т.п.
            same = len({c.yaml_type for c in items}) == 1
            return Codec(
                list[items[0].yaml_type if same else Any],
                encode=lambda v: [c.encode(x) for c, x in zip(items, v, strict=False)],
                decode=lambda v: tuple(c.decode(x) for c, x in zip(items, v, strict=False)),
            )
        item = make_codec(args[0]) if args else Codec(Any)
        return Codec(
            list[item.yaml_type],
            encode=lambda v: [item.encode(x) for x in v],
            decode=lambda v: py_container(item.decode(x) for x in v),
        )

    if container in _MAP_CONTAINERS:
        key = make_codec(args[0]) if args else Codec(str)
        val = make_codec(args[1]) if args else Codec(Any)
        return Codec(
            dict[key.yaml_type, val.yaml_type],
            encode=lambda d: {key.encode(k): val.encode(v) for k, v in d.items()},
            decode=lambda d: {key.decode(k): val.decode(v) for k, v in d.items()},
        )

    name = getattr(tp, "__qualname__", repr(tp))
    raise TypeError(
        f"Тип {tp!r} нельзя представить в YAML. Зарегистрируйте перевод: "
        f"register_type({name}, str, encode=..., decode=...)"
    )


def _cannot_encode(value: Any) -> Any:
    raise TypeError(
        "значение по умолчанию — готовый объект, его нельзя записать "
        "в конфиг; используйте None или уберите значение по умолчанию"
    )


def _is_optional(t: Any) -> bool:
    return t is Any or (get_origin(t) is Union and type(None) in get_args(t))


# --------------------------------------------------------------------------- #
# Декоратор
# --------------------------------------------------------------------------- #
def hydra_config(
    obj: Any = None,
    *,
    name: str | None = None,
    group: str | None = None,
    target: bool = True,
    exclude: cabc.Iterable[str] = (),
):
    """Сгенерировать dataclass-схему конфига по аргументам функции / __init__.

    Добавляет объекту:
      .Config       — dataclass для Hydra (OmegaConf.structured, ConfigStore);
      .from_config  — создать объект из конфига с переводом типов обратно.

    name/group — зарегистрировать схему в ConfigStore Hydra как <group>/<name>_schema.
                 Суффикс нужен, чтобы схема не совпадала по имени с файлом <name>.yaml:
                 такое совпадение Hydra считает устаревшим (automatic schema matching).
    target     — добавить поле _target_, чтобы работал hydra.utils.instantiate.
    exclude    — параметры, которых нет в конфиге: их значения задаёт код, создающий
                 объект, — from_config(cfg, dtype=...) или instantiate(cfg, dtype=...).
                 Передаются как есть, без перевода типов. Так общие для всего запуска
                 значения (dtype, device) нельзя задать отдельному объекту по ошибке.
    """
    excluded = frozenset(exclude)

    def wrap(obj: Any) -> Any:
        func = obj.__init__ if inspect.isclass(obj) else obj
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        P = inspect.Parameter

        unknown = excluded - sig.parameters.keys()
        if unknown:
            raise TypeError(
                f"{obj.__qualname__}: exclude содержит чужие параметры {sorted(unknown)}"
            )

        fields: list[tuple[str, Any, Any]] = []
        codecs: dict[str, Codec] = {}

        if target:
            path = f"{obj.__module__}.{obj.__qualname__}.from_config"
            fields.append(("_target_", str, field(default=path)))

        for pname, p in sig.parameters.items():
            if pname in ("self", "cls") or p.kind in (P.VAR_POSITIONAL, P.VAR_KEYWORD):
                continue
            if pname in excluded:
                continue
            if p.kind is P.POSITIONAL_ONLY:
                raise TypeError(
                    f"{obj.__qualname__}: positional-only параметр "
                    f"{pname!r} нельзя передать из конфига"
                )

            has_default = p.default is not P.empty
            if pname in hints:
                tp = hints[pname]
            elif has_default and p.default is not None:
                tp = type(p.default)  # аннотации нет — выводим по дефолту
            else:
                tp = Any

            try:
                codec = make_codec(tp)
            except TypeError as e:
                raise TypeError(f"{obj.__qualname__}.{pname}: {e}") from None
            yaml_t = codec.yaml_type

            if not has_default:
                f = field(default=MISSING)  # "???" — обязательное поле в Hydra
            else:
                if p.default is None and not _is_optional(yaml_t):
                    yaml_t = yaml_t | None
                try:
                    value = None if p.default is None else codec.encode(p.default)
                except Exception as e:
                    raise TypeError(
                        f"{obj.__qualname__}.{pname}: значение по умолчанию "
                        f"{p.default!r} не переводится в YAML: {e}"
                    ) from None
                if isinstance(value, (list, dict)):
                    f = field(default_factory=functools.partial(copy.deepcopy, value))
                else:
                    f = field(default=value)

            fields.append((pname, yaml_t, f))
            codecs[pname] = codec

        Config = make_dataclass(f"{obj.__name__}Config", fields)
        Config.__module__ = obj.__module__
        Config.__qualname__ = f"{obj.__qualname__}.Config"
        Config.__doc__ = inspect.getdoc(obj)

        def from_config(cfg: Any = None, /, **overrides: Any) -> Any:
            """Создать объект из конфига (DictConfig, dict или Config)."""
            data = _as_dict(cfg)
            data.update(overrides)
            data.pop("_target_", None)
            kwargs = {}
            for key, value in data.items():
                if key in excluded:  # значение задаёт код, а не конфиг
                    kwargs[key] = value
                    continue
                if key not in codecs:
                    raise TypeError(f"{obj.__qualname__}: неизвестный параметр {key!r}")
                if isinstance(value, str) and value == MISSING:
                    raise ValueError(f"{obj.__qualname__}: обязательный параметр {key!r} не задан")
                try:
                    kwargs[key] = codecs[key].decode(value)
                except Exception as e:
                    raise ValueError(f"{obj.__qualname__}.{key}: {e}") from e
            return obj(**kwargs)

        obj.__hydra_config__ = Config
        obj.Config = Config
        obj.from_config = staticmethod(from_config) if inspect.isclass(obj) else from_config

        if name is not None:
            from hydra.core.config_store import ConfigStore

            ConfigStore.instance().store(name=_schema_name(name), group=group, node=Config)
            _NAMED_CONFIGS[(group, name)] = Config
        return obj

    return wrap(obj) if obj is not None else wrap


def write_config_tree(root: str | pathlib.Path, *, overwrite: bool = False) -> list[pathlib.Path]:
    """Разложить конфиги, объявленные через @hydra_config(name=..., group=...), по файлам.

    Каждый конфиг пишется в <root>/<group>/<name>.yaml (без group — в <root>/<name>.yaml),
    вложенная группа "model/basis" становится вложенной папкой. Файл начинается с
    defaults: [<name>_schema, _self_] — так Hydra проверяет его по схеме; дальше идут
    значения по умолчанию, обязательные параметры записываются как ??? и их надо заполнить.
    Это та раскладка, в которой Hydra ищет конфиги в config_path.

    Существующие файлы не трогаются, если не передан overwrite=True: их могли уже
    отредактировать руками. Возвращает пути записанных файлов.
    """
    root = pathlib.Path(root)
    written: list[pathlib.Path] = []
    for (group, name), Config in sorted(
        _NAMED_CONFIGS.items(), key=lambda item: (item[0][0] or "", item[0][1])
    ):
        directory = root.joinpath(*group.split("/")) if group else root
        path = directory / f"{name}.yaml"
        if path.exists() and not overwrite:
            log.info(f"[config] {path} exists, kept")
            continue
        directory.mkdir(parents=True, exist_ok=True)
        body = OmegaConf.to_container(OmegaConf.structured(Config), resolve=False)
        assert isinstance(body, dict)
        body.pop("_target_", None)  # приходит из схемы, в файле его править не нужно
        defaults = f"defaults:\n  - {_schema_name(name)}\n  - _self_\n"
        path.write_text(
            defaults + ("\n" + OmegaConf.to_yaml(OmegaConf.create(body)) if body else "")
        )
        written.append(path)
    return written


def _schema_name(name: str) -> str:
    return f"{name}_schema"


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        # не asdict(): он рекурсивно превратит вложенные конфиги в dict — это ок,
        # но так дешевле и сохраняет уже созданные объекты
        return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    if isinstance(cfg, cabc.Mapping):  # dict и omegaconf.DictConfig
        return dict(cfg.items())
    raise TypeError(f"ожидался dict, DictConfig или dataclass, получено {type(cfg)!r}")
