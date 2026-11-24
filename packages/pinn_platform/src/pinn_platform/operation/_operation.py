from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # `type` aliases are evaluated lazily, so a runtime import is not needed and
    # would be circular: core/__init__ imports Operation from this module.
    from ..core.artifact import Artifact

type OpArg = dict[str, Artifact]


class Operation(ABC):
    @abstractmethod
    def run(self, inputs: OpArg) -> OpArg: ...

    def getMeta(self) -> dict[str, Any]:
        return {}
