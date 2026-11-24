from dataclasses import dataclass
from typing import Any


@dataclass
class Artifact:
    value: Any
    meta: dict[str, Any]
