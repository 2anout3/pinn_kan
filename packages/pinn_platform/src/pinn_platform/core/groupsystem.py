from dataclasses import dataclass
from typing import Self

from .node import PipelineNode


@dataclass(frozen=True, order=True)
class Path:
    parts: tuple[str, ...]

    @staticmethod
    def isValidPath(splited: list[str]) -> bool:
        for p in splited:
            if not p:
                return False
        return True

    @classmethod
    def parse(cls, s: str) -> Self:
        splited = s.split("/")
        if not cls.isValidPath(splited):
            raise RuntimeError("Path.parse: not valid path")
        return cls(tuple(splited))

    def __truediv__(self, child: "str | Path") -> "Path":
        child = child if isinstance(child, Path) else Path.parse(child)
        return Path(self.parts + child.parts)

    def __iter__(self):
        return iter(self.parts)

    def __len__(self):
        return len(self.parts)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Path(self.parts[index])
        return self.parts[index]

    def getSub(self, s: slice):
        return Path(self.parts[s])

    def __repr__(self):
        return "/".join(self.parts)

    def isEmpty(self):
        return len(self.parts) == 0

    @property
    def parent(self):
        return Path(self.parts[:-1])

    @property
    def label(self):
        return self.parts[-1]


class _GSNode:
    def __init__(
        self, label: str, parent: "_GSNode | None" = None, pipeNode: PipelineNode | None = None
    ):
        self._label = label
        self._pipeNode = pipeNode
        self._children: dict[str, _GSNode] = {}
        self._parent: _GSNode | None = parent

    def isPipeNode(self):
        return self._pipeNode is not None

    def __setitem__(self, label: str, node: "_GSNode"):
        if self.isPipeNode():
            raise RuntimeError("_GSNode: can not set child for PipelineNode")
        self._children[label] = node

    def __contains__(self, childLabel):
        return childLabel in self._children

    def __getitem__(self, childLabel):
        return self._children[childLabel]


class GroupSystem:
    def __init__(self, root: _GSNode | None = None):
        self._root = root if root is not None else _GSNode("root", None)

    @property
    def rootLabel(self):
        return self._root._label

    @property
    def isRealRoot(self):
        return self._root._parent is None

    def addGroup(self, path: Path, parents=False, exist=True):
        if path.isEmpty():
            raise RuntimeError("GroupSystem.addGroup: path have to be not empty")
        parentNode = self._toGroup(path.parent, parents)
        label = path.label

        if label in parentNode:
            if parentNode[label].isPipeNode():
                raise RuntimeError(
                    f"GroupSystem.addGroup: {path!r} is a pipeline node, not a group"
                )
            if not exist:
                raise RuntimeError(f"GroupSystem.addGroup: group system already has had {path!r}")
        else:
            parentNode[label] = _GSNode(label, parentNode)

    def addNode(self, path: Path, node: PipelineNode, parents=False, exist=True):
        parentNode = self._toGroup(path.parent, parents)
        label = path.label

        if label in parentNode:
            if not exist or parentNode[label]._pipeNode is not node:
                raise RuntimeError(f"GroupSystem.addNode: group system already has had {path!r}")
        else:
            parentNode[label] = _GSNode(label, parentNode, node)

    def removeNode(self, path: Path) -> None:
        parent, entry = self._nodeEntry(path)
        del parent._children[entry._label]

    def replaceNode(self, path: Path, node: PipelineNode) -> None:
        _, entry = self._nodeEntry(path)
        entry._pipeNode = node

    def _nodeEntry(self, path: Path) -> tuple[_GSNode, _GSNode]:
        parent, entry = self._entry(path)
        if not entry.isPipeNode():
            raise RuntimeError(f"GroupSystem: {path!r} is a group, not a pipeline node")
        return parent, entry

    def getAllGroupNodes(self):
        pass

    def _toGroup(self, path: Path, parents=False) -> _GSNode:
        currNode = self._root
        for i, part in enumerate(path):
            if part not in currNode:
                if not parents:
                    raise RuntimeError(
                        f"GroupSystem._toGroup: there is not {path[: i + 1]!r} in GroupSystem"
                    )
                currNode[part] = _GSNode(part, currNode)
            currNode = currNode[part]
            if currNode.isPipeNode():
                raise RuntimeError(
                    f"GroupSystem._toGroup: {path[: i + 1]!r} is a pipeline node, not a group"
                )

        return currNode

    def getSubGroup(self, path: Path) -> "GroupSystem":
        return GroupSystem(self._toGroup(path))

    def getNode(self, path: Path) -> PipelineNode | None:
        _, entry = self._entry(path)
        return entry._pipeNode

    def _entry(self, path: Path) -> tuple[_GSNode, _GSNode]:
        if path.isEmpty():
            raise RuntimeError("GroupSystem: path must not be empty")
        parentNode = self._toGroup(path.parent)
        if path.label not in parentNode:
            raise RuntimeError(f"GroupSystem: there is not {path!r} in GroupSystem")
        return parentNode, parentNode[path.label]

    def up(self, times=1) -> "GroupSystem":
        currNode = self._root
        for _ in range(times):
            if currNode._parent is None:
                raise RuntimeError("GroupSystem.up: can not up")
            currNode = currNode._parent
        return GroupSystem(currNode)

    def _print(self, currNode: _GSNode | None = None, parent: Path | None = None):
        currNode = currNode if currNode is not None else self._root
        path = parent / currNode._label if parent is not None else Path.parse(currNode._label)
        print(f"{path!r}")
        for child in currNode._children:
            self._print(currNode[child], path)
