from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

from .groupsystem import GroupSystem, Path
from .node import NodeInput, PipelineNode

type _Edges = dict[PipelineNode, list[PipelineNode]]


class Pipeline:
    def __init__(self):
        self.graph: _Edges = {}
        self.dependencies: _Edges = {}
        self.roots: dict[PipelineNode, None] = {}
        self.rootSystem: GroupSystem = GroupSystem()
        self.currSystem: GroupSystem = self.rootSystem
        self._nodeLocations: dict[PipelineNode, tuple[GroupSystem, Path]] = {}
        self._pendingReplacements: set[PipelineNode] = set()

    @property
    def rootLabel(self):
        return self.rootSystem.rootLabel

    def add(self, nodes: Iterable[PipelineNode]):
        for node in nodes:
            if node in self.graph:
                raise RuntimeError(f"Pipeline::add: node with name={node.name} has already exists")
            self._createNode(node)

    def removeNode(self, node: PipelineNode | Path) -> None:
        """Удаляет узел и его рёбра. Оставшиеся входы должны иметь допустимые источники."""
        target = self._getNode(node)
        graph = self._editedEdges(self.graph, target, None)
        dependencies = self._editedEdges(self.dependencies, target, None)
        self._validateGraph(graph, dependencies, {})

        if target in self._nodeLocations:
            group, path = self._nodeLocations[target]
            group.removeNode(path)
            del self._nodeLocations[target]
        self._installGraph(graph, dependencies)
        self._pendingReplacements.discard(target)
        self._invalidateNodes(graph)

    def replaceNode(
        self,
        node: PipelineNode | Path,
        replacement: PipelineNode,
        /,
        *,
        keepIncomingEdges: bool = False,
    ) -> None:
        """Заменяет узел и по умолчанию строит входящие рёбра по входам нового узла.

        Путь и исходящие рёбра сохраняются. Потребители используют прежние имена
        выходов. Параметр `keepIncomingEdges` сохраняет все входящие рёбра,
        включая связи, задающие порядок выполнения.

        Новый узел должен выдавать те же выходы. Вызовите `commit()` перед запуском
        изменённого пайплайна.
        """
        if replacement in self.graph:
            raise ValueError("Pipeline.replaceNode: replacement already belongs to this pipeline")

        target = self._getNode(node)
        graph, dependencies, inputs = self._prepareReplacement(
            target, replacement, keepIncomingEdges=keepIncomingEdges
        )
        self._replaceLocation(target, replacement)
        for reader, bindings in inputs.items():
            reader.inputs = bindings
        replacement._inheritOutputContract(target)
        self._installGraph(graph, dependencies)
        self._pendingReplacements.discard(target)
        self._pendingReplacements.add(replacement)

    def commit(self) -> None:
        """Сбрасывает результаты заменённых узлов и их потомков перед следующим запуском."""
        if not self._pendingReplacements:
            return
        self.validate()

        affected = set(self._pendingReplacements)
        pending = deque(affected)
        while pending:
            for child in self.graph[pending.popleft()]:
                if child not in affected:
                    affected.add(child)
                    pending.append(child)
        self._invalidateNodes(affected)
        self._pendingReplacements.clear()

    def _prepareReplacement(
        self, target: PipelineNode, replacement: PipelineNode, *, keepIncomingEdges: bool
    ) -> tuple[_Edges, _Edges, dict[PipelineNode, dict[str, NodeInput | None]]]:
        graph = self._editedEdges(self.graph, target, replacement)
        dependencies = self._editedEdges(self.dependencies, target, replacement)
        if not keepIncomingEdges:
            for parent in dict.fromkeys(dependencies[replacement]):
                graph[parent] = [child for child in graph[parent] if child is not replacement]

            parents = dict.fromkeys(
                binding[0] for binding in replacement.inputs.values() if binding is not None
            )
            for parent in parents:
                if parent not in graph:
                    raise ValueError(
                        f"replacement input source {parent.label} is unavailable after replacement"
                    )
                graph[parent].append(replacement)
            dependencies[replacement] = list(parents)

        inputs: dict[PipelineNode, dict[str, NodeInput | None]] = {}
        for reader in self.graph:
            if reader is target:
                continue
            if bindings := reader.getInputs(target):
                inputs[reader] = reader.inputs | {
                    name: (replacement, output) for name, output in bindings.items()
                }
        self._validateGraph(graph, dependencies, inputs)
        return graph, dependencies, inputs

    def _replaceLocation(self, target: PipelineNode, replacement: PipelineNode) -> None:
        if target in self._nodeLocations:
            group, path = self._nodeLocations[target]
            group.replaceNode(path, replacement)
            self._nodeLocations[replacement] = self._nodeLocations.pop(target)
            replacement.name = path.label
        else:
            replacement.name = None

    @staticmethod
    def _editedEdges(
        edges: _Edges, target: PipelineNode, replacement: PipelineNode | None
    ) -> _Edges:
        if replacement is None:
            return {
                node: [neighbor for neighbor in neighbors if neighbor is not target]
                for node, neighbors in edges.items()
                if node is not target
            }
        return {
            (replacement if node is target else node): [
                replacement if neighbor is target else neighbor for neighbor in neighbors
            ]
            for node, neighbors in edges.items()
        }

    def _installGraph(self, graph: _Edges, dependencies: _Edges) -> None:
        self.graph.clear()
        self.graph.update(graph)
        self.dependencies.clear()
        self.dependencies.update(dependencies)
        self.roots.clear()
        self.roots.update(dict.fromkeys(node for node in graph if not dependencies[node]))

    @staticmethod
    def _invalidateNodes(nodes: Iterable[PipelineNode]) -> None:
        for node in nodes:
            node.results = {}
            node.was_executed = False

    def connect(self, start: PipelineNode | Path, end: Iterable[PipelineNode | Path]):
        startNode = self._getNode(start)
        for obj in end:
            node = self._getNode(obj)
            if node in self.roots:
                del self.roots[node]

            self.graph[startNode].append(node)
            self.dependencies[node].append(startNode)

    def createGroup(self, path: Path, parents: bool = True, exist: bool = False):
        groupSystem, path = self._resolve(path)
        groupSystem.addGroup(path, parents, exist)

    @contextmanager
    def group(self, path: Path) -> Iterator[GroupSystem]:
        self.createGroup(path, exist=True)
        previous = self.currSystem
        self.downCurrGroup(path)
        try:
            yield self.currSystem
        finally:
            self.currSystem = previous

    def getNode(self, path: Path) -> PipelineNode:
        return self._getNode(path)

    def _getNode(self, obj: PipelineNode | Path):
        if isinstance(obj, PipelineNode):
            if obj not in self.graph:
                raise RuntimeError(
                    f"Pipeline._getNode: node {obj.label} was not added to the pipeline"
                )
            return obj
        groupSystem, path = self._resolve(obj)
        node = groupSystem.getNode(path)
        if node is None:
            raise RuntimeError(f"Pipeline._getNode: {obj!r} is a group, not a node")
        return node

    def _resolve(self, path: Path) -> tuple[GroupSystem, Path]:
        if self._isAbsPath(path):
            return self.rootSystem, self._cutRoot(path)
        return self.currSystem, path

    def _isAbsPath(self, path: Path) -> bool:
        return not path.isEmpty() and path[0] == self.rootSystem.rootLabel

    def _cutRoot(self, path: Path) -> Path:
        return path[1:]

    def _createNode(self, node: PipelineNode):
        if node.name is not None:
            if not node.name or "/" in node.name:
                raise ValueError(
                    f"Pipeline.add: node name {node.name!r} must be non-empty and contain no '/'"
                )
            path = Path.parse(node.name)
            self.currSystem.addNode(path, node, exist=False)
            self._nodeLocations[node] = (self.currSystem, path)

        self.graph[node] = []
        self.roots[node] = None
        self.dependencies[node] = []

    def downCurrGroup(self, path: Path):
        groupSystem, path = self._resolve(path)
        self.currSystem = groupSystem.getSubGroup(path)

    def upCurrGroup(self, times=1):
        self.currSystem = self.currSystem.up(times)

    def setCurrGroup(self, absPath: Path):
        if not self._isAbsPath(absPath):
            raise ValueError(
                f"Pipeline.setCurrGroup: {absPath!r} must start with {self.rootLabel!r}"
            )
        self.currSystem = self.rootSystem.getSubGroup(self._cutRoot(absPath))

    def validate(self) -> list[PipelineNode]:
        return self._validateGraph(self.graph, self.dependencies, {})

    @staticmethod
    def _validateGraph(
        graph: _Edges,
        dependencies: _Edges,
        inputs: dict[PipelineNode, dict[str, NodeInput | None]],
    ) -> list[PipelineNode]:
        in_degree = {node: len(dependencies[node]) for node in graph}
        ready = deque(node for node, degree in in_degree.items() if degree == 0)
        order: list[PipelineNode] = []

        while ready:
            node = ready.popleft()
            order.append(node)
            for child in graph[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)

        if len(order) != len(graph):
            blocked = [node.label for node, degree in in_degree.items() if degree > 0]
            raise ValueError(
                "pipeline graph contains a cycle; blocked nodes: " + ", ".join(sorted(blocked))
            )

        ancestors: dict[PipelineNode, set[PipelineNode]] = {}
        for node in order:
            upstream = set(dependencies[node])
            for dependency in dependencies[node]:
                upstream.update(ancestors[dependency])
            ancestors[node] = upstream
            for localName, source in inputs.get(node, node.inputs).items():
                if source is None:
                    raise ValueError(f"node {node.label} has an unconnected input {localName!r}")
                if source[0] not in upstream:
                    raise ValueError(
                        f"node {node.label} reads {source[1]!r} (as {localName!r}) "
                        f"from node {source[0].label}, which is not among its dependencies"
                    )
        return order

    def run(self):
        if self._pendingReplacements:
            raise RuntimeError("Pipeline.run: call commit() after replacing nodes")
        for node in self.validate():
            node.execute()
