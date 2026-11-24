from ..operation._operation import OpArg, Operation
from .artifact import Artifact

type NodeInput = tuple[PipelineNode, str]


class PipelineNode:
    def __init__(
        self,
        operation: Operation,
        inputs: dict[str, NodeInput | None],
        name=None,
    ):
        self._operation = operation
        self.inputs = inputs
        self.name = name
        self.results: OpArg = {}
        self.was_executed = False
        self._expectedOutputs: frozenset[str] | None = None

    @property
    def label(self) -> str:
        return self.name if self.name is not None else "<unnamed>"

    def _resolve(self, source: NodeInput | None) -> Artifact:
        if source is None:
            raise RuntimeError(f"node {self.label} has an unconnected input")
        node, outputName = source
        if outputName not in node.results:
            raise KeyError(
                f"node {self.label} reads {outputName!r} from node {node.label}, "
                f"which has outputs {sorted(node.results)}"
            )
        return node.results[outputName]

    def execute(self):
        if self.was_executed:
            return

        localInputs = {
            localName: self._resolve(source) for localName, source in self.inputs.items()
        }

        results = self._operation.run(localInputs)
        if self._expectedOutputs is not None and results.keys() != self._expectedOutputs:
            raise ValueError(
                f"node {self.label} returned outputs {sorted(results)}, "
                f"expected {sorted(self._expectedOutputs)}"
            )
        self.results = results
        self.was_executed = True

    def _inheritOutputContract(self, node: "PipelineNode") -> None:
        self._expectedOutputs = (
            frozenset(node.results) if node.was_executed else node._expectedOutputs
        )

    def replaceInput(self, arg: str, newInput: NodeInput):
        if arg not in self.inputs:
            raise RuntimeError("asdasd")
        self.inputs[arg] = newInput

    def getInputs(self, node: "PipelineNode") -> dict[str, str]:
        return {
            innerName: arg[1]
            for innerName, arg in self.inputs.items()
            if arg is not None and arg[0] is node
        }

    def getEmptyInputs(self) -> list[str]:
        return [name for name, v in self.inputs.items() if v is None]

    def removeInput(self, arg: str):
        self.inputs[arg] = None
