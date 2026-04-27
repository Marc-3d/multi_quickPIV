"""Batch runtime state for the multi_quickPIV GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from multi_quickpiv_gui.gui.dialogs import BatchRunOptions
from multi_quickpiv_gui.workflow.params import WorkflowParams
from multi_quickpiv_gui.workflow.pipeline import BatchPIVResult, PIVPairResult


@dataclass(slots=True)
class BatchRuntimeState:
    """State container for one running or completed batch session."""

    running: bool = False
    paused: bool = False
    abort_requested: bool = False

    options: BatchRunOptions | None = None
    export_path: Path | None = None
    pair_results: list[PIVPairResult] = field(default_factory=list)

    total_pairs: int = 0
    next_pair_index: int = 0
    params: WorkflowParams | None = None

    def reset(self) -> None:
        """Reset the batch runtime state back to idle."""
        self.running = False
        self.paused = False
        self.abort_requested = False
        self.options = None
        self.export_path = None
        self.pair_results.clear()
        self.total_pairs = 0
        self.next_pair_index = 0
        self.params = None

    def start(
        self,
        *,
        options: BatchRunOptions,
        export_path: Path | None,
        total_pairs: int,
        params: WorkflowParams,
    ) -> None:
        """Initialize a fresh batch run."""
        self.running = True
        self.paused = False
        self.abort_requested = False
        self.options = options
        self.export_path = export_path
        self.pair_results = []
        self.total_pairs = total_pairs
        self.next_pair_index = 0
        self.params = params

    def append_result(self, result: PIVPairResult) -> None:
        """Append one completed pair result and advance the batch index."""
        self.pair_results.append(result)
        self.next_pair_index += 1

    def is_finished(self) -> bool:
        """Return True when all pair steps are complete."""
        return self.next_pair_index >= self.total_pairs

    def build_batch_result(self) -> BatchPIVResult:
        """Build a BatchPIVResult from the accumulated pair results."""
        return BatchPIVResult(pair_results=list(self.pair_results))