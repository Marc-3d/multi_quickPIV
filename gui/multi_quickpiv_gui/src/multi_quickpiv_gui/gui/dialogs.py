"""Dialog windows for the multi_quickPIV GUI."""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import simpledialog, ttk


@dataclass(slots=True)
class BatchRunOptions:
    """User choices for a batch PIV run."""

    preview_mode: str = "live"
    export_after_run: bool = False
    export_format: str = "npz"


class BatchRunDialog(simpledialog.Dialog):
    """Modal dialog for configuring a batch run."""

    def __init__(self, parent) -> None:
        self.result: BatchRunOptions | None = None
        self.var_preview_mode = tk.StringVar(value="live")
        self.var_export_after = tk.BooleanVar(value=False)
        self.var_export_format = tk.StringVar(value="npz")
        super().__init__(parent, title="Batch PIV Options")

    def body(self, master):
        ttk.Label(
            master,
            text="Choose how this batch run should behave.",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        preview_frame = ttk.LabelFrame(master, text="Preview", padding=8)
        preview_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Radiobutton(
            preview_frame,
            text="Live preview",
            value="live",
            variable=self.var_preview_mode,
        ).grid(row=0, column=0, sticky="w")

        ttk.Radiobutton(
            preview_frame,
            text="No preview",
            value="off",
            variable=self.var_preview_mode,
        ).grid(row=1, column=0, sticky="w")

        export_frame = ttk.LabelFrame(master, text="Export", padding=8)
        export_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        ttk.Checkbutton(
            export_frame,
            text="Export automatically after batch run",
            variable=self.var_export_after,
            command=self._toggle_export_widgets,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(export_frame, text="Format").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

        self.export_format_combo = ttk.Combobox(
            export_frame,
            textvariable=self.var_export_format,
            values=("npz", "h5"),
            state="readonly",
            width=10,
        )
        self.export_format_combo.grid(row=1, column=1, sticky="w", pady=(8, 0))

        self._toggle_export_widgets()
        return preview_frame

    def buttonbox(self):
        box = ttk.Frame(self)

        ttk.Button(box, text="Run", command=self.ok).pack(side="left", padx=5, pady=5)
        ttk.Button(box, text="Cancel", command=self.cancel).pack(side="left", padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def _toggle_export_widgets(self) -> None:
        state = "readonly" if self.var_export_after.get() else "disabled"
        self.export_format_combo.config(state=state)

    def apply(self) -> None:
        self.result = BatchRunOptions(
            preview_mode=self.var_preview_mode.get(),
            export_after_run=bool(self.var_export_after.get()),
            export_format=self.var_export_format.get(),
        )