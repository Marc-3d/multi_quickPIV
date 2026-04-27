"""Minimal Tkinter app shell for the extracted multi_quickPIV GUI pipeline."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import tkinter.ttk as ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from multi_quickpiv_gui.backend.export import (
    save_batch_result,
    save_pair_result,
    save_piv_animation,
)

from multi_quickpiv_gui.backend.io import (
    LoadedPIVResult,
    LoadedStack,
    load_saved_piv_result,
    load_stack,
)
from multi_quickpiv_gui.gui.preview import (
    PreviewState,
    draw_loaded_frame,
    draw_vector_field_only,
    ensure_preview_artists,
    reset_preview_state,
)
from multi_quickpiv_gui.gui.dialogs import BatchRunDialog, BatchRunOptions
from multi_quickpiv_gui.runtime.batch import BatchRuntimeState

from multi_quickpiv_gui.gui.params_form import (
    ParamsFormState,
    build_params_panel,
    build_workflow_params,
    create_params_form_state,
)

from multi_quickpiv_gui.workflow.pipeline import (
    BatchPIVResult,
    PIVPairResult,
    run_piv_pair,
)

class MultiQuickPIVApp:
    """Minimal application shell around the extracted pipeline."""
  
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("multi_quickPIV GUI")
        self.root.geometry("1500x850")

        self.loaded_stack: LoadedStack | None = None
        self.loaded_piv_result: LoadedPIVResult | None = None
        self.current_result: PIVPairResult | BatchPIVResult | None = None
        self.current_export_name_hint = "piv_result"
        self.current_single_pair_indices: tuple[int, int] | None = None
        self.preview_state = PreviewState()
        self._status_after_id: str | None = None
        self.params_form: ParamsFormState = create_params_form_state(self.root)

        self.batch = BatchRuntimeState()

        self._build_variables()
        self._build_layout()
        self._set_idle_state()
        self.root.after(300, self.show_piv_info)

    def _build_variables(self) -> None:
        self.var_file_name = tk.StringVar(value="File: (none)")
        self.var_result = tk.StringVar(value="Pick a file to start")
        self.var_status = tk.StringVar(value="")
        self.var_frame = tk.IntVar(value=0)

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, columnspan=3, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, textvariable=self.var_file_name).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(top, textvariable=self.var_result).grid(
            row=0, column=1, sticky="ew", padx=12
        )

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=1, column=0, sticky="nsw")

        center = ttk.Frame(self.root, padding=8)
        center.grid(row=1, column=1, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)

        right = ttk.Frame(self.root, padding=8)
        right.grid(row=1, column=2, sticky="nse")

        bottom = ttk.Frame(self.root, padding=8)
        bottom.grid(row=2, column=0, columnspan=3, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self._build_parameter_panel(left)
        self._build_preview_panel(center)
        self._build_action_panel(right)

        ttk.Label(bottom, textvariable=self.var_status, anchor="w").grid(
            row=0, column=0, sticky="ew"
        )

    def _build_parameter_panel(self, parent: ttk.Frame) -> None:
        """Build the parameter panel from the shared form state."""
        build_params_panel(parent, self.params_form)

    def _build_preview_panel(self, parent: ttk.Frame) -> None:
        self.figure = Figure(figsize=(7, 7), dpi=100)
        self.preview_ax = self.figure.add_subplot(111)
        self.preview_ax.set_title("No file loaded")
        self.preview_ax.axis("off")

        self.preview_canvas = FigureCanvasTkAgg(self.figure, master=parent)
        widget = self.preview_canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")

        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        slider_frame.columnconfigure(1, weight=1)

        ttk.Label(slider_frame, text="Frame").grid(row=0, column=0, sticky="w")
        self.slider_frame = tk.Scale(
            slider_frame,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.var_frame,
            command=self._on_frame_slider,
        )
        self.slider_frame.grid(row=0, column=1, sticky="ew")

    def _build_action_panel(self, parent: ttk.Frame) -> None:
        self.btn_load = ttk.Button(parent, text="Load file", command=self.on_load_file)
        self.btn_load_piv_result = ttk.Button(
            parent, text="Load PIV result", command=self.on_load_piv_result
        )
        self.btn_load_piv_result.grid(row=1, column=0, sticky="ew", pady=4)
        self.btn_load.grid(row=0, column=0, sticky="ew", pady=4)

        self.btn_single = ttk.Button(
            parent, text="Run single PIV", command=self.on_run_single
        )
        self.btn_single.grid(row=2, column=0, sticky="ew", pady=4)

        self.btn_batch = ttk.Button(
            parent, text="Run batch PIV", command=self.on_run_batch
        )
        self.btn_batch.grid(row=3, column=0, sticky="ew", pady=4)

        self.btn_pause = ttk.Button(
            parent, text="Pause", command=self.on_pause_batch
        )
        self.btn_pause.grid(row=4, column=0, sticky="ew", pady=4)
        self.btn_pause.config(state="disabled")

        self.btn_continue = ttk.Button(
            parent, text="Continue", command=self.on_continue_batch
        )
        self.btn_continue.grid(row=5, column=0, sticky="ew", pady=4)
        self.btn_continue.config(state="disabled")

        self.btn_abort = ttk.Button(
            parent, text="Abort", command=self.on_abort_batch
        )
        self.btn_abort.grid(row=6, column=0, sticky="ew", pady=4)
        self.btn_abort.config(state="disabled")

        self.btn_export = ttk.Button(
            parent, text="Export current result", command=self.on_export_current
        )
        self.btn_export.grid(row=7, column=0, sticky="ew", pady=4)

        self.btn_export_video = ttk.Button(
            parent, text="Export video / GIF", command=self.on_export_animation
        )
        self.btn_export_video.grid(row=8, column=0, sticky="ew", pady=4)

        ttk.Separator(parent, orient="horizontal").grid(
            row=9, column=0, sticky="ew", pady=(12, 8)
        )

        ttk.Label(parent, text="Batch progress").grid(
            row=10, column=0, sticky="w", pady=(0, 4)
        )
        self.progress = ttk.Progressbar(parent, mode="determinate", length=220)
        self.progress.grid(row=11, column=0, sticky="ew")
        
    def _set_batch_running_state(self) -> None:
        """Disable normal actions and enable batch controls during a running batch."""
        self.btn_load.config(state="disabled")
        self.btn_load_piv_result.config(state="disabled")
        self.btn_single.config(state="disabled")
        self.btn_batch.config(state="disabled")
        self.btn_export.config(state="disabled")
        self.btn_export_video.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_continue.config(state="disabled")
        self.btn_abort.config(state="normal")

    def _set_batch_idle_state(self) -> None:
        """Restore normal actions after a batch stops or finishes."""
        self.btn_load.config(state="normal")
        self.btn_load_piv_result.config(state="normal")
        if self.loaded_stack is None:
            self._set_idle_state()
        else:
            self._set_loaded_state()
            self.btn_export.config(
                state="normal" if self.current_result is not None else "disabled"
            )
            self.btn_export_video.config(
                state="normal"
                if isinstance(self.current_result, BatchPIVResult)
                else "disabled"
            )
        self.btn_pause.config(state="disabled")
        self.btn_continue.config(state="disabled")
        self.btn_abort.config(state="disabled")
    
    def _set_idle_state(self) -> None:
        self.btn_single.config(state="disabled")
        self.btn_batch.config(state="disabled")
        self.btn_export.config(state="disabled")
        self.btn_export_video.config(state="disabled")

    def _set_loaded_state(self) -> None:
        self.btn_single.config(state="normal")
        self.btn_batch.config(state="normal")
        self.btn_export_video.config(state="disabled")

    def _show_loaded_frame(self, frame_index: int) -> None:
        if self.loaded_stack is None:
            return

        frame = self.loaded_stack.data[frame_index]
        draw_loaded_frame(
            self.preview_ax,
            self.preview_canvas,
            self.preview_state,
            frame,
            title=f"Frame {frame_index}",
        )

    def _show_pair_result(self, result: PIVPairResult, *, title: str) -> None:
        ensure_preview_artists(
            self.preview_ax,
            self.preview_canvas,
            self.preview_state,
            result.img1,
            result.img2,
            result.xg,
            result.yg,
            result.u,
            result.v,
            title=title,
        )

    def _set_result_view_state(self) -> None:
        """UI state for inspecting a saved PIV result without an image stack."""
        self.btn_single.config(state="disabled")
        self.btn_batch.config(state="disabled")
        self.btn_export.config(state="disabled")
        self.btn_pause.config(state="disabled")
        self.btn_continue.config(state="disabled")
        self.btn_abort.config(state="disabled")
        self.btn_export_video.config(
            state="normal"
            if self.loaded_piv_result is not None and self.loaded_piv_result.u.ndim == 3
            else "disabled"
        )

    def _loaded_piv_field_count(self) -> int:
        """Return the number of vector fields in the loaded saved result."""
        if self.loaded_piv_result is None:
            return 0

        if self.loaded_piv_result.u.ndim == 2:
            return 1
        if self.loaded_piv_result.u.ndim == 3:
            return int(self.loaded_piv_result.u.shape[0])

        raise ValueError(
            f"Unsupported saved U shape: {self.loaded_piv_result.u.shape}"
        )

    def _show_loaded_piv_result(self, field_index: int) -> None:
        """Display a saved vector field without requiring an image stack."""
        if self.loaded_piv_result is None:
            return

        result = self.loaded_piv_result

        if result.u.ndim == 2:
            u = result.u
            v = result.v
            title = f"Loaded PIV result: {result.source_path.name}"
        elif result.u.ndim == 3:
            idx = max(0, min(field_index, result.u.shape[0] - 1))
            u = result.u[idx]
            v = result.v[idx]
            title = f"Loaded PIV result: Field {idx}"
        else:
            raise ValueError(
                f"Unsupported saved U shape: {result.u.shape}"
            )

        draw_vector_field_only(
            self.preview_ax,
            self.preview_canvas,
            self.preview_state,
            result.xg,
            result.yg,
            u,
            v,
            title=title,
        )
    
    def _show_result_for_frame_index(self, frame_index: int) -> None:
        """Show the appropriate result view for the current slider position."""
        if self.loaded_piv_result is not None:
            self._show_loaded_piv_result(frame_index)
            return

        if self.current_result is None:
            self._show_loaded_frame(frame_index)
            return

        if isinstance(self.current_result, BatchPIVResult):
            if not self.current_result.pair_results:
                self._show_loaded_frame(frame_index)
                return

            pair_index = max(
                0,
                min(frame_index, len(self.current_result.pair_results) - 1),
            )
            pair_result = self.current_result.pair_results[pair_index]
            self._show_pair_result(
                pair_result,
                title=f"Batch PIV: Frame {pair_index} → {pair_index + 1}",
            )
            return

        if self.current_single_pair_indices is not None:
            t1, t2 = self.current_single_pair_indices
            title = f"Single PIV: Frame {t1} → {t2}"
        else:
            title = "Single PIV result"

        self._show_pair_result(self.current_result, title=title)

    def _ask_batch_run_options(self) -> BatchRunOptions | None:
        """Show the batch-run options dialog."""
        dialog = BatchRunDialog(self.root)
        return dialog.result

    def _ask_batch_export_path(self, *, file_format: str) -> Path | None:
        """Ask for the export path before starting the batch run."""
        ext = ".npz" if file_format == "npz" else ".h5"
        filetypes = (
            [("NumPy zipped", "*.npz")]
            if file_format == "npz"
            else [("HDF5", "*.h5")]
        )

        save_path = filedialog.asksaveasfilename(
            title="Export batch PIV result",
            initialfile=self._build_export_name_hint(mode="batch"),
            defaultextension=ext,
            filetypes=filetypes,
        )
        if not save_path:
            return None
        return Path(save_path)

    def _export_batch_result_direct(
        self,
        result: BatchPIVResult,
        out_path: Path,
    ) -> None:
        """Export a finished batch result directly to the chosen path."""
        export_path = save_batch_result(out_path, result)
        self._set_status("Export complete", 3000)
        self.var_result.set(f"Saved: {export_path.path.name}")

    def _on_frame_slider(self, _value: str) -> None:
        if self.loaded_stack is None and self.loaded_piv_result is None:
            return
        self._show_result_for_frame_index(int(self.var_frame.get()))
        
    def on_load_file(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.tif *.tiff *.h5")]
        )
        if not path:
            return

        try:
            loaded = load_stack(path)
            self.loaded_stack = loaded
            self.loaded_piv_result = None
            self.current_result = None
            self.current_single_pair_indices = None
            self.current_export_name_hint = loaded.source_path.stem
            reset_preview_state(self.preview_state)

            self.var_file_name.set(f"File: {loaded.source_path.name}")
            self.var_result.set(
                f"Loaded: {loaded.source_path.name} – Shape: {loaded.shape}"
            )
            self._set_status("File loaded", 3000)

            self.slider_frame.config(to=max(loaded.num_frames - 1, 0))
            self.var_frame.set(0)
            self._show_loaded_frame(0)
            self._set_loaded_state()
            self.btn_export.config(state="disabled")
            self.btn_export_video.config(state="disabled")
            self.progress["value"] = 0
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self.var_result.set(f"Loading failed: {exc}")
            self._set_status("Load failed")
    
    def on_load_piv_result(self) -> None:
        """Load a saved PIV result for vector-field-only inspection."""
        path = filedialog.askopenfilename(
            filetypes=[("PIV result files", "*.npz *.h5")]
        )
        if not path:
            return

        try:
            loaded = load_saved_piv_result(path)

            self.loaded_stack = None
            self.loaded_piv_result = loaded
            self.current_result = None
            self.current_single_pair_indices = None
            self.current_export_name_hint = loaded.source_path.stem
            reset_preview_state(self.preview_state)

            self.var_file_name.set(f"PIV result: {loaded.source_path.name}")
            self.var_result.set(
                f"Loaded PIV result: {loaded.source_path.name} – "
                f"U shape: {loaded.u.shape}"
            )
            self._set_status("PIV result loaded", 3000)

            field_count = self._loaded_piv_field_count()
            self.slider_frame.config(to=max(field_count - 1, 0))
            self.var_frame.set(0)
            self._show_loaded_piv_result(0)
            self._set_result_view_state()
            self.btn_export_video.config(
                state="normal" if loaded.u.ndim == 3 else "disabled"
            )
            self.progress["value"] = 0

        except Exception as exc:
            messagebox.showerror("Load PIV result error", str(exc))
            self.var_result.set(f"Loading PIV result failed: {exc}")
            self._set_status("Load PIV result failed")

    def on_run_single(self) -> None:
        if self.loaded_stack is None:
            messagebox.showerror("Error", "No file loaded.")
            return

        num_frames = self.loaded_stack.num_frames
        t1 = simpledialog.askinteger(
            "Single PIV",
            f"Start frame (0-{num_frames - 2}):",
            minvalue=0,
            maxvalue=num_frames - 2,
            parent=self.root,
        )
        if t1 is None:
            return

        try:
            params = build_workflow_params(self.params_form)
            result = run_piv_pair(
                self.loaded_stack.data[t1],
                self.loaded_stack.data[t1 + 1],
                params=params,
            )
            self.current_result = result
            self.current_single_pair_indices = (t1, t1 + 1)
            self.current_export_name_hint = self._build_export_name_hint(
                mode="single",
                frame1_idx=t1,
                frame2_idx=t1 + 1,
            )
            self._show_pair_result(
                result,
                title=f"Single PIV: Frame {t1} → {t1 + 1}",
            )
            self.var_result.set(
                f"Single PIV complete: {t1} → {t1 + 1} | "
                f"grid={result.xg.shape} | SN={result.sn is not None}"
            )
            self._set_status("Single PIV complete", 3000)
            self.btn_export.config(state="normal")
            self.btn_export_video.config(state="disabled")
        except Exception as exc:
            messagebox.showerror("Run error", str(exc))
            self.var_result.set(f"Single PIV failed: {exc}")
            self._set_status("Single PIV failed")

    def on_run_batch(self) -> None:
        if self.loaded_stack is None:
            messagebox.showerror("Error", "No file loaded.")
            return

        options = self._ask_batch_run_options()
        if options is None:
            return

        export_path: Path | None = None
        if options.export_after_run:
            export_path = self._ask_batch_export_path(
                file_format=options.export_format
            )
            if export_path is None:
                return

        try:
            params = build_workflow_params(self.params_form)
        except Exception as exc:
            messagebox.showerror("Batch error", str(exc))
            self.var_result.set(f"Batch PIV failed: {exc}")
            self._set_status("Batch PIV failed")
            return

        total_pairs = self.loaded_stack.num_frames - 1
        self.batch.start(
            options=options,
            export_path=export_path,
            total_pairs=total_pairs,
            params=params,
        )

        self.progress["maximum"] = max(self.batch.total_pairs, 1)
        self.progress["value"] = 0

        self._set_batch_running_state()
        self._set_status("Batch started")
        self.root.after(1, self._run_next_batch_step)

    def _run_next_batch_step(self) -> None:
        """Run one batch step and schedule the next one."""
        if not self.batch.running:
            return

        if self.batch.abort_requested:
            self.batch.reset()
            self.current_result = None
            self.current_single_pair_indices = None
            self._set_batch_idle_state()
            self._set_status("Batch aborted", 3000)
            self.var_result.set("Batch aborted")
            if self.loaded_stack is not None:
                self._show_loaded_frame(int(self.var_frame.get()))
            return

        if self.batch.paused:
            self._set_status("Batch paused")
            return

        if self.loaded_stack is None or self.batch.params is None:
            self.batch.reset()
            self._set_batch_idle_state()
            self._set_status("Batch stopped")
            return

        if self.batch.is_finished():
            result = self.batch.build_batch_result()
            export_path = self.batch.export_path
            options = self.batch.options

            self.current_result = result
            self.current_single_pair_indices = None
            self.current_export_name_hint = self._build_export_name_hint(mode="batch")
            self.batch.reset()
            self._set_batch_idle_state()
            self.btn_export_video.config(state="normal")

            if (
                options is not None
                and options.export_after_run
                and export_path is not None
            ):
                try:
                    self._export_batch_result_direct(result, export_path)
                except Exception as exc:
                    messagebox.showerror("Export error", str(exc))
                    self.var_result.set(f"Batch export failed: {exc}")
                    self._set_status("Batch finished, export failed")
                return

            self.var_result.set(
                f"Batch PIV complete: {len(result.pair_results)} pairs | "
                f"grid={result.xg.shape if result.xg is not None else None} | "
                f"SN={result.sn_list is not None}"
            )
            self._set_status("Batch PIV complete", 3000)
            return

        t = self.batch.next_pair_index

        try:
            pair_result = run_piv_pair(
                self.loaded_stack.data[t],
                self.loaded_stack.data[t + 1],
                params=self.batch.params,
            )
            self.batch.append_result(pair_result)
            self.progress["value"] = self.batch.next_pair_index

            if self.batch.options is not None and self.batch.options.preview_mode == "live":
                self._show_pair_result(
                    pair_result,
                    title=f"Batch PIV: Frame {t} → {t + 1}",
                )

            self._set_status(
                f"Processed pair {self.batch.next_pair_index}/{self.batch.total_pairs}"
            )
            self.root.after(1, self._run_next_batch_step)

        except Exception as exc:
            self.batch.reset()
            self._set_batch_idle_state()
            messagebox.showerror("Batch error", str(exc))
            self.var_result.set(f"Batch PIV failed: {exc}")
            self._set_status("Batch PIV failed")

    def on_abort_batch(self) -> None:
        """Request that the running batch stop after the current step."""
        if not self.batch.running:
            return

        was_paused = self.batch.paused
        self.batch.abort_requested = True
        self.batch.paused = False
        self.btn_pause.config(state="disabled")
        self.btn_continue.config(state="disabled")
        self._set_status("Batch abort requested...")

        if was_paused:
            self.root.after(1, self._run_next_batch_step)

    def on_pause_batch(self) -> None:
        """Pause the running batch after the current step."""
        if not self.batch.running:
            return
        if self.batch.paused:
            return

        self.batch.paused = True
        self.btn_pause.config(state="disabled")
        self.btn_continue.config(state="normal")
        self._set_status("Batch paused")

    def on_continue_batch(self) -> None:
        """Resume a paused batch."""
        if not self.batch.paused:
            return

        self.batch.paused = False
        self.btn_pause.config(state="normal")
        self.btn_continue.config(state="disabled")
        self._set_status("Batch resuming...")
        self.root.after(1, self._run_next_batch_step)

    def on_export_current(self) -> None:
        if self.current_result is None:
            messagebox.showerror("Export error", "No result available for export.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Export PIV result",
            initialfile=self.current_export_name_hint,
            defaultextension=".npz",
            filetypes=[
                ("NumPy zipped", "*.npz"),
                ("HDF5", "*.h5"),
            ],
        )
        if not out_path:
            return

        try:
            if isinstance(self.current_result, PIVPairResult):
                export_path = save_pair_result(out_path, self.current_result)
            else:
                export_path = save_batch_result(out_path, self.current_result)

            self._set_status("Export complete", 3000)
            self.var_result.set(f"Saved: {export_path.path.name}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))
            self.var_result.set(f"Export failed: {exc}")
            self._set_status("Export failed")

    def on_export_animation(self) -> None:
        """Export the current batch-like vector field sequence as MP4 or GIF."""
        u = None
        v = None
        xg = None
        yg = None

        if isinstance(self.current_result, BatchPIVResult):
            if not self.current_result.pair_results:
                messagebox.showerror("Export video error", "No batch result available.")
                return
            u = self.current_result.u_list
            v = self.current_result.v_list
            xg = self.current_result.xg
            yg = self.current_result.yg

        elif self.loaded_piv_result is not None and self.loaded_piv_result.u.ndim == 3:
            u = self.loaded_piv_result.u
            v = self.loaded_piv_result.v
            xg = self.loaded_piv_result.xg
            yg = self.loaded_piv_result.yg

        else:
            messagebox.showerror(
                "Export video error",
                "Animation export requires a batch result or a loaded multi-field PIV result.",
            )
            return

        out_path = filedialog.asksaveasfilename(
            title="Export PIV animation",
            initialfile=self.current_export_name_hint,
            defaultextension=".mp4",
            filetypes=[
                ("MP4 Video", "*.mp4"),
                ("GIF Animation", "*.gif"),
            ],
        )
        if not out_path:
            return

        try:
            u_arr = u if isinstance(u, tuple) else u
            v_arr = v if isinstance(v, tuple) else v

            if isinstance(u_arr, list):
                import numpy as _np
                u_arr = _np.stack(u_arr)
            if isinstance(v_arr, list):
                import numpy as _np
                v_arr = _np.stack(v_arr)

            export_path = save_piv_animation(
                out_path,
                u=u_arr,
                v=v_arr,
                xg=xg,
                yg=yg,
            )
            self._set_status("Video export complete", 3000)
            self.var_result.set(f"Saved animation: {export_path.path.name}")

        except Exception as exc:
            messagebox.showerror("Export video error", str(exc))
            self.var_result.set(f"Video export failed: {exc}")
            self._set_status("Video export failed")

    def _build_export_name_hint(
        self,
        *,
        mode: str,
        frame1_idx: int | None = None,
        frame2_idx: int | None = None,
    ) -> str:
        """Build the default export filename (without extension)."""
        if self.loaded_stack is None:
            stem = "piv_result"
        else:
            stem = self.loaded_stack.source_path.stem

        if mode == "single" and frame1_idx is not None and frame2_idx is not None:
            return f"{stem}_piv_{frame1_idx:04d}_to_{frame2_idx:04d}"
        if mode == "batch":
            return f"{stem}_batch_piv"
        return stem
    
    def show_piv_info(self) -> None:
        """Show an informational pop-up about the current PIV capabilities."""
        msg = (
            "This program is currently configured for 2D PIV analysis only.\n\n"
            "In principle, the code can be extended to 3D PIV by adding the "
            "required data handling and correlation routines.\n\n"
            "The current implementation uses an FFT-based windowed "
            "cross-correlation algorithm (via the multi_quickPIV library) "
            "for the PIV evaluation."
        )
        messagebox.showinfo("PIV Configuration Information", msg)

    def _set_status(self, text: str, timeout_ms: int | None = None) -> None:
        """Set the status-bar text, optionally clearing it after a delay."""
        if self._status_after_id is not None:
            try:
                self.root.after_cancel(self._status_after_id)
            except Exception:
                pass
            self._status_after_id = None

        self.var_status.set(text)

        if timeout_ms is not None and timeout_ms > 0:
            def _clear_status() -> None:
                self.var_status.set("")
                self._status_after_id = None

            self._status_after_id = self.root.after(timeout_ms, _clear_status)  


def main() -> None:
    """Entry point for the minimal GUI shell."""
    root = tk.Tk()
    app = MultiQuickPIVApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()