"""Export helpers for NPZ/HDF5 and animated PIV outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import h5py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

try:
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None

from multi_quickpiv_gui.workflow.pipeline import BatchPIVResult, PIVPairResult


@dataclass(slots=True)
class ExportPath:
    """Path written by one export operation."""

    path: Path


def _normalize_export_path(path: str | Path) -> Path:
    """
    Normalize the requested export path.

    Supported suffixes:
    - .npz
    - .h5
    - .mp4
    - .gif

    If no suffix is provided, default to .npz.
    """
    out_path = Path(path)

    if out_path.suffix == "":
        out_path = out_path.with_suffix(".npz")

    suffix = out_path.suffix.lower()
    if suffix not in {".npz", ".h5", ".mp4", ".gif"}:
        raise ValueError("Unsupported export format. Use .npz, .h5, .mp4, or .gif")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def save_piv_arrays(
    out_path: str | Path,
    *,
    u: np.ndarray,
    v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    sn: np.ndarray | None = None,
) -> ExportPath:
    """
    Save one PIV result payload in the format selected by the file suffix.

    Supported output formats:
    - .npz
    - .h5
    """
    out_path = _normalize_export_path(out_path)

    u_arr = np.asarray(u)
    v_arr = np.asarray(v)
    xg_arr = np.asarray(xg)
    yg_arr = np.asarray(yg)

    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v must have the same shape.")

    sn_arr: np.ndarray | None = None
    if sn is not None:
        sn_arr = np.asarray(sn)
        if sn_arr.shape != u_arr.shape:
            raise ValueError("sn must have the same shape as u and v.")

    suffix = out_path.suffix.lower()

    if suffix == ".npz":
        save_kwargs = {
            "U": u_arr,
            "V": v_arr,
            "xgrid": xg_arr,
            "ygrid": yg_arr,
        }
        if sn_arr is not None:
            save_kwargs["SN"] = sn_arr

        np.savez(out_path, **save_kwargs)

    elif suffix == ".h5":
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("U", data=u_arr)
            hf.create_dataset("V", data=v_arr)
            hf.create_dataset("xgrid", data=xg_arr)
            hf.create_dataset("ygrid", data=yg_arr)
            if sn_arr is not None:
                hf.create_dataset("SN", data=sn_arr)

    else:
        raise ValueError("save_piv_arrays only supports .npz and .h5 outputs.")

    return ExportPath(path=out_path)


def save_pair_result(
    out_path: str | Path,
    result: PIVPairResult,
) -> ExportPath:
    """Save one frame-pair result in the selected format."""
    return save_piv_arrays(
        out_path,
        u=result.u,
        v=result.v,
        xg=result.xg,
        yg=result.yg,
        sn=result.sn,
    )


def save_batch_result(
    out_path: str | Path,
    result: BatchPIVResult,
) -> ExportPath:
    """Save one batch result in the selected format."""
    if not result.pair_results:
        raise ValueError("Cannot export an empty batch result.")

    u = np.stack(result.u_list)
    v = np.stack(result.v_list)

    sn_list = result.sn_list
    sn = np.stack(sn_list) if sn_list is not None else None

    xg = result.xg
    yg = result.yg
    if xg is None or yg is None:
        raise ValueError("Batch result is missing grid coordinates.")

    return save_piv_arrays(
        out_path,
        u=u,
        v=v,
        xg=xg,
        yg=yg,
        sn=sn,
    )


def save_piv_animation(
    out_path: str | Path,
    *,
    u: np.ndarray,
    v: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
) -> ExportPath:
    """
    Save a quiver-only PIV animation as MP4 or GIF.

    Expected shapes:
    - u, v: (T, H, W)
    - xg, yg: (H, W)
    """
    out_path = _normalize_export_path(out_path)

    u_arr = np.asarray(u)
    v_arr = np.asarray(v)
    xg_arr = np.asarray(xg)
    yg_arr = np.asarray(yg)

    if u_arr.ndim != 3 or v_arr.ndim != 3:
        raise ValueError("Animation export requires u and v with shape (T, H, W).")
    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v must have the same shape.")
    if xg_arr.shape != yg_arr.shape:
        raise ValueError("xg and yg must have the same shape.")
    if xg_arr.shape != u_arr.shape[1:]:
        raise ValueError("xg/yg shape must match the spatial shape of u/v.")

    fig, ax = plt.subplots(figsize=(6, 6))

    try:
        ax.set_aspect("equal")
        ax.set_xlim(float(np.min(xg_arr)), float(np.max(xg_arr)))
        ax.set_ylim(float(np.min(yg_arr)), float(np.max(yg_arr)))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("PIV: Field 0")

        quiv = ax.quiver(
            xg_arr,
            yg_arr,
            v_arr[0],
            u_arr[0],
            scale=87,
            width=0.004,
        )

        def update(frame_index: int):
            quiv.set_UVC(v_arr[frame_index], u_arr[frame_index])
            ax.set_title(f"PIV: Field {frame_index}")
            return (quiv,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=u_arr.shape[0],
            interval=100,
            blit=True,
        )

        suffix = out_path.suffix.lower()

        if suffix == ".mp4":
            ffmpeg_exe = None

            if imageio_ffmpeg is not None:
                try:
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ffmpeg_exe = None

            if ffmpeg_exe is None:
                ffmpeg_exe = shutil.which("ffmpeg")

            if ffmpeg_exe is None:
                gif_path = out_path.with_suffix(".gif")
                writer = animation.PillowWriter(fps=10)
                ani.save(str(gif_path), writer=writer)
                return ExportPath(path=gif_path)

            matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
            writer = animation.FFMpegWriter(
                fps=10,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p"],
            )
            ani.save(str(out_path), writer=writer, dpi=150)

        elif suffix == ".gif":
            writer = animation.PillowWriter(fps=10)
            ani.save(str(out_path), writer=writer)

        else:
            raise ValueError("Animation export only supports .mp4 and .gif outputs.")

    finally:
        plt.close(fig)

    return ExportPath(path=out_path)