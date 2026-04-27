"""Input/output helpers for image stacks and saved PIV results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tifffile


@dataclass(slots=True)
class LoadedStack:
    """Loaded image stack plus minimal metadata."""

    data: np.ndarray
    source_path: Path
    dataset_name: str | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the stack shape."""
        return self.data.shape

    @property
    def num_frames(self) -> int:
        """Return the number of frames in the stack."""
        return int(self.data.shape[0])


@dataclass(slots=True)
class LoadedPIVResult:
    """Loaded saved PIV arrays from NPZ or HDF5."""

    u: np.ndarray
    v: np.ndarray
    xg: np.ndarray
    yg: np.ndarray
    sn: np.ndarray | None = None
    source_path: Path | None = None


def robust_read_h5(path: str | Path) -> tuple[np.ndarray, str]:
    """
    Read an HDF5 file and choose a reasonable default dataset name.

    Mirrors the current GUI behavior:
    - prefer 'tiff_data' if present
    - otherwise use the first dataset key
    """
    path_obj = Path(path)

    with h5py.File(path_obj, "r") as hf:
        if "tiff_data" in hf:
            key = "tiff_data"
        else:
            key = next(iter(hf.keys()))
        data = np.array(hf[key])

    return data, key


def normalize_stack(data: np.ndarray) -> np.ndarray:
    """
    Normalize loaded image data into a stack with shape (T, H, W).

    Current behavior follows the legacy GUI:
    - if input is 2D, convert it to a 1-frame stack
    - require at least 2 frames for PIV use
    """
    arr = np.asarray(data)

    if arr.ndim == 2:
        arr = arr[None, :, :]

    if arr.ndim != 3:
        raise ValueError(
            f"Expected image stack with 2 or 3 dimensions, got shape {arr.shape}."
        )

    if arr.shape[0] < 2:
        raise ValueError("At least 2 frames are required for PIV.")

    return arr


def load_stack(path: str | Path) -> LoadedStack:
    """
    Load a TIFF/TIFF or HDF5 stack and normalize it for PIV use.
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        data = tifffile.imread(path_obj)
        dataset_name = None
    elif suffix == ".h5":
        data, dataset_name = robust_read_h5(path_obj)
    else:
        raise ValueError("Invalid file type. Supported: .tif, .tiff, .h5")

    normalized = normalize_stack(data)
    return LoadedStack(
        data=normalized,
        source_path=path_obj,
        dataset_name=dataset_name,
    )


def load_saved_piv_result(path: str | Path) -> LoadedPIVResult:
    """
    Load a previously saved PIV result from NPZ or HDF5.
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".npz":
        with np.load(path_obj) as data:
            u = np.array(data["U"])
            v = np.array(data["V"])
            xg = np.array(data["xgrid"])
            yg = np.array(data["ygrid"])
            sn = np.array(data["SN"]) if "SN" in data else None

    elif suffix == ".h5":
        with h5py.File(path_obj, "r") as hf:
            u = np.array(hf["U"])
            v = np.array(hf["V"])
            xg = np.array(hf["xgrid"])
            yg = np.array(hf["ygrid"])
            sn = np.array(hf["SN"]) if "SN" in hf else None
    else:
        raise ValueError("Invalid result file type. Supported: .npz, .h5")

    return LoadedPIVResult(
        u=u,
        v=v,
        xg=xg,
        yg=yg,
        sn=sn,
        source_path=path_obj,
    )