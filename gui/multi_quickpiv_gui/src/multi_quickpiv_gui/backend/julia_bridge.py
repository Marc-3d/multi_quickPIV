"""Bridge to the Julia multi_quickPIV backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import subprocess

import numpy as np

_JL = None
_J = None


@dataclass(slots=True)
class JuliaPIVResult:
    """Result returned from one Julia-backed PIV computation."""

    u: np.ndarray
    v: np.ndarray
    xg: np.ndarray
    yg: np.ndarray
    sn: np.ndarray | None = None


def _repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[3]


def _julia_env_dir() -> Path:
    """Return the local Julia environment directory."""
    return _repo_root() / "julia_env"


def _ensure_julia_bindir_on_path() -> None:
    """
    Ensure Julia's real bin directory is on PATH.

    This mirrors the working quick_PIV_GUI_v3.3 approach for Windows + JuliaUp
    embedded via PyJulia.
    """
    julia_exe_override = os.environ.get("JULIA_EXE")
    if julia_exe_override and os.path.exists(julia_exe_override):
        julia_bin = os.path.dirname(julia_exe_override)
        os.environ["PATH"] = julia_bin + os.pathsep + os.environ.get("PATH", "")
        return

    julia_cmd = shutil.which("julia")
    if not julia_cmd:
        return

    try:
        bindir = subprocess.check_output(
            [julia_cmd, "-e", "print(Sys.BINDIR)"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        if bindir and os.path.isdir(bindir):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
            return
    except Exception:
        pass

    os.environ["PATH"] = (
        os.path.dirname(julia_cmd) + os.pathsep + os.environ.get("PATH", "")
    )


def ensure_julia_initialized() -> None:
    """Initialize the embedded Julia runtime and load multi_quickPIV."""
    global _JL, _J

    if _JL is not None and _J is not None:
        return

    _ensure_julia_bindir_on_path()

    import julia
    from julia import Main as J

    try:
        JL = julia.Julia(compiled_modules=True)
    except Exception:
        JL = julia.Julia(compiled_modules=False)

    J.JULIA_ENV_DIR = str(_julia_env_dir())

    J.eval(
        """
using Pkg
Pkg.activate(JULIA_ENV_DIR)
try
    Pkg.instantiate()
catch err
    @warn "Pkg.instantiate() failed (continuing)" exception=(err, catch_backtrace())
end

try
    using multi_quickPIV
catch
    @warn "multi_quickPIV missing, installing..."
    using Pkg
    Pkg.add(url="https://github.com/Marc-3d/multi_quickPIV.git")
    using multi_quickPIV
end

function run_piv(img1::Array{Float64,2}, img2::Array{Float64,2};
                 corr_alg="nsqecc", interSize=(64,64), searchMargin=(128,128),
                 step=(32,32), computeSN=true)

    pivparams = multi_quickPIV.setPIVParameters(
        corr_alg=corr_alg,
        interSize=interSize,
        searchMargin=searchMargin,
        step=step,
        computeSN=computeSN,
    )

    VF, SN = multi_quickPIV.PIV(img1, img2, pivparams)
    U = VF[1, :, :]
    V = VF[2, :, :]
    vfsize = size(U)
    stepv = multi_quickPIV._step(pivparams)[1:2]
    isize = multi_quickPIV._isize(pivparams)[1:2]
    xgrid = [(x - 1) * stepv[2] + div(isize[2], 2) for y in 1:vfsize[1], x in 1:vfsize[2]]
    ygrid = [(y - 1) * stepv[1] + div(isize[1], 2) for y in 1:vfsize[1], x in 1:vfsize[2]]
    return -U, V, xgrid, ygrid, SN
end
"""
    )

    _JL = JL
    _J = J


def run_piv(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    inter_size: tuple[int, int] = (64, 64),
    search_margin: tuple[int, int] = (128, 128),
    step: tuple[int, int] = (32, 32),
    compute_sn: bool = True,
    corr_alg: str = "nsqecc",
) -> JuliaPIVResult:
    """Run one PIV computation through the embedded Julia backend."""
    ensure_julia_initialized()

    assert _J is not None

    _J.img1 = np.asarray(img1, dtype=np.float64)
    _J.img2 = np.asarray(img2, dtype=np.float64)
    _J.corr_alg = corr_alg

    _J.eval(
        f"U_, V_, xg_, yg_, SN_ = run_piv("
        f"img1, img2; "
        f"corr_alg=corr_alg, "
        f"interSize=({inter_size[0]}, {inter_size[1]}), "
        f"searchMargin=({search_margin[0]}, {search_margin[1]}), "
        f"step=({step[0]}, {step[1]}), "
        f"computeSN={'true' if compute_sn else 'false'})"
    )

    u = np.array(_J.eval("U_"))
    v = np.array(_J.eval("V_"))
    xg = np.array(_J.eval("xg_"))
    yg = np.array(_J.eval("yg_"))

    sn = None
    if compute_sn:
        sn = np.array(_J.eval("SN_"))
        if sn.size == 0:
            raise RuntimeError(
                "SN_ came back empty from Julia while computeSN is enabled."
            )

    return JuliaPIVResult(u=u, v=v, xg=xg, yg=yg, sn=sn)