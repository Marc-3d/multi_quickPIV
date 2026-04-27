"""GUI-independent backend post-processing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multi_quickpiv_gui.workflow.params import WorkflowParams


@dataclass(slots=True)
class PostProcessResult:
    """Processed vector field plus metadata about the applied filters."""

    u: np.ndarray
    v: np.ndarray
    sn: np.ndarray | None
    sn_replaced: int = 0


def median_despike(
    u: np.ndarray,
    v: np.ndarray,
    *,
    ksize: int = 3,
    threshold: float = 3.5,
    use_magnitude: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove outliers using a median filter."""
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")

    if ksize < 1:
        raise ValueError("Median-despike window size must be at least 1.")
    if ksize % 2 == 0:
        ksize += 1

    half = ksize // 2
    height, width = u.shape

    u_clean = u.copy()
    v_clean = v.copy()

    u_pad = np.pad(u, ((half, half), (half, half)), mode="reflect")
    v_pad = np.pad(v, ((half, half), (half, half)), mode="reflect")

    for i in range(height):
        i0, i1 = i, i + ksize
        for j in range(width):
            j0, j1 = j, j + ksize
            w_u = u_pad[i0:i1, j0:j1]
            w_v = v_pad[i0:i1, j0:j1]

            if use_magnitude:
                w_m = np.hypot(w_u, w_v)
                med_m = np.median(w_m)
                mad_m = np.median(np.abs(w_m - med_m)) + 1e-9
                cur_m = np.hypot(u[i, j], v[i, j])

                if cur_m > med_m + threshold * mad_m:
                    u_clean[i, j] = np.median(w_u)
                    v_clean[i, j] = np.median(w_v)
            else:
                med_u = np.median(w_u)
                med_v = np.median(w_v)
                mad_u = np.median(np.abs(w_u - med_u)) + 1e-9
                mad_v = np.median(np.abs(w_v - med_v)) + 1e-9

                if (
                    abs(u[i, j] - med_u) > threshold * mad_u
                    or abs(v[i, j] - med_v) > threshold * mad_v
                ):
                    u_clean[i, j] = med_u
                    v_clean[i, j] = med_v

    return u_clean, v_clean


def sn_threshold_filter(
    u: np.ndarray,
    v: np.ndarray,
    sn: np.ndarray,
    *,
    sn_min: float,
    ksize: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Replace vectors with SN < sn_min using a local median.

    Returns the filtered (u, v) fields and the number of replaced vectors.
    """
    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")
    if sn.shape != u.shape:
        raise ValueError(f"SN shape {sn.shape} does not match vector shape {u.shape}.")
    if sn_min <= 0:
        raise ValueError("SN minimum threshold must be greater than 0.")

    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2

    u_out = u.copy()
    v_out = v.copy()

    mask = sn < sn_min
    n_replaced = int(np.sum(mask))
    if n_replaced == 0:
        return u_out, v_out, 0

    u_pad = np.pad(u, ((half, half), (half, half)), mode="reflect")
    v_pad = np.pad(v, ((half, half), (half, half)), mode="reflect")

    ys, xs = np.where(mask)
    for i, j in zip(ys, xs):
        w_u = u_pad[i:i + ksize, j:j + ksize]
        w_v = v_pad[i:i + ksize, j:j + ksize]
        u_out[i, j] = np.median(w_u)
        v_out[i, j] = np.median(w_v)

    return u_out, v_out, n_replaced


def apply_postprocessing(
    u: np.ndarray,
    v: np.ndarray,
    *,
    params: WorkflowParams,
    sn: np.ndarray | None = None,
) -> PostProcessResult:
    """
    Apply all configured post-processing steps to a computed vector field.
    """
    params.validate()

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape.")

    u_out = np.asarray(u, dtype=np.float64).copy()
    v_out = np.asarray(v, dtype=np.float64).copy()
    sn_out = None if sn is None else np.asarray(sn, dtype=np.float64).copy()

    post = params.postprocess

    if post.median_despike.enabled:
        u_out, v_out = median_despike(
            u_out,
            v_out,
            ksize=post.median_despike.ksize,
            threshold=post.median_despike.threshold,
            use_magnitude=True,
        )

    sn_replaced = 0
    if post.sn_filter.enabled:
        if sn_out is None:
            raise ValueError(
                "SN filtering is enabled, but no SN array was provided."
            )

        u_out, v_out, sn_replaced = sn_threshold_filter(
            u_out,
            v_out,
            sn_out,
            sn_min=post.sn_filter.minimum,
            ksize=post.median_despike.ksize,
        )

    return PostProcessResult(
        u=u_out,
        v=v_out,
        sn=sn_out,
        sn_replaced=sn_replaced,
    )