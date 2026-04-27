"""Preview and visualization helpers for frame and PIV display."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PreviewState:
    """Track the currently drawn preview artists and axes limits."""

    im1: object | None = None
    im2: object | None = None
    quiv: object | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None


def reset_preview_state(state: PreviewState) -> None:
    """Clear the tracked preview artists."""
    state.im1 = None
    state.im2 = None
    state.quiv = None
    state.xlim = None
    state.ylim = None


def draw_loaded_frame(ax, canvas, state: PreviewState, frame: np.ndarray, *, title: str) -> None:
    """Draw a single loaded frame without vector overlays."""
    reset_preview_state(state)

    ax.clear()
    ax.set_aspect("equal")
    ax.imshow(frame, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

    canvas.draw()


def draw_vector_field_only(
    ax,
    canvas,
    state: PreviewState,
    xg: np.ndarray,
    yg: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    title: str,
) -> None:
    """Draw a saved vector field without requiring an image stack."""
    reset_preview_state(state)

    ax.clear()
    ax.set_aspect("equal")

    quiv = ax.quiver(
        xg,
        yg,
        v,
        u,
        color="red",
        scale=87,
        width=0.004,
    )

    ax.set_xlim(float(np.min(xg)) - 10, float(np.max(xg)) + 10)
    ax.set_ylim(float(np.min(yg)) - 10, float(np.max(yg)) + 10)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    canvas.draw()

    state.quiv = quiv
    state.xlim = ax.get_xlim()
    state.ylim = ax.get_ylim()


def ensure_preview_artists(
    ax,
    canvas,
    state: PreviewState,
    img1: np.ndarray,
    img2: np.ndarray,
    xg: np.ndarray,
    yg: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    title: str,
) -> None:
    """Create or re-create the preview: two images overlaid with a quiver plot."""
    reset_preview_state(state)

    ax.clear()
    ax.set_aspect("equal")

    im1 = ax.imshow(img1, cmap="Reds")
    im2 = ax.imshow(img2, cmap="Blues", alpha=0.5)

    quiv = ax.quiver(
        xg,
        yg,
        v,
        u,
        color="red",
        scale=87,
        width=0.004,
    )

    ax.set_xlim(float(np.min(xg)) - 10, float(np.max(xg)) + 10)
    ax.set_ylim(float(np.min(yg)) - 10, float(np.max(yg)) + 10)
    ax.set_title(title)
    ax.axis("off")

    canvas.draw()

    state.im1 = im1
    state.im2 = im2
    state.quiv = quiv
    state.xlim = ax.get_xlim()
    state.ylim = ax.get_ylim()


def update_preview_artists(
    ax,
    canvas,
    state: PreviewState,
    img1: np.ndarray,
    img2: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    title: str,
) -> bool:
    """Update existing preview artists in place."""
    if state.im1 is None or state.im2 is None or state.quiv is None:
        return False

    state.im1.set_data(img1)
    state.im2.set_data(img2)
    state.quiv.set_UVC(v, u)

    ax.set_title(title)

    if state.xlim is not None and state.ylim is not None:
        ax.set_xlim(state.xlim)
        ax.set_ylim(state.ylim)

    canvas.draw()
    return True