"""Minimalist matplotlib styling helpers."""

from matplotlib.axes import Axes


def apply_minimalist_style(ax: Axes) -> None:
    """Apply minimalist style: hide top/right spines, offset left/bottom."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
