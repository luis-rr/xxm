"""Lines drawn as chunks, so that overlapping parts of a trajectory stack alpha.

A line drawn with `ax.plot` is one path: where a long trajectory crosses
itself, its alpha does not accumulate. Split into chunks that share their
endpoints and drawn as one `LineCollection`, crossings between chunks stack,
which reveals where the trajectory spends its time. Crossings within one chunk
still do not, so the number of chunks trades density resolution for drawing
cost; a fixed count (`DEFAULT_NUM_CHUNKS`) keeps the cost bounded however long
the line is.

Data (`chunk_bounds`, `chunk_line`) is separate from drawing
(`plot_chunked_line`). The module depends only on numpy and matplotlib.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

DEFAULT_NUM_CHUNKS = 100


def chunk_bounds(
    valid,
    *,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    breaks=None,
) -> np.ndarray:
    """Inclusive index ranges `(K, 2)` of the chunks of a line, in order.

    `valid` `(T,)` marks the samples that can be drawn (e.g. finite ones).
    The line is cut at about `num_chunks` evenly spaced indices, at the
    `breaks` indices, and around invalid samples; chunks that contain an
    invalid sample are dropped. Consecutive chunks of a valid run share their
    boundary sample, so the drawn line has no gaps. Every chunk has at least
    two samples.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 1:
        raise ValueError(f'valid must have shape (T,); got {valid.shape}')
    if num_chunks < 1:
        raise ValueError(f'num_chunks must be at least 1; got {num_chunks}')
    num_samples = len(valid)
    if num_samples < 2:
        return np.zeros((0, 2), dtype=int)

    even = np.linspace(0, num_samples - 1, min(num_chunks, num_samples - 1) + 1)
    changes = np.flatnonzero(np.diff(valid.astype(int)))  # valid[i] != valid[i + 1]
    edges = np.concatenate(
        [
            np.rint(even).astype(int),
            changes,
            changes + 1,
            [] if breaks is None else np.asarray(breaks, dtype=int).ravel(),
        ]
    )
    edges = np.unique(edges[(edges >= 0) & (edges < num_samples)].astype(int))

    starts, stops = edges[:-1], edges[1:]
    invalid_before = np.concatenate([[0], np.cumsum(~valid)])  # invalid in [0, i)
    complete = invalid_before[stops + 1] - invalid_before[starts] == 0
    return np.stack([starts, stops], axis=-1)[complete]


def chunk_line(points, bounds) -> list[np.ndarray]:
    """The chunks `points[start : stop + 1]` of `(T, D)` points, one per bound."""
    points = np.asarray(points)
    return [points[start : stop + 1] for start, stop in np.asarray(bounds)]


def plot_chunked_line(
    ax: Axes,
    points,
    *,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
    breaks=None,
    **kwargs,
) -> LineCollection:
    """Draw a `(T, 2)` line as chunks in one `LineCollection`; return it.

    Non-finite samples leave gaps. `kwargs` go to `LineCollection` (e.g.
    `color`, `linewidth`, `alpha`, `zorder`, `label`); caps default to
    `'butt'`. The data limits are updated and the view autoscaled, as with
    `ax.plot`.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f'points must have shape (T, 2); got {points.shape}')
    bounds = chunk_bounds(
        np.all(np.isfinite(points), axis=-1), num_chunks=num_chunks, breaks=breaks
    )
    kwargs.setdefault('capstyle', 'butt')
    lines = LineCollection(chunk_line(points, bounds), **kwargs)
    ax.add_collection(lines)
    ax.autoscale_view()
    return lines
