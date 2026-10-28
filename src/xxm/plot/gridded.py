"""Matplotlib plots for finite gridded probability distributions.

All functions accept unbatched distributions and an explicit Matplotlib Axes.
Colors represent cell-averaged *probability density*, not cell mass.
The plotting layer does not renormalize or alter the gridded distribution.
"""

from typing import Any, Literal, overload

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.patches import StepPatch
from numpy.typing import ArrayLike, NDArray

from ..core.dists.gridded import GriddedDistribution, TemporalGriddedDistribution

Kind1D = Literal['stairs', 'fill']
Kind2D = Literal['mesh', 'contour', 'contourf']

__all__ = (
    'plot_gridded_1d',
    'plot_gridded_1d_fill',
    'plot_gridded_1d_stairs',
    'plot_gridded_2d',
    'plot_gridded_2d_contour',
    'plot_gridded_2d_contourf',
    'plot_gridded_2d_mesh',
    'plot_temporal_gridded_1d',
    'plot_temporal_gridded_1d_contour',
    'plot_temporal_gridded_1d_contourf',
    'plot_temporal_gridded_1d_mesh',
)


def _edges_2d(
    dist: GriddedDistribution,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return np.asarray(dist.edges[0], dtype=float), np.asarray(
        dist.edges[1], dtype=float
    )


def _centers(edges: NDArray[np.float64]) -> NDArray[np.float64]:
    return (edges[:-1] + edges[1:]) / 2


def _mesh(
    ax: Axes,
    xedges: NDArray[np.float64],
    yedges: NDArray[np.float64],
    density: NDArray[np.float64],
    *args: Any,
    **kwargs: Any,
) -> QuadMesh:
    return ax.pcolormesh(xedges, yedges, density.T, *args, **kwargs)


def _contour(
    ax: Axes,
    xedges: NDArray[np.float64],
    yedges: NDArray[np.float64],
    density: NDArray[np.float64],
    *args: Any,
    **kwargs: Any,
) -> QuadContourSet:
    return ax.contour(_centers(xedges), _centers(yedges), density.T, *args, **kwargs)


def _contourf(
    ax: Axes,
    xedges: NDArray[np.float64],
    yedges: NDArray[np.float64],
    density: NDArray[np.float64],
    *args: Any,
    **kwargs: Any,
) -> QuadContourSet:
    return ax.contourf(_centers(xedges), _centers(yedges), density.T, *args, **kwargs)


# 1D: spatial density as a step plot.


def plot_gridded_1d_stairs(
    ax: Axes,
    dist: GriddedDistribution,
    **kwargs: Any,
) -> StepPatch:
    """Plot a 1D cell-average density as an unfilled step curve."""
    return ax.stairs(
        np.asarray(dist.density()),
        np.asarray(dist.edges[0]),
        fill=False,
        **kwargs,
    )


def plot_gridded_1d_fill(
    ax: Axes,
    dist: GriddedDistribution,
    **kwargs: Any,
) -> StepPatch:
    """Plot a filled 1D step density."""
    return ax.stairs(
        np.asarray(dist.density()),
        np.asarray(dist.edges[0]),
        fill=True,
        **kwargs,
    )


def plot_gridded_1d(
    ax: Axes,
    dist: GriddedDistribution,
    *,
    kind: Kind1D = 'stairs',
    **kwargs: Any,
) -> StepPatch:
    """Dispatch to the step or filled-step 1D plot."""
    if kind == 'stairs':
        return plot_gridded_1d_stairs(ax, dist, **kwargs)
    if kind == 'fill':
        return plot_gridded_1d_fill(ax, dist, **kwargs)
    raise ValueError(f'Unknown plot kind: {kind}')


# 2D: spatial density on a plane.


def plot_gridded_2d_mesh(
    ax: Axes,
    dist: GriddedDistribution,
    *args: Any,
    **kwargs: Any,
) -> QuadMesh:
    """Plot 2D cell-average density using cell boundaries."""
    xedges, yedges = _edges_2d(dist)
    return _mesh(
        ax, xedges, yedges, np.asarray(dist.density(), dtype=float), *args, **kwargs
    )


def plot_gridded_2d_contour(
    ax: Axes,
    dist: GriddedDistribution,
    *args: Any,
    **kwargs: Any,
) -> QuadContourSet:
    """Contour 2D cell-average density at bin centers."""
    xedges, yedges = _edges_2d(dist)
    return _contour(
        ax, xedges, yedges, np.asarray(dist.density(), dtype=float), *args, **kwargs
    )


def plot_gridded_2d_contourf(
    ax: Axes,
    dist: GriddedDistribution,
    *args: Any,
    **kwargs: Any,
) -> QuadContourSet:
    """Fill contour bands of 2D cell-average density at bin centers."""
    xedges, yedges = _edges_2d(dist)
    return _contourf(
        ax, xedges, yedges, np.asarray(dist.density(), dtype=float), *args, **kwargs
    )


@overload
def plot_gridded_2d(
    ax: Axes,
    dist: GriddedDistribution,
    *args: Any,
    kind: Literal['mesh'] = 'mesh',
    **kwargs: Any,
) -> QuadMesh: ...


@overload
def plot_gridded_2d(
    ax: Axes,
    dist: GriddedDistribution,
    *args: Any,
    kind: Literal['contour', 'contourf'],
    **kwargs: Any,
) -> QuadContourSet: ...


@overload
def plot_gridded_2d(
    ax: Axes,
    dist: GriddedDistribution,
    *args: Any,
    kind: Kind2D,
    **kwargs: Any,
) -> QuadMesh | QuadContourSet: ...


def plot_gridded_2d(
    ax: Axes,
    dist: GriddedDistribution,
    *args: Any,
    kind: Kind2D = 'mesh',
    **kwargs: Any,
) -> QuadMesh | QuadContourSet:
    """Dispatch to mesh, contour, or contourf for a 2D grid."""
    if kind == 'mesh':
        return plot_gridded_2d_mesh(ax, dist, *args, **kwargs)
    if kind == 'contour':
        return plot_gridded_2d_contour(ax, dist, *args, **kwargs)
    if kind == 'contourf':
        return plot_gridded_2d_contourf(ax, dist, *args, **kwargs)
    raise ValueError(f'Unknown plot kind: {kind}')


# Temporal 1D: time on horizontal axis, spatial coordinate on vertical axis.


def _temporal_1d_data(
    dist: TemporalGriddedDistribution,
    time_edges: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    steps = int(np.asarray(dist.lengths.values))
    xedges = (
        np.arange(steps + 1, dtype=float) - 0.5
        if time_edges is None
        else np.asarray(time_edges, dtype=float)
    )
    yedges = np.asarray(dist.dist.edges[0][0], dtype=float)
    density = np.asarray(dist.density(), dtype=float)[:steps].T
    return xedges, yedges, density


def plot_temporal_gridded_1d_mesh(
    ax: Axes,
    dist: TemporalGriddedDistribution,
    *args: Any,
    time_edges: ArrayLike | None = None,
    **kwargs: Any,
) -> QuadMesh:
    """Plot a 1D marginal density over discrete time as a heatmap."""
    xedges, yedges, density = _temporal_1d_data(dist, time_edges)
    return ax.pcolormesh(xedges, yedges, density, *args, **kwargs)


def plot_temporal_gridded_1d_contour(
    ax: Axes,
    dist: TemporalGriddedDistribution,
    *args: Any,
    time_edges: ArrayLike | None = None,
    **kwargs: Any,
) -> QuadContourSet:
    """Plot isodensity contours; these interpolate visually between times."""
    xedges, yedges, density = _temporal_1d_data(dist, time_edges)
    return ax.contour(_centers(xedges), _centers(yedges), density, *args, **kwargs)


def plot_temporal_gridded_1d_contourf(
    ax: Axes,
    dist: TemporalGriddedDistribution,
    *args: Any,
    time_edges: ArrayLike | None = None,
    **kwargs: Any,
) -> QuadContourSet:
    """Plot filled isodensity contours, interpolating between times."""
    xedges, yedges, density = _temporal_1d_data(dist, time_edges)
    return ax.contourf(_centers(xedges), _centers(yedges), density, *args, **kwargs)


@overload
def plot_temporal_gridded_1d(
    ax: Axes,
    dist: TemporalGriddedDistribution,
    *args: Any,
    kind: Literal['mesh'] = 'mesh',
    time_edges: ArrayLike | None = None,
    **kwargs: Any,
) -> QuadMesh: ...


@overload
def plot_temporal_gridded_1d(
    ax: Axes,
    dist: TemporalGriddedDistribution,
    *args: Any,
    kind: Literal['contour', 'contourf'],
    time_edges: ArrayLike | None = None,
    **kwargs: Any,
) -> QuadContourSet: ...


@overload
def plot_temporal_gridded_1d(
    ax: Axes,
    dist: TemporalGriddedDistribution,
    *args: Any,
    kind: Kind2D,
    time_edges: ArrayLike | None = None,
    **kwargs: Any,
) -> QuadMesh | QuadContourSet: ...


def plot_temporal_gridded_1d(
    ax: Axes,
    dist: TemporalGriddedDistribution,
    *args: Any,
    kind: Kind2D = 'mesh',
    time_edges: ArrayLike | None = None,
    **kwargs: Any,
) -> QuadMesh | QuadContourSet:
    """Dispatch a temporal 1D density plot."""
    if kind == 'mesh':
        return plot_temporal_gridded_1d_mesh(
            ax, dist, *args, time_edges=time_edges, **kwargs
        )
    if kind == 'contour':
        return plot_temporal_gridded_1d_contour(
            ax, dist, *args, time_edges=time_edges, **kwargs
        )
    if kind == 'contourf':
        return plot_temporal_gridded_1d_contourf(
            ax, dist, *args, time_edges=time_edges, **kwargs
        )
    raise ValueError(f'Unknown plot kind: {kind}')
