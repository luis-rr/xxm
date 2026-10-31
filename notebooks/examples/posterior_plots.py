"""Gridded posterior plots for temporal Gaussian marginals.

Shared by the notebooks. The posterior density is white to magenta, the
posterior mean is a magenta line and the truth a black one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from xxm import plot
from xxm.core.dists.gaussian import TemporalGaussian
from xxm.core.dists.gridded import GriddedDistribution, TemporalGriddedDistribution

DENSITY_CMAP = LinearSegmentedColormap.from_list(
    'posterior_density',
    ['white', plot.INFERRED_COLOR],
)


def grid_edges(
    posterior: TemporalGaussian,
    variable: int,
    *,
    cells_per_std: float,
    truth: jax.Array | None = None,
    interval: float = 4.0,
    max_cells: int = 400,
) -> jax.Array:
    """Grid edges covering one variable's posterior bulk and its (T,) truth.

    Cells are sized from the median posterior standard deviation, so the
    resolution follows the uncertainty rather than the range of the values.
    """
    mean = np.asarray(posterior.mean.get(squeeze=False))[:, variable]
    std = np.sqrt(np.asarray(posterior.variance.get(squeeze=False))[:, variable])

    bounds = [mean - interval * std, mean + interval * std]
    if truth is not None:
        bounds.append(np.asarray(truth))

    low, high = np.min(bounds), np.max(bounds)
    width = np.median(std) / cells_per_std
    num_cells = int(np.clip(np.ceil((high - low) / width), 1, max_cells))
    return jnp.linspace(low, high, num_cells + 1)


def plot_density_over_time(
    ax,
    posterior: TemporalGaussian,
    variable: int,
    *,
    truth: jax.Array | None = None,
    cells_per_std: float = 6.0,
):
    """Plot one variable's gridded posterior density over time.

    `posterior` is unbatched; `truth` has shape (T,).
    """
    edges = grid_edges(
        posterior,
        variable,
        cells_per_std=cells_per_std,
        truth=truth,
    )
    grid = TemporalGriddedDistribution.from_gaussian(
        posterior,
        (edges,),
        variables=(variable,),
    )
    mesh = plot.plot_temporal_gridded_1d(
        ax,
        grid,
        cmap=DENSITY_CMAP,
        rasterized=True,
    )

    if truth is not None:
        ax.plot(truth, color=plot.TRUE_COLOR, linewidth=0.5, label='true')

    ax.plot(
        posterior.mean.get(squeeze=False)[:, variable],
        color=plot.INFERRED_COLOR,
        linewidth=0.75,
        label='posterior mean',
    )

    return mesh


def plot_posterior_over_time(
    posterior: TemporalGaussian,
    truth: jax.Array | None = None,
    *,
    symbol: str = 'x',
    titles: list[str] | None = None,
    cells_per_std: float = 6.0,
    figsize=None,
):
    """Plot gridded posterior densities over time, one row per variable.

    `posterior` has batch () or (K,), giving one column per batch entry.
    `truth` has the matching shape (T, D) or (K, T, D). Rows are labelled
    with `symbol` and the variable index.
    """
    if posterior.batch_shape:
        posteriors = [posterior.get(k) for k in range(posterior.batch_shape[0])]
    else:
        posteriors = [posterior]
    columns = len(posteriors)

    truths = None
    if truth is not None:
        truths = jnp.reshape(truth, (columns, *jnp.shape(truth)[-2:]))
    rows = posterior.variable_dim

    _, axs = plt.subplots(
        nrows=rows,
        ncols=columns,
        figsize=figsize or (4 * columns, 1.25 * rows),
        sharex='all',
        squeeze=False,
        constrained_layout=True,
    )

    for (d, k), ax in np.ndenumerate(axs):
        mesh = plot_density_over_time(
            ax,
            posteriors[k],
            d,
            truth=None if truths is None else truths[k, :, d],
            cells_per_std=cells_per_std,
        )
        ax.figure.colorbar(mesh, ax=ax, label='density')
        ax.set(ylabel=rf'${symbol}_{d + 1}$')

    if titles is not None:
        for title, ax in zip(titles, axs[0], strict=True):
            ax.set(title=title)

    axs[0, 0].legend(fontsize='small')
    for ax in axs[-1]:
        ax.set(xlabel='time')

    return axs


def plot_occupancy_density(
    ax,
    posterior: TemporalGaussian,
    *,
    truth: jax.Array | None = None,
    cells_per_std: float = 5,
    **kwargs,
):
    """Draw the time-averaged posterior density of the first two variables.

    `posterior` is unbatched; `truth` has shape (T, 2) and only widens the
    grid to cover it.
    """
    edges = tuple(
        grid_edges(
            posterior,
            d,
            cells_per_std=cells_per_std,
            truth=None if truth is None else truth[:, d],
        )
        for d in range(2)
    )
    grid = GriddedDistribution.from_gaussian_occupancy(
        posterior,
        edges,
        variables=(0, 1),
    )
    kwargs = {'cmap': DENSITY_CMAP, 'rasterized': True} | kwargs
    return plot.plot_gridded_2d(ax, grid, **kwargs)


def plot_occupancy(
    posterior: TemporalGaussian,
    truth: jax.Array | None = None,
    *,
    symbol: str = 'x',
    cells_per_std: float = 5,
    ax=None,
    figsize=(2.5, 2.5),
):
    """Plot the time-averaged density with posterior mean and truth.

    `posterior` is unbatched; `truth` has shape (T, 2). A new figure is
    created unless `ax` is given.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    plot_occupancy_density(
        ax,
        posterior,
        truth=truth,
        cells_per_std=cells_per_std,
    )

    if truth is not None:
        ax.plot(
            truth[:, 0],
            truth[:, 1],
            color=plot.TRUE_COLOR,
            linewidth=0.5,
            alpha=0.5,
            label='true',
        )

    mean = posterior.mean.get(squeeze=False)
    ax.plot(
        mean[:, 0],
        mean[:, 1],
        color=plot.INFERRED_COLOR,
        linewidth=0.75,
        label='posterior mean',
    )
    ax.set(xlabel=rf'${symbol}_1$', ylabel=rf'${symbol}_2$')
    plot.square_axes(ax)

    return ax
