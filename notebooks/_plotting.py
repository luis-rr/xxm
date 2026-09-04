import itertools

import matplotlib.colors
import matplotlib.patches
import numpy as np
from matplotlib import pyplot as plt

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian

TRACE_COLORS = ('k', 'xkcd:magenta') + tuple(matplotlib.colormaps['Dark2'].colors)  # type: ignore


def state_cmap(num_states: int) -> matplotlib.colors.ListedColormap:
    """Return a discrete colormap with one color per state."""
    base = matplotlib.colormaps['tab10'].colors  # type: ignore
    colors = [base[index % len(base)] for index in range(num_states)]
    return matplotlib.colors.ListedColormap(colors)


def state_norm(num_states: int) -> matplotlib.colors.BoundaryNorm:
    """Return a normalization centered on integer state labels."""
    return matplotlib.colors.BoundaryNorm(
        np.arange(num_states + 1) - 0.5,
        num_states,
    )


def _num_states(states, num_states=None) -> int:
    if num_states is not None:
        return num_states

    states = np.asarray(states)
    return int(states.max()) + 1 if states.size else 1


def plot_state_1d(
    ax,
    states,
    *,
    num_states=None,
    aspect='auto',
    interpolation='none',
    alpha=0.25,
    **kwargs,
):
    """Plot a discrete state sequence as a one-row image."""
    states = np.asarray(states)
    num_states = _num_states(states, num_states)

    image = ax.imshow(
        states[None, :],
        aspect=aspect,
        interpolation=interpolation,
        cmap=state_cmap(num_states),
        norm=state_norm(num_states),
        transform=ax.get_xaxis_transform(),
        extent=(0, len(states), 0, 1),
        alpha=alpha,
        **kwargs,
    )

    ax.set(
        xlabel='time',
        yticks=[],
    )

    return image


def plot_traces_image(
    ax,
    values,
    *,
    xlabel='time',
    ylabel='variable',
    desc='',
    **kwargs,
):
    """Plot a multivariate sequence with variables on rows and time on columns."""
    values = np.asarray(values)

    if values.ndim != 2:
        raise ValueError(f'expected values with shape (T, N), got {values.shape}')

    image = ax.imshow(
        values.T,
        aspect='auto',
        interpolation='none',
        **kwargs,
    )

    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
    )

    cbar = ax.figure.colorbar(image, ax=ax)
    cbar.set_label(desc)

    return image


def plot_traces_image_comparison(
    values0,
    values1,
    *,
    xlabel='time',
    ylabel='variable',
    name0='observations',
    name1='reconstruction',
    desc='',
    **kwargs,
):
    """Backward-compatible multivariate sequence comparison helper."""
    _, axs = plt.subplots(
        nrows=2,
        constrained_layout=True,
        figsize=(8, 6),
    )

    plot_traces_image(
        axs[0],
        values0,
        xlabel=xlabel,
        ylabel=ylabel,
        desc=desc,
        **kwargs,
    )
    axs[0].set(title=name0)

    plot_traces_image(
        axs[1],
        values1,
        xlabel=xlabel,
        ylabel=ylabel,
        desc=desc,
        **kwargs,
    )
    axs[1].set(title=name1)


def _plot_traces_1d(ax, traces, **kwargs):

    for trace, color in zip(
        traces.T,
        itertools.cycle(TRACE_COLORS),
    ):
        ax.plot(trace, color=color, **kwargs)


def _plot_traces_2d(ax, traces, **kwargs):
    traces = np.asarray(traces)

    if traces.ndim == 2:
        traces = traces[None, :, :]

    assert traces.ndim == 3 and traces.shape[2] == 2, (
        f'expected traces with shape (T, 2), got {traces.shape}'
    )
    for trace, color in zip(
        traces,
        itertools.cycle(TRACE_COLORS),
    ):
        ax.plot(
            trace[:, 0],
            trace[:, 1],
            color=kwargs.pop('color', color),
            **kwargs,
        )


def plot_traces_2d(ax, traces, **kwargs):
    """Uncolored 2D latent trajectory plot."""
    traces = np.asarray(traces)

    _plot_traces_2d(ax, traces, **kwargs)
    ax.scatter(traces[0, 0], traces[0, 1], color='g', zorder=10)
    ax.scatter(traces[-1, 0], traces[-1, 1], color='r', zorder=10)
    ax.set(title='latent states', aspect='equal')


def plot_seq_1d(
    ax,
    states,
    traces,
    desc='observations',
) -> None:
    """Plot all continuous dimensions over time with a discrete state overlay."""
    traces = np.asarray(traces)

    plot_state_1d(ax, states)
    _plot_traces_1d(ax, traces)

    ax.set(
        xlabel='time',
        ylabel=f'{desc}',
    )


def plot_seq_1d_comparison(
    states0,
    traces0,
    states1,
    traces1,
):
    """Backward-compatible time-series comparison helper."""
    _, axs = plt.subplots(
        nrows=2,
        constrained_layout=True,
        figsize=(8, 6),
    )

    plot_seq_1d(axs[0], states0, traces0)
    axs[0].set(title='True states')

    plot_seq_1d(axs[1], states1, traces1)
    axs[1].set(title='Inferred states')


def plot_seq_2d(
    ax,
    states,
    traces,
    desc='observations',
    num_states=None,
    linewidth=0.75,
    alpha=0.35,
    size=12,
    mark_endpoints=True,
    **kwargs,
) -> None:
    """Plot all continuous dimensions coloring points by discrete state."""

    values = np.asarray(traces)

    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f'expected values with shape (T, 2), got {values.shape}')

    ax.plot(
        values[:, 0],
        values[:, 1],
        linewidth=linewidth,
        alpha=alpha,
    )

    if states is not None:
        states = np.asarray(states)
        num_states = _num_states(states, num_states)

        ax.scatter(
            values[:, 0],
            values[:, 1],
            c=states,
            s=size,
            cmap=state_cmap(num_states),
            norm=state_norm(num_states),
            **kwargs,
        )

    if mark_endpoints:
        ax.scatter(values[0, 0], values[0, 1], marker='o', zorder=10)
        ax.scatter(values[-1, 0], values[-1, 1], marker='x', zorder=10)

    ax.set(
        xlabel=r'$x_1$',
        ylabel=r'$x_2$',
        aspect='equal',
    )

    ax.set_xlabel(f'{desc} 1')
    ax.set_ylabel(f'{desc} 2')


def plot_seq_2d_comparison(
    states0,
    traces0,
    states1,
    traces1,
):
    """Backward-compatible 2D sequence comparison helper."""
    _, axs = plt.subplots(
        ncols=2,
        sharex='all',
        sharey='all',
        figsize=(8, 4),
    )

    plot_seq_2d(
        axs[0],
        states=states0,
        traces=traces0,
    )
    plot_seq_2d(
        axs[1],
        states=states1,
        traces=traces1,
    )


def plot_dyn_2d(
    ax,
    mean,
    xlim=(-3, 3),
    ylim=(-3, 3),
    num_points=15,
    **kwargs,
):
    x = np.linspace(*xlim, num_points)  # type: ignore
    y = np.linspace(*ylim, num_points)  # type: ignore
    xx, yy = np.meshgrid(x, y)

    points = np.stack([xx, yy], axis=-1)

    next_points = np.asarray(mean(points))
    displacement = next_points - points

    ax.quiver(
        xx,
        yy,
        displacement[..., 0],
        displacement[..., 1],
        **kwargs,
    )

    ax.set(
        xlim=xlim,
        ylim=ylim,
        aspect='equal',
        xlabel=r'$x_1$',
        ylabel=r'$x_2$',
    )


def plot_dyn_linear(
    ax,
    affine: Affine,
    **kwargs,
):
    if affine.input_ndim == 2 and affine.input_shape[0] == 1:
        affine = affine.input_squeeze()

    if affine.input_shape != (2,) or affine.output_dim != 2:
        raise ValueError(
            '2D dynamics require an affine map (2,) -> (2,), '
            f'got {affine.input_shape} -> ({affine.output_dim},)'
        )

    plot_dyn_2d(
        ax,
        affine.apply,
        **kwargs,
    )


def plot_dyn_linear_gaussian(
    ax,
    dynamics: LinearGaussian,
    **kwargs,
):
    """Plot the conditional-mean displacement field of a 2D LinearGaussian."""
    if dynamics.batch_shape:
        raise ValueError(
            'expected an unbatched LinearGaussian; '
            'select one batch element before plotting'
        )

    plot_dyn_linear(
        ax,
        dynamics.affine,
        **kwargs,
    )


def plot_dyn_linear_gaussian_comparison(
    dynamics0: LinearGaussian,
    dynamics1: LinearGaussian,
    color0='k',
    color1='xkcd:magenta',
    ax=None,
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    plot_dyn_linear_gaussian(
        ax,
        dynamics0,
        color=color0,
        **kwargs,
    )

    plot_dyn_linear_gaussian(
        ax,
        dynamics1,
        color=color1,
        **kwargs,
    )


def plot_fit_progress(objective, name='Log Likelihood', ax=None) -> None:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    ax.plot(np.asarray(objective))

    ax.set(
        xlabel='iteration',
        ylabel=name,
    )


def plot_fit_progress_many(fits, highlight_idx=None, name='Log Likelihood', ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(np.asarray(fits.objective_traces).T)

    if highlight_idx is not None:
        _, objective = fits.get(highlight_idx)
        ax.plot(np.asarray(objective), color='k')

    ax.set(
        xlabel='iteration',
        ylabel=name,
    )


def plot_dyn_conditional_linear(
    dyn: LinearGaussian,
):
    """Plot all state-conditioned 2D linear-Gaussian dynamics."""
    _, axs = plt.subplots(
        figsize=(8, 6),
        ncols=dyn.batch_shape[0],
        sharex='all',
        sharey='all',
        constrained_layout=True,
    )

    axs = np.atleast_1d(axs)

    for state, ax in enumerate(axs):
        plot_dyn_linear_gaussian(
            ax,
            dyn.select(state),
        )
        ax.set_title(f'State {state}')


def plot_dyn_conditional_linear_comparison(
    d0: LinearGaussian,
    d1: LinearGaussian,
) -> None:
    """Plot two state-conditioned 2D linear-Gaussian dynamics fields."""
    if d0.batch_shape != d1.batch_shape:
        raise ValueError('Models must have the same batch shape.')

    _, axs = plt.subplots(
        figsize=(8, 6),
        ncols=d0.batch_shape[0],
        sharex='all',
        sharey='all',
        constrained_layout=True,
    )

    axs = np.atleast_1d(axs)

    for state, ax in enumerate(axs):
        plot_dyn_linear_gaussian(
            ax,
            d0.select(state),
            color='k',
        )
        plot_dyn_linear_gaussian(
            ax,
            d1.select(state),
            color='xkcd:magenta',
        )
        ax.set_title(f'State {state}')


def plot_gaussian_2d_ellipse(
    ax,
    gaussian: Gaussian,
    *,
    linestyle: str = '-',
    label: str | None = None,
    color='k',
    **kwargs,
) -> None:
    """Plot the one-standard-deviation ellipse of an unbatched 2D Gaussian."""
    if gaussian.variable_dim != 2 or gaussian.batch_shape:
        raise ValueError('expected an unbatched 2D Gaussian')

    eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(gaussian.covariance))

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(
        np.arctan2(
            eigenvectors[1, 0],
            eigenvectors[0, 0],
        )
    )

    ellipse = matplotlib.patches.Ellipse(
        tuple(np.asarray(gaussian.mean)),
        width=2 * np.sqrt(eigenvalues[0]),
        height=2 * np.sqrt(eigenvalues[1]),
        angle=angle,
        fill=False,
        linestyle=linestyle,
        linewidth=2,
        label=label,
        color=color,
        **kwargs,
    )

    ax.add_patch(ellipse)
    ax.scatter(*np.asarray(gaussian.mean), marker='o', s=40, color=color)
