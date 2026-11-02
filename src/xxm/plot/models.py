"""Figure constructors composed from :mod:`xxm.plot.core` primitives."""

from copy import copy
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import LinearGaussian
from xxm.plot import core as plot


def plot_traces_image_comparison(
    values0,
    values1,
    *,
    xlabel='time',
    ylabel='variable',
    name0='observations',
    name1='reconstruction',
    desc='',
    figsize=(8, 6),
    **kwargs,
):
    """Compare sequences with one shared value scale and colorbar."""
    values = [plot._traces(values0), plot._traces(values1)]
    options = (
        dict[str, Any](
            vmin=min(float(v.min()) for v in values),
            vmax=max(float(v.max()) for v in values),
        )
        | kwargs
    )
    if kwargs.get('norm') is not None:
        options.pop('vmin', None)
        options.pop('vmax', None)
        # Resolve an unset normalization against both panels before drawing.
        if isinstance(options['norm'], Normalize):
            options['norm'] = copy(options['norm'])
            options['norm'].autoscale_None(np.concatenate([v.ravel() for v in values]))
    fig, axs = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for ax, value, name in zip(axs[:, 0], values, (name0, name1)):
        plot.plot_traces_image(ax, value, xlabel=xlabel, ylabel=ylabel, **options)
        ax.set_title(name)
    fig.colorbar(axs.flat[0].images[0], ax=list(axs[:, 0]), label=desc)
    return axs[:, 0]


def plot_seq_1d_comparison(
    states0,
    traces0,
    states1,
    traces1,
    *,
    name0='True states',
    name1='Inferred states',
    figsize=(8, 6),
    **kwargs,
):
    _, axs = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    count = max(plot._num_states(states0), plot._num_states(states1))
    options = dict[str, Any](kwargs)
    options['state_kwargs'] = dict[str, Any](num_states=count) | (
        options.get('state_kwargs') or {}
    )
    for ax, states, traces, name in zip(
        axs[:, 0], (states0, states1), (traces0, traces1), (name0, name1)
    ):
        plot.plot_seq_1d(ax, states, traces, **options)
        ax.set_title(name)
    return axs[:, 0]


def plot_seq_2d_comparison(
    states0,
    traces0,
    states1,
    traces1,
    *,
    name0='True states',
    name1='Inferred states',
    figsize=(8, 4),
    **kwargs,
):
    _, axs = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    count = max(
        (plot._num_states(s) for s in (states0, states1) if s is not None), default=1
    )
    for ax, states, traces, name in zip(
        axs[0], (states0, states1), (traces0, traces1), (name0, name1)
    ):
        plot.plot_seq_2d(
            ax, states, traces, **(dict[str, Any](num_states=count) | kwargs)
        )
        ax.set_title(name)
    return axs[0]


def plot_fit_progress(
    objective,
    *,
    name='Log Likelihood',
    title='',
    figsize=(6, 6),
    linthresh=1e-3,
    **kwargs,
):
    """Plot an objective and its iteration-to-iteration changes."""
    values = np.asarray(objective)
    if values.ndim != 1 or not values.size:
        raise ValueError(
            f'expected nonempty objective with shape (T,), got {values.shape}'
        )
    objectives = values[None]

    values = np.asarray(objectives)

    _, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, squeeze=False)

    for ax, traces in zip(axs[:, 0], (values.T, np.diff(values, axis=1).T)):
        if traces.shape[0]:
            plot.plot_traces_1d(
                ax, traces, **(dict[str, Any](linewidth=0.7, alpha=0.6) | kwargs)
            )

        ax.set_xlabel('iteration')
    axs[0, 0].set(ylabel=name, title=title)
    axs[1, 0].set_ylabel('change in ' + name)
    axs[1, 0].set_yscale('symlog', linthresh=linthresh)
    axs[1, 0].axhline(0, color='k', linewidth=0.5)
    return axs[:, 0]


def plot_fit_progress_many(
    objectives,
    *,
    name='Log Likelihood',
    title='',
    figsize=(6, 6),
    linthresh=1e-3,
    **kwargs,
):
    """Plot many objective traces (shape (N, T)) and their changes."""
    values = np.asarray(objectives)
    if values.ndim != 2 or not values.size:
        raise ValueError(
            f'expected nonempty objectives with shape (N, T), got {values.shape}'
        )

    _, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, squeeze=False)

    for ax, traces in zip(axs[:, 0], (values.T, np.diff(values, axis=1).T)):
        if traces.shape[0]:
            plot.plot_traces_1d(
                ax, traces, **(dict[str, Any](linewidth=0.7, alpha=0.6) | kwargs)
            )

        ax.set_xlabel('iteration')
    axs[0, 0].set(ylabel=name, title=title)
    axs[1, 0].set_ylabel('change in ' + name)
    axs[1, 0].set_yscale('symlog', linthresh=linthresh)
    axs[1, 0].axhline(0, color='k', linewidth=0.5)
    return axs[:, 0]


def plot_dyn_linear_gaussian_comparison(
    ax,
    dynamics0: LinearGaussian,
    dynamics1: LinearGaussian,
    *,
    color0=plot.TRUE_COLOR,
    color1=plot.INFERRED_COLOR,
    name0='true',
    name1='inferred',
    kwargs0=None,
    kwargs1=None,
    **kwargs,
):
    """Overlay two conditional means on the caller's axis."""
    plot.plot_dyn_linear_gaussian(
        ax,
        dynamics0,
        **(dict[str, Any](color=color0, label=name0) | kwargs | (kwargs0 or {})),
    )
    plot.plot_dyn_linear_gaussian(
        ax,
        dynamics1,
        **(dict[str, Any](color=color1, label=name1) | kwargs | (kwargs1 or {})),
    )


def plot_dyn_conditional_linear(dyn: LinearGaussian, **kwargs):
    if len(dyn.batch_shape) != 1 or not dyn.batch_shape[0]:
        raise ValueError(f'expected dynamics batch (K,), got {dyn.batch_shape}')
    return plot_dyn_grid(
        dyn.affine.broadcast((1,), axis=0),
        col_names=[f'State {i}' for i in range(dyn.batch_shape[0])],
        **kwargs,
    )[0]


def plot_dyn_conditional_linear_comparison(
    d0: LinearGaussian, d1: LinearGaussian, **kwargs
):
    if (
        len(d0.batch_shape) != 1
        or d0.batch_shape != d1.batch_shape
        or not d0.batch_shape[0]
    ):
        raise ValueError(
            f'expected matching dynamics batches (K,), got {d0.batch_shape} and {d1.batch_shape}'
        )
    _, axs = plt.subplots(
        1,
        d0.batch_shape[0],
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for state, ax in enumerate(axs[0]):
        plot_dyn_linear_gaussian_comparison(
            ax, d0.select(state), d1.select(state), **kwargs
        )
        ax.set_title(f'State {state}')
    return axs[0]


def _grid(batch_shape, row_names, col_names, figsize, suptitle):
    if len(batch_shape) != 2 or not all(batch_shape):
        raise ValueError(
            f'expected nonempty batch shape (rows, columns), got {batch_shape}'
        )
    rows, cols = batch_shape
    for names, count, label in (
        (row_names, rows, 'row_names'),
        (col_names, cols, 'col_names'),
    ):
        if names is not None and len(names) != count:
            raise ValueError(f'expected {count} {label}, got {len(names)}')
    fig, axs = plt.subplots(
        rows,
        cols,
        squeeze=False,
        sharex=True,
        sharey=True,
        figsize=figsize or (1 + 2 * cols, 1 + 2 * rows),
        constrained_layout=True,
    )
    fig.suptitle(suptitle)
    return fig, axs


def _label_grid(axs, row_names, col_names):
    for (i, j), ax in np.ndenumerate(axs):
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        if i == 0 and col_names is not None:
            ax.set_title(col_names[j])
        if j == 0 and row_names is not None:
            ax.set_ylabel(f'{row_names[i]}\n{ax.get_ylabel()}')


def plot_dyn_grid(
    affine_batch: Affine,
    *,
    xlim=(-4, 4),
    ylim=(-4, 4),
    row_names=None,
    col_names=None,
    suptitle='',
    figsize=None,
    **kwargs,
):
    """Draw an Affine batch (rows, columns); return a 2D axes array."""
    _, axs = _grid(affine_batch.batch_shape, row_names, col_names, figsize, suptitle)
    for index, ax in np.ndenumerate(axs):
        plot.plot_dyn_linear(
            ax, affine_batch.select(index), xlim=xlim, ylim=ylim, **kwargs
        )
    _label_grid(axs, row_names, col_names)
    return axs


def plot_dyn_mismatch_grid(
    affine0_batch: Affine,
    affine1_batch: Affine,
    *,
    xlim=(-4, 4),
    ylim=(-4, 4),
    row_names=None,
    col_names=None,
    suptitle='',
    figsize=None,
    relative=False,
    relative_eps=1e-3,
    num_points=100,
    vmin=0.0,
    vmax=None,
    mesh_kwargs=None,
    **kwargs,
):
    """Compare matching (rows, columns) batches with one normalization and colorbar."""
    shape = affine0_batch.batch_shape
    if len(shape) != 2 or not all(shape) or shape != affine1_batch.batch_shape:
        raise ValueError(
            f'expected matching nonempty 2D batches, got {shape} and {affine1_batch.batch_shape}'
        )
    options = dict[str, Any](mesh_kwargs or {})
    if options.get('norm') is None:
        options.pop('norm', None)
        if vmax is None:
            vmax = max(
                float(
                    plot._dyn_mismatch(
                        affine0_batch.select(index),
                        affine1_batch.select(index),
                        xlim=xlim,
                        ylim=ylim,
                        num_points=num_points,
                        relative=relative,
                        relative_eps=relative_eps,
                    )[-1].max()
                )
                for index in np.ndindex(shape)
            )
        options['norm'] = Normalize(
            options.pop('vmin', vmin), options.pop('vmax', vmax)
        )
    elif isinstance(options['norm'], Normalize):
        options['norm'] = copy(options['norm'])
        magnitudes = [
            plot._dyn_mismatch(
                affine0_batch.select(index),
                affine1_batch.select(index),
                xlim=xlim,
                ylim=ylim,
                num_points=num_points,
                relative=relative,
                relative_eps=relative_eps,
            )[-1].ravel()
            for index in np.ndindex(shape)
        ]
        options['norm'].autoscale_None(np.concatenate(magnitudes))
    if 'vmin' in options or 'vmax' in options:
        raise ValueError('pass either norm or mesh vmin/vmax, not both')
    fig, axs = _grid(shape, row_names, col_names, figsize, suptitle)
    for index, ax in np.ndenumerate(axs):
        plot.plot_dyn_mismatch(
            ax,
            affine0_batch.select(index),
            affine1_batch.select(index),
            xlim=xlim,
            ylim=ylim,
            relative=relative,
            relative_eps=relative_eps,
            num_points=num_points,
            mesh_kwargs=options,
            **kwargs,
        )
    _label_grid(axs, row_names, col_names)
    fig.colorbar(
        axs.flat[0].collections[0],
        ax=list(axs.flat),
        label='relative displacement mismatch' if relative else 'displacement mismatch',
    )
    return axs


def plot_latent_recovery(
    true_latents,
    inferred_latents,
    *,
    figsize=(8, 4),
    name_left='True',
    name_right='Inferred',
):
    return plot_seq_2d_comparison(
        None,
        true_latents,
        None,
        inferred_latents,
        figsize=figsize,
        name0=name_left,
        name1=name_right,
        desc='latent',
    )


def plot_latent_obs(true_states, true_latents, observations):
    """Compose latent traces, observations and a trajectory in one figure."""
    fig, axs = plt.subplots(
        3, 1, figsize=(8, 7), squeeze=False, constrained_layout=True
    )
    plot.plot_seq_1d(axs[0, 0], true_states, true_latents, desc='latent')
    image = plot.plot_traces_image(axs[1, 0], observations, ylabel='observations')
    fig.colorbar(image, ax=axs[1, 0])
    plot.plot_seq_2d(axs[2, 0], true_states, true_latents, desc='latent')
    axs[1, 0].sharex(axs[0, 0])
    return axs[:, 0]
