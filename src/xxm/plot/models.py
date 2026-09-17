"""Figure constructors composed from :mod:`xxm.plot.core` primitives."""

from copy import copy
from typing import Any, Literal

import jax.numpy as jnp
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


def plot_fit_progress(objective, *, name='Log Likelihood', title='', **kwargs):
    """Plot an objective and its iteration-to-iteration changes."""
    values = np.asarray(objective)
    if values.ndim != 1 or not values.size:
        raise ValueError(
            f'expected nonempty objective with shape (T,), got {values.shape}'
        )
    return plot_fit_progress_many(values[None], name=name, title=title, **kwargs)


def plot_fit_progress_many(
    objectives,
    highlight_idx=None,
    *,
    name='Log Likelihood',
    title='',
    figsize=(6, 6),
    **kwargs,
):
    """Plot objective arrays with shape (runs, iterations), optionally highlighting one."""
    values = np.asarray(objectives)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError(
            f'expected objectives with shape (runs, iterations), got {values.shape}'
        )
    if highlight_idx is not None and not 0 <= highlight_idx < len(values):
        raise ValueError(f'highlight_idx must be in [0, {len(values)})')
    _, axs = plt.subplots(2, 1, figsize=figsize, sharex=True, squeeze=False)
    for ax, traces in zip(axs[:, 0], (values.T, np.diff(values, axis=1).T)):
        if traces.shape[0]:
            plot.plot_traces_1d(
                ax, traces, **(dict[str, Any](linewidth=0.7, alpha=0.6) | kwargs)
            )
            if highlight_idx is not None:
                plot.plot_traces_1d(
                    ax, traces[:, highlight_idx, None], color=plot.TRUE_COLOR
                )
        ax.set_xlabel('iteration')
    axs[0, 0].set(ylabel=name, title=title)
    axs[1, 0].set_ylabel('change in ' + name)
    axs[1, 0].set_yscale('symlog', linthresh=1e-3)
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


def _collect_ltr_dynamics(model):
    """Assemble baseline and identity-plus-motif maps, with state on rows."""
    baseline = model.baseline_dynamics.affine
    motifs = jnp.asarray(model.motifs)
    if not baseline.batch_shape:
        coefficients = jnp.concatenate(
            [baseline.coefficients[None], motifs + jnp.eye(model.latent_dim)], axis=0
        )[None]
        bias = jnp.concatenate(
            [baseline.bias[None], jnp.zeros(motifs.shape[:-1])], axis=0
        )[None]
    else:
        coefficients = jnp.concatenate(
            [baseline.coefficients[:, None], motifs + jnp.eye(model.latent_dim)], axis=1
        )
        bias = jnp.concatenate(
            [baseline.bias[:, None], jnp.zeros(motifs.shape[:-1])], axis=1
        )
    return Affine(coefficients=coefficients, bias=bias)


def plot_dyn_model(model, *, title_prefix=None, **kwargs):
    prefix = '' if title_prefix is None else f'{title_prefix}: '
    affines = _collect_ltr_dynamics(model)
    return plot_dyn_grid(
        affines,
        col_names=[prefix + 'Baseline']
        + [f'{prefix}Motif {p + 1}' for p in range(model.num_motifs)],
        row_names=[f'State {k}' for k in range(affines.batch_shape[0])]
        if model.baseline_dynamics.batch_shape
        else None,
        **kwargs,
    )


def plot_dyn_model_mismatch(model0, model1, *, title_prefix=None, **kwargs):
    prefix = '' if title_prefix is None else f'{title_prefix}: '
    affines = _collect_ltr_dynamics(model0)
    return plot_dyn_mismatch_grid(
        affines,
        _collect_ltr_dynamics(model1),
        col_names=[prefix + 'Baseline']
        + [f'{prefix}Motif {p + 1}' for p in range(model0.num_motifs)],
        row_names=[f'State {k}' for k in range(affines.batch_shape[0])]
        if model0.baseline_dynamics.batch_shape
        else None,
        **kwargs,
    )


def plot_dyn_temporal(
    dyns: Affine, representative_indices, *, suptitle='Temporal dynamics', **kwargs
):
    return plot_dyn_grid(
        dyns,
        col_names=[f'Time {t}' for t in representative_indices],
        row_names=[f'State {s}' for s in range(dyns.batch_shape[0])],
        suptitle=suptitle,
        **kwargs,
    )


def plot_dyn_temporal_mismatch(
    dyns0: Affine,
    dyns1: Affine,
    representative_indices,
    *,
    suptitle='Temporal dynamics mismatch',
    **kwargs,
):
    return plot_dyn_mismatch_grid(
        dyns0,
        dyns1,
        col_names=[f'Time {t}' for t in representative_indices],
        row_names=[f'State {s}' for s in range(dyns0.batch_shape[0])],
        suptitle=suptitle,
        **kwargs,
    )


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


def plot_weight_recovery(
    ax,
    true_weights,
    inferred_mean,
    inferred_std,
    *,
    interval=2.0,
    xlabel='transition',
    ylabel='temporal weight',
    title='',
    legend=True,
    true_kwargs=None,
    inferred_kwargs=None,
    band_kwargs=None,
):
    """Compare (T,) or (T, P) weights and mean +/- interval * standard deviation."""
    values, mean, std = [
        np.asarray(v) for v in (true_weights, inferred_mean, inferred_std)
    ]
    if (
        values.shape != mean.shape
        or mean.shape != std.shape
        or values.ndim not in (1, 2)
        or not all(values.shape)
    ):
        raise ValueError(
            f'expected matching (T,) or (T, P) arrays, got {values.shape}, {mean.shape}, {std.shape}'
        )
    if np.any(std < 0) or interval < 0:
        raise ValueError('expected nonnegative standard deviation and interval')
    if values.ndim == 1:
        values, mean, std = values[:, None], mean[:, None], std[:, None]
    plot.plot_traces_1d(
        ax,
        values,
        **(dict[str, Any](color=plot.TRUE_COLOR, label='true') | (true_kwargs or {})),
    )
    plot.plot_traces_1d(
        ax,
        mean,
        **(
            dict[str, Any](color=plot.INFERRED_COLOR, label='inferred')
            | (inferred_kwargs or {})
        ),
    )
    for m, s in zip(mean.T, std.T):
        ax.fill_between(
            np.arange(len(m)),
            m - interval * s,
            m + interval * s,
            **(
                dict[str, Any](
                    facecolor=plot.INFERRED_COLOR, edgecolor='none', alpha=0.15
                )
                | (band_kwargs or {})
            ),
        )
    ax.axhline(0, color=plot.TRUE_COLOR, linewidth=0.5)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict[str, Any](zip(labels, handles))
        ax.legend(unique.values(), unique.keys())


def plot_weight_recovery_grid(
    true_weights,
    inferred_mean,
    inferred_std,
    *,
    interval=2.0,
    sharex: bool | Literal['all', 'none', 'row', 'col'] = 'all',
    sharey: bool | Literal['all', 'none', 'row', 'col'] = 'all',
    figsize=None,
    **kwargs,
):
    values, mean, std = [
        np.asarray(v) for v in (true_weights, inferred_mean, inferred_std)
    ]
    if (
        values.ndim != 3
        or not all(values.shape)
        or values.shape != mean.shape
        or mean.shape != std.shape
    ):
        raise ValueError(
            f'expected matching weights (K, T, P), got {values.shape}, {mean.shape}, {std.shape}'
        )
    states, _, motifs = values.shape
    _, axs = plt.subplots(
        states,
        motifs,
        squeeze=False,
        figsize=figsize or (2.5 * motifs, states),
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True,
    )
    for (k, p), ax in np.ndenumerate(axs):
        plot_weight_recovery(
            ax,
            values[k, :, p],
            mean[k, :, p],
            std[k, :, p],
            **(
                dict[str, Any](
                    interval=interval,
                    title=f'State {k}, motif {p + 1}',
                    legend=k == p == 0,
                )
                | kwargs
            ),
        )
    return axs


def plot_drift_weights(weights, indices=None, *, figsize=(4, 1.5)):
    _, axs = plt.subplots(1, 1, figsize=figsize, squeeze=False)
    ax = axs[0, 0]
    lines = plot.plot_traces_1d(ax, weights)
    for p, line in enumerate(lines):
        line.set_label(rf'$\lambda_{p + 1}$')
    ax.axhline(0, color=plot.TRUE_COLOR, linewidth=0.75, linestyle='--')
    if indices is not None:
        for index in np.asarray(indices):
            ax.axvline(index, color=plot.TRUE_COLOR, linewidth=0.5, alpha=0.25)
    ax.set(xlabel='time', ylabel=r'$\lambda$')
    ax.legend()
    return axs[0]


def plot_weight_path(
    weights, indices=None, *, figsize=(1.5, 1.5), show_labels=False, show_cbar=False
):
    values = plot._traces(weights, 2)
    fig, axs = plt.subplots(
        1, 1, figsize=figsize, constrained_layout=True, squeeze=False
    )
    ax = axs[0, 0]
    plot.plot_traces_2d(ax, values, mark_endpoints=False, linewidth=0.5, alpha=0.3)
    points = ax.scatter(values[:, 0], values[:, 1], c=np.arange(len(values)), s=8)
    if indices is not None:
        if show_labels:
            plot.plot_time_markers_2d(ax, values, indices)
        else:
            selected = values[np.asarray(indices)]
            ax.scatter(
                selected[:, 0], selected[:, 1], s=50, facecolor='none', edgecolor='k'
            )
    ax.set(xlabel=r'$\lambda_1$', ylabel=r'$\lambda_2$')
    plot.square_axes(ax)
    if show_cbar:
        fig.colorbar(points, ax=ax, label='transition')
    return axs[0]


def plot_weight_path_recovery(
    true_weights, inferred_weights, *, title='Temporal path recovery', figsize=(5, 5)
):
    _, axs = plt.subplots(1, 1, figsize=figsize, constrained_layout=True, squeeze=False)
    ax = axs[0, 0]
    plot.plot_trace_2d(
        ax, true_weights, color=plot.TRUE_COLOR, line_kwargs={'label': 'true'}
    )
    plot.plot_trace_2d(ax, inferred_weights, line_kwargs={'label': 'inferred'})
    ax.set(xlabel=r'$\lambda_1$', ylabel=r'$\lambda_2$', title=title)
    ax.legend()
    return axs[0]


def plot_dyn_snapshots_comparison(
    dynamics0: LinearGaussian,
    dynamics1: LinearGaussian,
    indices,
    *,
    weights0=None,
    weights1=None,
    name0='true',
    name1='inferred',
    xlim=(-4, 4),
    ylim=(-4, 4),
    figsize=None,
    detailed=False,
):
    indices = np.asarray(indices)
    if indices.ndim != 1 or not len(indices):
        raise ValueError(f'expected nonempty indices (N,), got {indices.shape}')
    _, axs = plt.subplots(
        1,
        len(indices),
        squeeze=False,
        figsize=figsize or (3.5 * len(indices), 3.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for ax, index in zip(axs[0], indices):
        plot_dyn_linear_gaussian_comparison(
            ax,
            dynamics0.select(index),
            dynamics1.select(index),
            xlim=xlim,
            ylim=ylim,
            name0=name0,
            name1=name1,
        )
        lines = [f't={index}']
        if detailed:
            for weights, name in ((weights0, name0), (weights1, name1)):
                if weights is not None:
                    lines.append(f'{name}: ' + str(np.asarray(weights[index])))
        ax.set_title('\n'.join(lines))
    return axs[0]


def plot_process_noise_mismatch(true_noise, dynamics_mismatch, *, figsize=(8, 5)):
    """Compare precomputed process-noise and dynamics-mismatch vectors (T, D)."""
    _, axs = plt.subplots(
        2, 1, figsize=figsize, sharex=True, squeeze=False, constrained_layout=True
    )
    for ax, values, title, color in zip(
        axs[:, 0],
        (true_noise, dynamics_mismatch),
        ('True realized process noise', 'Dynamics mismatch'),
        (plot.TRUE_COLOR, plot.INFERRED_COLOR),
    ):
        values = plot._traces(values)
        plot.plot_traces_1d(
            ax, np.linalg.norm(values, axis=-1)[:, None], color=color, linewidth=0.75
        )
        ax.set(xlabel='transition', ylabel='latent norm', title=title)
    return axs[:, 0]


def plot_dynamics_action_error(error, *, title='Dynamics-action error'):
    """Plot an already-computed scalar error per transition."""
    _, axs = plt.subplots(1, 1, figsize=(9, 3), squeeze=False, constrained_layout=True)
    values = np.asarray(error)
    if values.ndim != 1:
        raise ValueError(f'expected error (T,), got {values.shape}')
    plot.plot_traces_1d(
        axs[0, 0], values[:, None], linewidth=0.75, color=plot.INFERRED_COLOR
    )
    axs[0, 0].set(xlabel='transition', ylabel='dynamics-action error', title=title)
    return axs[0]


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


def plot_gated_weight_paths(
    states,
    weights,
    *,
    inferred_weights=None,
    inferred_covariance=None,
    indices=None,
    linewidth=1.25,
    show_zero_lines=True,
    sharex: bool | Literal['all', 'none', 'row', 'col'] = 'all',
    sharey: bool | Literal['all', 'none', 'row', 'col'] = 'all',
):
    """Plot one 2D temporal-weight path per state.

    True active segments use the corresponding state color. True inactive
    segments are black and drawn above active segments. When supplied, the
    inferred posterior-mean path is solid magenta and drawn above both.

    The activity of the segment ``weights[k, j] -> weights[k, j + 1]`` is determined
    by ``states[j + 2]``, matching the gated-random-walk generative convention.
    """
    states = np.asarray(states)
    weights = np.asarray(weights)
    if weights.ndim != 3 or weights.shape[-1] != 2:
        raise ValueError(
            f'expected weights with shape (K, T - 1, 2), got {weights.shape}'
        )
    num_states, num_transitions, _ = weights.shape
    if states.shape != (num_transitions + 1,):
        raise ValueError(
            f'states and weights have incompatible lengths: {states.shape} and {weights.shape}'
        )
    if inferred_weights is not None:
        inferred_weights = np.asarray(inferred_weights)
        if inferred_weights.shape != weights.shape:
            raise ValueError(
                f'inferred_weights must match weights shape, got {inferred_weights.shape} and {weights.shape}'
            )
    if inferred_covariance is not None:
        inferred_covariance = np.asarray(inferred_covariance)
        expected = (num_states, num_transitions, 2, 2)
        if inferred_covariance.shape != expected:
            raise ValueError(
                f'inferred_covariance must have shape {expected}, got {inferred_covariance.shape}'
            )
    segment_states = states[2:]
    cmap = plot.state_cmap(num_states)
    _, axs = plt.subplots(
        ncols=num_states,
        figsize=(3 * num_states, 3),
        constrained_layout=True,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )
    for state, ax in enumerate(axs[0]):
        true_trace = weights[state]
        active = segment_states == state
        segment_labels = np.where(active, state, -1)
        plot.plot_state_trace_2d(
            ax,
            true_trace,
            segment_labels,
            state_colors={state: cmap(state), -1: plot.INACTIVE_COLOR},
            state_zorders={state: 1.0, -1: 2.0},
            linewidth=linewidth,
        )
        if inferred_weights is not None:
            covariance = None
            if inferred_covariance is not None:
                covariance = inferred_covariance[state]
            plot.plot_trace_2d(
                ax,
                inferred_weights[state],
                covariance=covariance,
                color=plot.INFERRED_COLOR,
                linewidth=linewidth,
                zorder=3.0,
            )
        if indices is not None:
            plot.plot_time_markers_2d(ax, true_trace, indices)
        if show_zero_lines:
            ax.axhline(0.0, color='0.6', linewidth=0.5, zorder=0.0)
            ax.axvline(0.0, color='0.6', linewidth=0.5, zorder=0.0)
        ax.autoscale()
        plot.square_axes(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set(xlabel='$\\lambda_1$', ylabel='$\\lambda_2$', title=f'State {state}')
    return axs[0]


def plot_gated_drift_weights(states, weights, figsize=None):
    """Plot state-specific temporal weights and active transition intervals."""
    states = np.asarray(states)
    weights = np.asarray(weights)
    if weights.ndim != 3 or not all(weights.shape):
        raise ValueError(
            f'expected weights with shape (K, T - 1, P), got {weights.shape}'
        )
    num_states, num_steps, num_motifs = weights.shape
    transition_states = states[1:]
    if transition_states.shape != (num_steps,):
        raise ValueError('states and temporal weights have incompatible lengths')
    figsize = figsize or (4, 1 + 1 * num_states)
    _, axs = plt.subplots(
        nrows=1 + num_states,
        figsize=figsize,
        sharex='all',
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={'height_ratios': [1] + [2] * num_states},
    )
    axs = axs[:, 0]
    for ax in axs[2:]:
        ax.sharey(axs[1])
    plot.plot_state_1d(axs[0], transition_states, num_states=num_states, alpha=0.5)
    axs[0].set(title='Active dynamics state', ylabel='state', xlabel='')
    time = np.arange(num_steps)
    cmap = plot.state_cmap(num_states)
    for state in range(num_states):
        ax = axs[state + 1]
        active = np.concatenate([np.asarray([False]), states[2:] == state])
        ax.fill_between(
            time,
            0.0,
            1.0,
            where=active,
            transform=ax.get_xaxis_transform(),
            color=cmap(state),
            alpha=0.25,
            linewidth=0.0,
        )
        for motif in range(num_motifs):
            plot.plot_traces_1d(
                ax,
                weights[state, :, motif, None],
                color=plot.TRUE_COLOR,
                linewidth=1.0,
                label=f'$\\lambda_{motif + 1}$',
            )
        ax.set(ylabel=f'State {state}\n' + '$\\lambda$', title='')
        if num_motifs == 1:
            ax.legend()
    for ax in axs.ravel():
        ax.tick_params(left=False, labelleft=False)
    for ax in axs[:-1]:
        ax.tick_params(bottom=False)
    axs[-1].set_xlabel('time')
    return axs
