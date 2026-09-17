"""Host-side Matplotlib primitives for statistical sequences and dynamics."""

import itertools
from typing import Any, Literal

import jax.numpy as jnp
import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)
from matplotlib.patches import Ellipse

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian

TRACE_COLORS = ('k', 'xkcd:magenta', *matplotlib.colormaps['Dark2'](np.arange(8)))
STATE_COLORS = tuple(matplotlib.colormaps['tab10'](np.arange(10)))
TRUE_COLOR = 'k'
INFERRED_COLOR = 'xkcd:magenta'
INACTIVE_COLOR = 'k'
DYNAMICS_CMAP = LinearSegmentedColormap.from_list(
    'dynamics', [TRUE_COLOR, INFERRED_COLOR]
)


def square_axes(ax: Axes, aspect='equal') -> None:
    """Give both coordinates the same symmetric range."""
    extent = max(abs(v) for v in (*ax.get_xlim(), *ax.get_ylim()))
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect=aspect)


def state_cmap(num_states: int) -> ListedColormap:
    if num_states < 1:
        raise ValueError(f'expected positive num_states, got {num_states}')
    return ListedColormap(
        [STATE_COLORS[i % len(STATE_COLORS)] for i in range(num_states)]
    )


def state_norm(num_states: int) -> BoundaryNorm:
    if num_states < 1:
        raise ValueError(f'expected positive num_states, got {num_states}')
    return BoundaryNorm(np.arange(num_states + 1) - 0.5, num_states)


def _num_states(states, num_states=None) -> int:
    values = np.asarray(states)
    if values.ndim != 1 or not values.size:
        raise ValueError(
            f'expected nonempty states with shape (T,), got {values.shape}'
        )
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0)
        or np.any(values != np.floor(values))
    ):
        raise ValueError('expected nonnegative integer state labels')
    required = int(values.max()) + 1
    if num_states is not None and num_states < required:
        raise ValueError(f'expected num_states >= {required}, got {num_states}')
    return required if num_states is None else num_states


def _traces(values, dimension=None):
    values = np.asarray(values)
    if (
        values.ndim != 2
        or not all(values.shape)
        or (dimension is not None and values.shape[1] != dimension)
    ):
        raise ValueError(
            f'expected nonempty traces with shape (T, {dimension or "D"}), got {values.shape}'
        )
    return values


def plot_state_1d(
    ax: Axes,
    states,
    *,
    num_states=None,
    aspect: float | Literal['auto', 'equal'] = 'auto',
    interpolation='none',
    alpha=0.25,
    **kwargs,
):
    """Draw states in axes-height coordinates, centered on integer sample times."""
    states = np.asarray(states)
    count = _num_states(states, num_states)
    options = dict[str, Any](
        cmap=state_cmap(count),
        norm=state_norm(count),
        extent=(-0.5, len(states) - 0.5, 0, 1),
        transform=ax.get_xaxis_transform(),
    )
    options.update(kwargs)
    limits = ax.get_ylim()
    autoscale_y = ax.get_autoscaley_on()
    image = ax.imshow(
        states[None, :],
        aspect=aspect,
        interpolation=interpolation,
        alpha=alpha,
        **options,
    )
    ax.set_ylim(limits)
    ax.set_autoscaley_on(autoscale_y)
    ax.set_xlabel('time')
    return image


def plot_traces_1d(ax: Axes, traces, **kwargs):
    """Draw one line per column of a (T, D) array."""
    values = _traces(traces)
    lines = []
    for trace, color in zip(values.T, itertools.cycle(TRACE_COLORS)):
        lines.extend(ax.plot(trace, **(dict[str, Any](color=color) | kwargs)))
    return lines


def plot_traces_2d(ax: Axes, traces, *, mark_endpoints=True, **kwargs):
    """Draw a (T, 2) trajectory or a (B, T, 2) batch."""
    values = np.asarray(traces)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[-1] != 2 or not all(values.shape):
        raise ValueError(
            f'expected traces with shape (T, 2) or (B, T, 2), got {values.shape}'
        )
    for trace, color in zip(values, itertools.cycle(TRACE_COLORS)):
        ax.plot(trace[:, 0], trace[:, 1], **(dict[str, Any](color=color) | kwargs))
        if mark_endpoints:
            ax.scatter(*trace[0], color='g', zorder=10)
            ax.scatter(*trace[-1], color='r', zorder=10)
    ax.set(xlabel='$x_1$', ylabel='$x_2$', aspect='equal')


def plot_traces_image(ax: Axes, values, *, xlabel='time', ylabel='variable', **kwargs):
    """Draw (T, D) values; return the image without adding a colorbar."""
    values = _traces(values)
    image = ax.imshow(
        values.T, **(dict[str, Any](aspect='auto', interpolation='none') | kwargs)
    )
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return image


def plot_seq_1d(
    ax: Axes,
    states,
    traces,
    *,
    desc='observations',
    state_kwargs=None,
    trace_kwargs=None,
):
    values = _traces(traces)
    if np.shape(states) != (len(values),):
        raise ValueError(
            f'expected states with shape {(len(values),)}, got {np.shape(states)}'
        )
    plot_traces_1d(ax, values, **(trace_kwargs or {}))
    plot_state_1d(ax, states, **(state_kwargs or {}))
    ax.set(xlabel='time', ylabel=desc)


def plot_seq_2d(
    ax: Axes,
    states,
    traces,
    *,
    desc='observations',
    num_states=None,
    linewidth=0.75,
    alpha=0.35,
    size=12,
    mark_endpoints=True,
    line_kwargs=None,
    scatter_kwargs=None,
):
    values = _traces(traces, 2)
    if states is not None:
        count = _num_states(states, num_states)
        if np.shape(states) != (len(values),):
            raise ValueError(
                f'expected states with shape {(len(values),)}, got {np.shape(states)}'
            )
        options = dict[str, Any](
            c=np.asarray(states), s=size, cmap=state_cmap(count), norm=state_norm(count)
        ) | (scatter_kwargs or {})
        ax.scatter(values[:, 0], values[:, 1], **options)
    plot_traces_2d(
        ax,
        values,
        mark_endpoints=mark_endpoints,
        **(dict[str, Any](linewidth=linewidth, alpha=alpha) | (line_kwargs or {})),
    )
    ax.set(xlabel=f'{desc} 1', ylabel=f'{desc} 2')


def _trace_segments(traces):
    values = _traces(traces, 2)
    if len(values) < 2:
        raise ValueError('expected at least two trace points')
    return np.stack([values[:-1], values[1:]], axis=1)


def plot_state_trace_2d(
    ax: Axes,
    traces,
    states,
    *,
    state_colors=None,
    state_zorders=None,
    linewidth=1.25,
    **kwargs,
):
    """Color each adjacent segment by its label; states has shape (T - 1,)."""
    segments = _trace_segments(traces)
    states = np.asarray(states)
    if states.shape != (len(segments),):
        raise ValueError(
            f'expected states with shape {(len(segments),)}, got {states.shape}'
        )
    if state_colors is None:
        count = _num_states(states)
        state_colors = {k: state_cmap(count)(k) for k in range(count)}
    for label in np.unique(states):
        if label in state_colors:
            options = (
                dict[str, Any](
                    colors=[state_colors[label]],
                    linewidths=linewidth,
                    zorder=(state_zorders or {}).get(label, 1),
                    capstyle='butt',
                )
                | kwargs
            )
            ax.add_collection(
                LineCollection(list(segments[states == label]), **options)
            )
    ax.autoscale_view()


def plot_time_markers_2d(
    ax: Axes,
    traces,
    indices,
    *,
    size=55.0,
    edgecolor='k',
    text_offset=(5.0, 5.0),
    zorder=10.0,
    **kwargs,
):
    values = _traces(traces, 2)
    indices = np.asarray(indices)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= len(values))
    ):
        raise ValueError(
            f'expected 1D integer indices in [0, {len(values)}), got {indices}'
        )
    for index in indices:
        point = values[index]
        ax.scatter(
            *point,
            **(
                dict[str, Any](
                    s=size, facecolor='none', edgecolor=edgecolor, zorder=zorder
                )
                | kwargs
            ),
        )
        ax.annotate(
            f't={index}',
            (float(point[0]), float(point[1])),
            xytext=text_offset,
            textcoords='offset points',
            fontsize='small',
            zorder=zorder + 1,
        )


def _ellipse(ax: Axes, mean, covariance, scale=1.0, **kwargs):
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError(
            f'expected mean (2,) and covariance (2, 2), got {mean.shape} and {covariance.shape}'
        )
    if scale < 0:
        raise ValueError('expected nonnegative ellipse scale')
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) / 2)
    if np.min(eigenvalues) < -1e-7:
        raise ValueError('expected positive semidefinite covariance')
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    ellipse = Ellipse(
        (float(mean[0]), float(mean[1])),
        width=float(2 * scale * np.sqrt(max(eigenvalues[1], 0))),
        height=float(2 * scale * np.sqrt(max(eigenvalues[0], 0))),
        angle=float(angle),
        **kwargs,
    )
    ax.add_patch(ellipse)
    return ellipse


def plot_gaussian_2d(
    ax: Axes, gaussian: Gaussian, *, scale=1.0, mark_mean=True, **kwargs
):
    """Draw an unbatched Gaussian; scale is the number of standard deviations."""
    if gaussian.batch_shape or gaussian.variable_dim != 2:
        raise ValueError(
            f'expected unbatched 2D Gaussian, got batch {gaussian.batch_shape}, dimension {gaussian.variable_dim}'
        )
    options = dict[str, Any](fill=False, linewidth=2, color=TRUE_COLOR) | kwargs
    ellipse = _ellipse(ax, gaussian.mean, gaussian.covariance, scale, **options)
    if mark_mean:
        ax.scatter(*np.asarray(gaussian.mean), color=options['color'], s=40)
    ax.autoscale_view()
    return ellipse


def plot_trace_2d(
    ax: Axes,
    traces,
    covariance=None,
    *,
    color=INFERRED_COLOR,
    linewidth=1.25,
    zorder=3.0,
    covariance_stride=None,
    covariance_scale=2.0,
    covariance_alpha=0.08,
    line_kwargs=None,
    ellipse_kwargs=None,
):
    values = _traces(traces, 2)
    if covariance is not None:
        covariance = np.asarray(covariance)
        if covariance.shape != (len(values), 2, 2):
            raise ValueError(
                f'expected covariance with shape {(len(values), 2, 2)}, got {covariance.shape}'
            )
        stride = (
            max(1, len(values) // 20)
            if covariance_stride is None
            else covariance_stride
        )
        if not isinstance(stride, int) or stride < 1:
            raise ValueError('expected positive integer covariance_stride')
        for index in range(0, len(values), stride):
            _ellipse(
                ax,
                values[index],
                covariance[index],
                covariance_scale,
                **(
                    dict[str, Any](
                        facecolor=color,
                        edgecolor='none',
                        alpha=covariance_alpha,
                        zorder=zorder - 0.1,
                    )
                    | (ellipse_kwargs or {})
                ),
            )
    plot_traces_2d(
        ax,
        values,
        mark_endpoints=False,
        **(
            dict[str, Any](color=color, linewidth=linewidth, zorder=zorder)
            | (line_kwargs or {})
        ),
    )


def _affine_2d(affine: Affine) -> Affine:
    if affine.input_shape == (1, 2):
        affine = affine.squeeze_input()
    if affine.batch_shape or affine.input_shape != (2,) or affine.output_dim != 2:
        raise ValueError(
            f'expected unbatched affine (2,) -> (2,), got batch {affine.batch_shape}, {affine.input_shape} -> ({affine.output_dim},)'
        )
    return affine


def _evaluate_dyn_2d(mean, *, xlim, ylim, num_points):
    if num_points < 2 or xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
        raise ValueError('expected increasing limits and num_points >= 2')
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], num_points),
        np.linspace(ylim[0], ylim[1], num_points),
    )
    points = np.stack([xx, yy], axis=-1)
    next_points = np.asarray(mean(jnp.asarray(points)))
    if next_points.shape != points.shape:
        raise ValueError(
            f'expected mean output shape {points.shape}, got {next_points.shape}'
        )
    displacement = next_points - points
    return xx, yy, displacement[..., 0], displacement[..., 1]


def _field_color(dx, dy, color, norm):
    if isinstance(color, str):
        if color == 'speed':
            return np.hypot(dx, dy), norm
        if color == 'log_speed':
            return np.log(np.maximum(np.hypot(dx, dy), np.finfo(float).tiny)), norm
        if color == 'angle':
            return np.arctan2(dy, dx), norm if norm is not None else Normalize(
                -np.pi, np.pi
            )
    return None, norm


def plot_dyn_2d(ax: Axes, mean, *, xlim=(-3, 3), ylim=(-3, 3), num_points=15, **kwargs):
    """Plot discrete-time displacement mean(x) - x."""
    field = _evaluate_dyn_2d(mean, xlim=xlim, ylim=ylim, num_points=num_points)
    artist = ax.quiver(*field, **kwargs)
    ax.set(xlim=xlim, ylim=ylim, aspect='equal', xlabel='$x_1$', ylabel='$x_2$')
    return artist


def plot_dyn_linear_quiver(
    ax: Axes,
    affine: Affine,
    *,
    xlim=(-3, 3),
    ylim=(-3, 3),
    num_points=15,
    color='speed',
    cmap=None,
    norm=None,
    scale=None,
    width=0.004,
    **kwargs,
):
    """Draw arrows for affine(x) - x; color may be speed, log_speed or angle."""
    affine = _affine_2d(affine)
    field = _evaluate_dyn_2d(affine.apply, xlim=xlim, ylim=ylim, num_points=num_points)
    colors, norm = _field_color(field[2], field[3], color, norm)
    if colors is None:
        artist = ax.quiver(*field, color=color, scale=scale, width=width, **kwargs)
    else:
        artist = ax.quiver(
            *field,
            colors,
            cmap=DYNAMICS_CMAP if cmap is None else cmap,
            norm=norm,
            scale=scale,
            width=width,
            **kwargs,
        )
    ax.set(xlim=xlim, ylim=ylim, aspect='equal', xlabel='$x_1$', ylabel='$x_2$')
    return artist


def plot_dyn_linear_stream(
    ax: Axes,
    affine: Affine,
    *,
    xlim=(-3, 3),
    ylim=(-3, 3),
    num_points=15,
    color='speed',
    cmap=None,
    norm=None,
    density=1.5,
    linewidth=0.5,
    arrowsize=0.75,
    **kwargs,
):
    """Draw streamlines of the discrete-time displacement affine(x) - x."""
    affine = _affine_2d(affine)
    field = _evaluate_dyn_2d(affine.apply, xlim=xlim, ylim=ylim, num_points=num_points)
    colors, norm = _field_color(field[2], field[3], color, norm)
    artist = ax.streamplot(
        *field,
        color=color if colors is None else colors,
        cmap=DYNAMICS_CMAP if cmap is None else cmap,
        norm=norm,
        density=density,
        linewidth=linewidth,
        arrowsize=arrowsize,
        **kwargs,
    )
    ax.set(xlim=xlim, ylim=ylim, aspect='equal', xlabel='$x_1$', ylabel='$x_2$')
    return artist


def plot_dyn_linear(ax: Axes, affine: Affine, *, mode='quiver', **kwargs):
    affine = _affine_2d(affine)
    if mode == 'quiver':
        return plot_dyn_linear_quiver(ax, affine, **kwargs)
    if mode == 'stream':
        return plot_dyn_linear_stream(ax, affine, **kwargs)
    raise ValueError(f'unknown dynamics mode: {mode}')


def plot_dyn_linear_gaussian(ax: Axes, dynamics: LinearGaussian, **kwargs):
    """Display only the conditional mean, without implying process-noise geometry."""
    if dynamics.batch_shape:
        raise ValueError(
            f'expected unbatched LinearGaussian, got batch {dynamics.batch_shape}'
        )
    return plot_dyn_linear(ax, dynamics.affine, **kwargs)


def _dyn_mismatch(
    affine0, affine1, *, xlim, ylim, num_points, relative=False, relative_eps=1e-3
):
    if relative_eps <= 0:
        raise ValueError('expected positive relative_eps')
    xx, yy, dx0, dy0 = _evaluate_dyn_2d(
        _affine_2d(affine0).apply, xlim=xlim, ylim=ylim, num_points=num_points
    )
    _, _, dx1, dy1 = _evaluate_dyn_2d(
        _affine_2d(affine1).apply, xlim=xlim, ylim=ylim, num_points=num_points
    )
    dx, dy = dx1 - dx0, dy1 - dy0
    magnitude = np.hypot(dx, dy)
    if relative:
        magnitude = magnitude / np.maximum(np.hypot(dx0, dy0), relative_eps)
    return xx, yy, dx, dy, magnitude


def plot_dyn_mismatch(
    ax: Axes,
    affine0: Affine,
    affine1: Affine,
    *,
    xlim=(-4, 4),
    ylim=(-4, 4),
    num_points=100,
    quiver_points=15,
    normalize_quiver=False,
    relative=False,
    relative_eps=1e-3,
    vmin=0.0,
    vmax=None,
    mesh_kwargs=None,
    quiver_kwargs=None,
):
    """Plot displacement1 - displacement0, relative to |affine0(x) - x| if requested.

    Relative magnitudes divide by max(reference displacement speed, relative_eps).
    Arrows show the raw difference, or unit direction with normalize_quiver.
    """
    xx, yy, _, _, magnitude = _dyn_mismatch(
        affine0,
        affine1,
        xlim=xlim,
        ylim=ylim,
        num_points=num_points,
        relative=relative,
        relative_eps=relative_eps,
    )
    options = dict[str, Any](cmap=DYNAMICS_CMAP, shading='auto') | (mesh_kwargs or {})
    if 'norm' not in options:
        options = dict[str, Any](vmin=vmin, vmax=vmax) | options
    mesh = ax.pcolormesh(xx, yy, magnitude, **options)
    xx, yy, dx, dy, _ = _dyn_mismatch(
        affine0, affine1, xlim=xlim, ylim=ylim, num_points=quiver_points
    )
    if normalize_quiver:
        speed = np.maximum(np.hypot(dx, dy), np.finfo(float).tiny)
        dx, dy = dx / speed, dy / speed
    ax.quiver(
        xx,
        yy,
        dx,
        dy,
        **(
            dict[str, Any](color='k', angles='xy', scale_units='xy')
            | (quiver_kwargs or {})
        ),
    )
    ax.set(xlim=xlim, ylim=ylim, aspect='equal', xlabel='$x_1$', ylabel='$x_2$')
    return mesh
