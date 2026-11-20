"""Host-side Matplotlib primitives for statistical sequences and dynamics."""

import itertools
from typing import Any, Literal

import jax.numpy as jnp
import matplotlib
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
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
from xxm.plot import trail
from xxm.plot.lines import plot_chunked_line

TRACE_COLORS = ('k', 'xkcd:magenta', *matplotlib.colormaps['Dark2'](np.arange(8)))
STATE_COLORS = tuple(matplotlib.colormaps['tab10'](np.arange(10)))
TRUE_COLOR = 'k'
INFERRED_COLOR = 'xkcd:magenta'
INACTIVE_COLOR = 'k'
DYNAMICS_CMAP = LinearSegmentedColormap.from_list(
    'dynamics', [TRUE_COLOR, INFERRED_COLOR]
)
DYN_MODES = ('trails', 'stream', 'quiver')
DEFAULT_DYN_MODE = 'trails'

# print widths and the tallest figure that leaves room for a caption (Nature), in inches
SINGLE_COLUMN = 89 / 25.4
DOUBLE_COLUMN = 183 / 25.4
MAX_HEIGHT = 170 / 25.4

STYLE: dict[str, Any] = {
    # fonts: Arial if present, metric-compatible fallbacks otherwise
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    # sizes in points: small 5, medium 6, large 7 (Nature: 5-7 pt)
    'font.size': 6,
    'axes.titlesize': 6,
    'axes.labelsize': 6,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5,
    'legend.title_fontsize': 6,
    'figure.titlesize': 7,
    'figure.labelsize': 6,
    # math in the same sans-serif font as the text
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic',
    'mathtext.bf': 'sans:bold',
    'mathtext.sf': 'sans',
    'mathtext.fallback': 'stixsans',
    # pure black
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    # colour-blind-safe cycle (Okabe-Ito, Wong 2011), without yellow
    'axes.prop_cycle': cycler(
        color=[
            '#000000',
            '#e69f00',
            '#56b4e9',
            '#009e73',
            '#0072b2',
            '#d55e00',
            '#cc79a7',
        ]
    ),
    # spines: left and bottom only; full frames with set_spines
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.right': False,
    'axes.spines.top': False,
    # weights (Nature: 0.25-1 pt) and tick sizes, in points
    'axes.linewidth': 0.3,
    'xtick.major.width': 0.3,
    'ytick.major.width': 0.3,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 0.5,
    'lines.markersize': 3,
    'patch.linewidth': 0.5,
    'grid.linewidth': 0.3,
    'hatch.linewidth': 0.3,
    'errorbar.capsize': 1.5,
    # padding, in points
    'axes.labelpad': 1,
    'axes.titlepad': 3,
    'xtick.major.pad': 1,
    'ytick.major.pad': 1,
    'xtick.minor.pad': 1,
    'ytick.minor.pad': 1,
    # legend
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    # 3D: default panes for depth, no grid lines
    'axes3d.grid': False,
    # figure: single-column print size, laid out automatically
    'figure.figsize': (SINGLE_COLUMN, SINGLE_COLUMN / 2),
    'figure.dpi': 200,
    'figure.constrained_layout.use': True,
    # export: editable text in vector formats, rasters at >= 450 dpi
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}


def set_style(**overrides) -> None:
    """Apply `STYLE`, updated with `overrides`, to Matplotlib's rcParams (`plt.style.use`).

    Keys are rcParams names, which contain dots: pass them as
    `set_style(**{'font.size': 7})`. For one figure only, use
    `plt.rc_context(STYLE | overrides)` instead.
    """
    plt.style.use(STYLE | overrides)


def set_spines(ax: Axes, on: bool = True) -> None:
    """Show (`on`) or hide all four spines of `ax`.

    `STYLE` shows only the left and bottom spines; images, heatmaps and
    panels that need a closed frame call `set_spines(ax)`.
    """
    for spine in ax.spines.values():
        spine.set_visible(on)


def set_spines_grid(
    axs: np.ndarray,
    edge: bool = True,
    inner: bool = False,
    side: Literal['default', 'other', 'both'] = 'default',
) -> None:
    """Show the spines on the outer edge of a grid of axes and between its panels.

    `side` picks the spines to set: 'default' the left and bottom ones,
    'other' the top and right ones, 'both' all four; the others are left as
    they are. A spine on the outer edge of the grid (e.g. the left spine of
    the first column) follows `edge`, any other follows `inner`. Hidden spines
    also hide their tick marks. A 1D `axs` is treated as several rows.

    By default only the first column keeps its left spine and only the last
    row its bottom spine.
    """
    axs = np.asarray(axs)
    if axs.ndim == 1:
        axs = axs[:, None]
    rows, cols = axs.shape

    if side == 'default':
        sides = ['left', 'bottom']
    elif side == 'other':
        sides = ['top', 'right']
    else:
        assert side == 'both'
        sides = ['left', 'bottom', 'top', 'right']

    for i in range(rows):
        for j in range(cols):
            ax = axs[i, j]
            for name in sides:
                if name == 'left':
                    outer = j == 0
                elif name == 'right':
                    outer = j == cols - 1
                elif name == 'top':
                    outer = i == 0
                else:  # bottom
                    outer = i == rows - 1

                visible = edge if outer else inner
                ax.spines[name].set_visible(visible)
                if not visible:
                    ax.tick_params(**{name: False})


def square_axes(ax: Axes, aspect='equal') -> None:
    """Give both coordinates the same symmetric range."""
    extent = max(abs(v) for v in (*ax.get_xlim(), *ax.get_ylim()))
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect=aspect)


def symmetric_limits(
    values, *, margin: float = 0.1, quantile: float = 1.0
) -> tuple[float, float]:
    r"""Limits $\pm(1 + \text{margin})\, q$, with $q$ the `quantile` of $|v|$ over `values`.

    The default `quantile` covers all values; a lower one ignores outliers.
    Non-finite values are ignored.
    """
    magnitudes = np.abs(np.asarray(values, dtype=float)).ravel()
    magnitudes = magnitudes[np.isfinite(magnitudes)]
    if not magnitudes.size:
        raise ValueError('symmetric_limits needs at least one finite value')
    extent = (1 + margin) * float(np.quantile(magnitudes, quantile))
    return -extent, extent


def panel_note(ax: Axes, text: str):
    """Write a small note in the lower-left corner of a panel; return the `Text`."""
    return ax.text(
        0.03,
        0.03,
        text,
        transform=ax.transAxes,
        fontsize='small',
        bbox={'facecolor': 'white', 'alpha': 0.7, 'edgecolor': 'none', 'pad': 1},
    )


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


def _trace_batch(traces, dimension: int) -> np.ndarray:
    """A `(T, D)` trajectory or `(B, T, D)` batch as a `(B, T, D)` array."""
    values = np.asarray(traces)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[-1] != dimension or not all(values.shape):
        raise ValueError(
            f'expected traces with shape (T, {dimension}) or (B, T, {dimension}), '
            f'got {values.shape}'
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


def plot_states(ax: Axes, states, **kwargs):
    """Draw states as a panel of their own: framed, without ticks, labelled 'state'.

    `kwargs` go to `plot_state_1d`.
    """
    image = plot_state_1d(ax, states, **kwargs)
    set_spines(ax)
    hide_ticks(ax)
    ax.set(xlabel='', ylabel='state')
    return image


def share_time_axis(axs) -> None:
    """In a stack of panels, keep the x label and ticks on the bottom one only."""
    for ax in np.ravel(axs)[:-1]:
        ax.set(xlabel='')
        hide_ticks(ax, axis='x')


def plot_traces_1d(ax: Axes, traces, **kwargs):
    """Draw one line per column of a (T, D) array."""
    values = _traces(traces)
    lines = []
    for trace, color in zip(values.T, itertools.cycle(TRACE_COLORS)):
        lines.extend(ax.plot(trace, **(dict[str, Any](color=color) | kwargs)))
    return lines


def plot_traces_2d(ax: Axes, traces, *, mark_endpoints=True, **kwargs):
    """Draw a (T, 2) trajectory or a (B, T, 2) batch.

    Each trace is drawn as chunks (`xxm.plot.lines.plot_chunked_line`), so that with
    alpha its overlapping parts stack; `kwargs` go to the `LineCollection`
    (e.g. `color`, `linewidth`, `alpha`, `label`, `num_chunks`).
    """
    values = _trace_batch(traces, 2)
    for trace, color in zip(values, itertools.cycle(TRACE_COLORS)):
        plot_chunked_line(ax, trace, **(dict[str, Any](color=color) | kwargs))
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
    set_spines(ax)  # flow fields and heatmaps are framed
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
    set_spines(ax)  # the state shading is a background


def plot_seq_2d(
    ax: Axes,
    states,
    traces,
    *,
    desc='observations',
    num_states=None,
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
            c=np.asarray(states),
            s=size,
            cmap=state_cmap(count),
            norm=state_norm(count),
            alpha=0.5,
            edgecolor='none',
            zorder=10,
        )

        options = options | (scatter_kwargs or {})

        ax.scatter(values[:, 0], values[:, 1], **options)

    plot_traces_2d(
        ax,
        values,
        mark_endpoints=mark_endpoints,
        **(dict[str, Any](alpha=alpha) | (line_kwargs or {})),
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
                    zorder=(state_zorders or {}).get(label, 1),
                    capstyle='butt',
                )
                | kwargs
            )
            ax.add_collection(
                LineCollection(list(segments[states == label]), **options)
            )
    ax.autoscale_view()


def has_label(label) -> bool:
    """Whether a label is present, i.e. neither None nor NaN."""
    return label is not None and not (isinstance(label, float) and np.isnan(label))


def plot_points_by_label_2d(ax: Axes, points, labels, *, label_colors=None, **kwargs):
    """Scatter (T, 2) points colored by their (T,) labels; return {label: artist}.

    Missing labels (None or NaN) are not drawn. Labels absent from
    `label_colors` cycle through `TRACE_COLORS`.
    """
    values = _traces(points, 2)
    labels = np.asarray(labels)
    if labels.shape != (len(values),):
        raise ValueError(
            f'expected labels with shape {(len(values),)}, got {labels.shape}'
        )

    present = np.array([has_label(label) for label in labels], dtype=bool)
    colors = dict(label_colors or {})
    fallback = itertools.cycle(TRACE_COLORS)

    artists = {}
    for label in np.unique(labels[present]):
        color = colors[label] if label in colors else next(fallback)
        options = dict[str, Any](s=2, alpha=0.75, edgecolor='none', zorder=3) | kwargs
        artists[label] = ax.scatter(*values[labels == label].T, color=color, **options)
    return artists


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
    options = dict[str, Any](fill=False, color=TRUE_COLOR) | kwargs
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
        **(dict[str, Any](color=color, zorder=zorder) | (line_kwargs or {})),
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
    set_spines(ax)  # flow fields and heatmaps are framed
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
    set_spines(ax)  # flow fields and heatmaps are framed
    return artist


def binned_displacement(
    traces, *, xlim=(-3, 3), ylim=(-3, 3), num_bins: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Mean one-step displacement of 2D traces, binned by the starting point.

    For a `(T, 2)` trajectory or `(B, T, 2)` batch $z_t$, each bin of the
    `num_bins × num_bins` grid over `xlim × ylim` averages $z_{t+1} - z_t$
    over the steps starting in it: an estimate of
    $\mathbb{E}[z_{t+1} - z_t \mid z_t]$. Returns bin `centers` and `means`,
    both `(num_bins, num_bins, 2)` and indexed `[x bin, y bin]`, and `counts`
    `(num_bins, num_bins)`. Means are NaN in empty bins.
    """
    values = _trace_batch(traces, 2)
    starts = values[:, :-1].reshape(-1, 2)
    steps = np.diff(values, axis=1).reshape(-1, 2)

    def histogram(weights=None) -> np.ndarray:
        counts, _, _ = np.histogram2d(
            starts[:, 0],
            starts[:, 1],
            bins=num_bins,
            range=(xlim, ylim),
            weights=weights,
        )
        return counts

    counts = histogram()
    sums = np.stack([histogram(steps[:, k]) for k in range(2)], axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = np.where(counts[..., None] > 0, sums / counts[..., None], np.nan)

    edges = [np.linspace(low, high, num_bins + 1) for low, high in (xlim, ylim)]
    centers = np.stack(
        np.meshgrid(*[(e[:-1] + e[1:]) / 2 for e in edges], indexing='ij'), axis=-1
    )
    return centers, means, counts


def plot_dyn_binned_2d(
    ax: Axes,
    traces,
    *,
    xlim=(-3, 3),
    ylim=(-3, 3),
    num_bins: int = 11,
    min_count: int = 10,
    color=TRUE_COLOR,
    width=0.004,
    **kwargs,
):
    """Draw the binned mean displacement of 2D traces as arrows (`binned_displacement`).

    Bins with fewer than `min_count` steps are left empty. `kwargs` go to
    `ax.quiver`; e.g. `angles='xy', scale_units='xy', scale=1` draws arrows
    in data units, comparable with a model's displacement at the bin centers.
    """
    centers, means, counts = binned_displacement(
        traces, xlim=xlim, ylim=ylim, num_bins=num_bins
    )
    kept = counts >= min_count
    artist = ax.quiver(
        centers[kept, 0],
        centers[kept, 1],
        means[kept, 0],
        means[kept, 1],
        color=color,
        width=width,
        **kwargs,
    )
    ax.set(xlim=xlim, ylim=ylim, aspect='equal', xlabel='$x_1$', ylabel='$x_2$')
    set_spines(ax)  # flow fields and heatmaps are framed
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
        arrowsize=arrowsize,
        **kwargs,
    )
    ax.set(xlim=xlim, ylim=ylim, aspect='equal', xlabel='$x_1$', ylabel='$x_2$')
    set_spines(ax)  # flow fields and heatmaps are framed
    return artist


def plot_dyn_linear_trails(ax: Axes, affine: Affine, *, color=TRUE_COLOR, **kwargs):
    """Draw trails of affine(x) (`xxm.plot.trail.plot_dyn_trails`)."""
    artists = trail.plot_dyn_trails(ax, _affine_2d(affine), color=color, **kwargs)
    ax.set(xlabel='$x_1$', ylabel='$x_2$')
    set_spines(ax)  # flow fields and heatmaps are framed
    return artists


def plot_dyn_linear(ax: Axes, affine: Affine, *, mode=DEFAULT_DYN_MODE, **kwargs):
    """Draw 2D affine dynamics as `'trails'`, `'stream'` lines or `'quiver'` arrows."""
    affine = _affine_2d(affine)
    if mode == 'trails':
        return plot_dyn_linear_trails(ax, affine, **kwargs)
    if mode == 'quiver':
        return plot_dyn_linear_quiver(ax, affine, **kwargs)
    if mode == 'stream':
        return plot_dyn_linear_stream(ax, affine, **kwargs)
    raise ValueError(f'unknown dynamics mode {mode!r}; expected one of {DYN_MODES}')


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
            dict[str, Any](color='w', angles='xy', scale_units='xy')
            | (quiver_kwargs or {})
        ),
    )
    ax.set(xlim=xlim, ylim=ylim, aspect='equal', xlabel='$x_1$', ylabel='$x_2$')
    set_spines(ax)  # flow fields and heatmaps are framed
    return mesh


def hide_ticks(ax: Axes, *, axis: Literal['x', 'y', 'both'] = 'both') -> None:
    """Remove the ticks and tick labels of `ax` along `axis` ('x', 'y' or both)."""

    ax.tick_params(
        axis=axis,
        which='both',
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False,
    )
