r"""3D Matplotlib primitives and multiview figures.

The primitives mirror the 2D ones of `xxm.plot.core`: `plot_traces_3d` for
`plot_traces_2d` and `plot_chunked_line_3d` for
`xxm.plot.lines.plot_chunked_line`. Planes and lines are drawn from the
`Affine` embeddings of `xxm.plot.planes`: `plot_planes_3d` for a batch of
planes `(2,) -> (3,)` that occlude each other correctly, and `plot_line_3d`
for a line `(1,) -> (3,)`.

mplot3d has no depth buffer, and only the faces within one collection are
sorted by depth. Things that must occlude each other correctly are therefore
cut into small `Pieces3D` (`plane_pieces`, `plane_border_pieces`,
`trace_pieces`) and drawn together by one `plot_pieces_3d` call.

A multiview figure (`multiview_axes`) shows 3D space twice: a 3D axes in
latent coordinates, and three 2D panels on the planes of an orthonormal
frame $Q = [q_1, q_2, q_3]$ (`planes.frame_planes`), laid out like an
engineering drawing:

    (q1, q2)  |  3D
    (q1, q3)  |  (q2, q3)

Panels on the left share the $q_1$ axis, panels at the bottom the $q_3$
axis. Trajectories are drawn on both (`plot_traces_multiview`) or on the
panels only (`plot_traces_panels`). Flows are drawn only on the panels, with
the 2D tools (`plot_dyn_linear_multiview`); the 3D axes holds trajectories
and planes.
"""

import contextlib
import itertools
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from xxm.core.affine import Affine
from xxm.plot import core as plot
from xxm.plot import lines, planes, trail

LATENT_NAMES = ('$x_1$', '$x_2$', '$x_3$')
FRAME_PAIRS = tuple(itertools.combinations(range(3), 2))  # as planes.frame_planes


def plot_chunked_line_3d(
    ax: Axes3D,
    points,
    *,
    num_chunks: int = lines.DEFAULT_NUM_CHUNKS,
    breaks=None,
    **kwargs,
) -> Line3DCollection:
    """Draw a `(T, 3)` line as chunks in one `Line3DCollection`; return it.

    The 3D counterpart of `lines.plot_chunked_line`, with the same chunks
    (`lines.chunk_bounds`) and options; data limits are updated.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f'points must have shape (T, 3); got {points.shape}')
    bounds = lines.chunk_bounds(
        np.all(np.isfinite(points), axis=-1), num_chunks=num_chunks, breaks=breaks
    )
    kwargs.setdefault('capstyle', 'butt')
    collection = Line3DCollection(lines.chunk_line(points, bounds), **kwargs)
    ax.add_collection3d(collection)
    return collection


def plot_traces_3d(ax: Axes3D, traces, *, mark_endpoints=True, **kwargs):
    """Draw a (T, 3) trajectory or a (B, T, 3) batch, as `plot.plot_traces_2d`."""
    values = plot._trace_batch(traces, 3)
    for trace, color in zip(values, itertools.cycle(plot.TRACE_COLORS)):
        plot_chunked_line_3d(ax, trace, **(dict[str, Any](color=color) | kwargs))
        if mark_endpoints:
            ax.scatter(*trace[0], color='g', zorder=10)
            ax.scatter(*trace[-1], color='r', zorder=10)
    ax.set(xlabel=LATENT_NAMES[0], ylabel=LATENT_NAMES[1], zlabel=LATENT_NAMES[2])
    ax.set_aspect('equal')


def hide_ticks_3d(ax: Axes3D, *, labelpad: float = -15.0) -> None:
    """Remove the ticks, tick labels and grid of a 3D axes; keep its labels.

    mplot3d places axis labels `labelpad + 21` points out from the axis,
    reserving room for tick labels; the negative default `labelpad` takes most
    of it back.
    """
    ax.set(xticks=[], yticks=[], zticks=[])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.labelpad = labelpad


class Pieces3D(NamedTuple):
    """Small 3D polygons and open polylines, to be depth-sorted together.

    `vertices[k]` `(N_k, 3)` is filled in `facecolors[k]` and stroked in
    `edgecolors[k]` (RGBA, `(K, 4)`); a transparent color leaves it unfilled
    or unstroked. Built by `plane_pieces` and `trace_pieces`, drawn by
    `plot_pieces_3d`.
    """

    vertices: list[np.ndarray]
    facecolors: np.ndarray  # (K, 4)
    edgecolors: np.ndarray  # (K, 4)


_TRANSPARENT = (0.0, 0.0, 0.0, 0.0)


def _pieces(vertices: list[np.ndarray], *, face, edge) -> Pieces3D:
    count = len(vertices)
    return Pieces3D(
        vertices,
        np.tile(to_rgba(face), (count, 1)).reshape(count, 4),
        np.tile(to_rgba(edge), (count, 1)).reshape(count, 4),
    )


def plane_pieces(
    embeddings: Affine,
    *,
    xlim=(-1, 1),
    ylim=(-1, 1),
    colors: Sequence[Any] = ('C0', 'C1', 'C2'),
    alpha: float = 0.3,
    tiles: int = 6,
) -> Pieces3D:
    r"""Pieces of planes, each over `xlim × ylim`, that sort correctly by depth.

    `embeddings` is a batch `(K,)` of `(2,) -> (3,)` maps, e.g.
    `planes.frame_planes` or `planes.conditional_lift`, filled in `colors`
    with `alpha`. Each plane is cut into `tiles × tiles` tiles, and the tiles
    along its intersection lines with the other planes
    (`planes.plane_intersection`; parallel planes are not cut), so that
    pieces only meet at edges and are small enough to be compared locally.
    Their borders are `plane_border_pieces`.
    """
    _check_planes(embeddings, colors, tiles)
    return _concatenate(
        [
            _pieces(tiles_k, face=to_rgba(colors[k], alpha), edge=_TRANSPARENT)
            for k, tiles_k in enumerate(_plane_tiles(embeddings, xlim, ylim, tiles))
        ]
    )


def plane_border_pieces(
    embeddings: Affine,
    *,
    xlim=(-1, 1),
    ylim=(-1, 1),
    colors: Sequence[Any] = ('C0', 'C1', 'C2'),
    alpha: float = 0.3,
    tiles: int = 6,
) -> Pieces3D:
    """Borders of the planes of `plane_pieces`, as segments one tile long.

    Short segments sort by depth with the tiles; arguments are as for
    `plane_pieces`, with `colors` and `alpha` stroking the borders.
    """
    _check_planes(embeddings, colors, tiles)
    return _concatenate(
        [
            _pieces(
                _plane_border(embeddings.select(k), xlim, ylim, tiles),
                face=_TRANSPARENT,
                edge=to_rgba(colors[k], alpha),
            )
            for k in range(embeddings.batch_shape[0])
        ]
    )


def _check_planes(embeddings: Affine, colors: Sequence[Any], tiles: int) -> None:
    if len(embeddings.batch_shape) != 1:
        raise ValueError(
            f'expected embeddings with batch (K,); got {embeddings.batch_shape}'
        )
    if len(colors) < embeddings.batch_shape[0]:
        raise ValueError(
            f'expected {embeddings.batch_shape[0]} colors; got {len(colors)}'
        )
    if tiles < 1:
        raise ValueError(f'tiles must be at least 1; got {tiles}')


def trace_pieces(
    traces, *, color: Any = 'k', alpha: float = 1.0, chunk_steps: int = 2
) -> Pieces3D:
    """A `(T, 3)` trajectory or `(B, T, 3)` batch as open polylines of `chunk_steps` steps.

    The chunks are those of `lines.chunk_bounds`, short so that each lies on
    one side of nearby pieces; non-finite samples leave gaps.
    """
    if chunk_steps < 1:
        raise ValueError(f'chunk_steps must be at least 1; got {chunk_steps}')
    chunks = []
    for trace in plot._trace_batch(traces, 3):
        num_chunks = -(-(len(trace) - 1) // chunk_steps)  # ceil
        bounds = lines.chunk_bounds(
            np.all(np.isfinite(trace), axis=-1), num_chunks=max(num_chunks, 1)
        )
        chunks.extend(lines.chunk_line(trace, bounds))
    return _pieces(chunks, face=_TRANSPARENT, edge=to_rgba(color, alpha))


def plot_pieces_3d(
    ax: Axes3D, *pieces: Pieces3D, linewidth: float = 0.5, **kwargs
) -> Poly3DCollection:
    """Draw pieces in one `Poly3DCollection`, so that they are depth-sorted together.

    mplot3d draws without a depth buffer. Separate artists are painted whole,
    in the order of their nearest points, so e.g. a trajectory drawn apart
    from planes lies entirely in front of or behind them. Within one
    collection, faces are sorted by their mean depth at every draw: pass
    everything that should occlude correctly in one call. All pieces share
    `linewidth`, since mplot3d does not reorder per-face widths. `kwargs` go
    to `Poly3DCollection` (e.g. `zorder`).
    """
    combined = _concatenate(list(pieces))
    collection = Poly3DCollection(
        combined.vertices,
        closed=False,  # polylines stay open; polygons are filled regardless
        facecolors=combined.facecolors,
        edgecolors=combined.edgecolors,
        linewidths=linewidth,
        **kwargs,
    )
    ax.add_collection3d(collection)
    return collection


def plot_planes_3d(
    ax: Axes3D,
    embeddings: Affine,
    *,
    xlim=(-1, 1),
    ylim=(-1, 1),
    colors: Sequence[Any] = ('C0', 'C1', 'C2'),
    alpha: float = 0.3,
    tiles: int = 6,
    linewidth: float = 0.5,
    **kwargs,
) -> Poly3DCollection:
    """Draw planes and their borders, occluding each other correctly.

    `plane_pieces` and `plane_border_pieces`, drawn by `plot_pieces_3d`. To
    sort other things, e.g. a trajectory, together with the planes, pass
    their pieces to one `plot_pieces_3d` call instead.
    """
    options = dict[str, Any](
        xlim=xlim, ylim=ylim, colors=colors, alpha=alpha, tiles=tiles
    )
    return plot_pieces_3d(
        ax,
        plane_pieces(embeddings, **options),
        plane_border_pieces(embeddings, **options),
        linewidth=linewidth,
        **kwargs,
    )


def _concatenate(parts: list[Pieces3D]) -> Pieces3D:
    if not parts:
        return Pieces3D([], np.zeros((0, 4)), np.zeros((0, 4)))
    return Pieces3D(
        [vertices for part in parts for vertices in part.vertices],
        np.concatenate([part.facecolors for part in parts]).reshape(-1, 4),
        np.concatenate([part.edgecolors for part in parts]).reshape(-1, 4),
    )


def _plane_tiles(embeddings: Affine, xlim, ylim, tiles: int) -> list[list[np.ndarray]]:
    """Tiles `(N, 3)` of each plane, cut along its intersections with the others."""
    xs = np.linspace(xlim[0], xlim[1], tiles + 1)
    ys = np.linspace(ylim[0], ylim[1], tiles + 1)
    grid = [
        np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        for x0, x1 in itertools.pairwise(xs)
        for y0, y1 in itertools.pairwise(ys)
    ]

    num_planes = embeddings.batch_shape[0]
    pieces = []
    for k in range(num_planes):
        embedding = embeddings.select(k)
        _check_map(embedding, input_dim=2, name='plane')
        plane_pieces = grid
        for j in range(num_planes):
            if j != k:
                plane_pieces = _split_along_intersection(
                    plane_pieces, embedding, embeddings.select(j)
                )
        pieces.append(
            [np.asarray(embedding.apply(jnp.asarray(piece))) for piece in plane_pieces]
        )
    return pieces


def _plane_border(embedding: Affine, xlim, ylim, tiles: int) -> list[np.ndarray]:
    """Border of `xlim × ylim` on a plane, as `(2, 3)` segments one tile long."""
    xs = np.linspace(xlim[0], xlim[1], tiles + 1)
    ys = np.linspace(ylim[0], ylim[1], tiles + 1)
    perimeter = np.concatenate(
        [
            np.stack([xs, np.full_like(xs, ylim[0])], axis=-1)[:-1],
            np.stack([np.full_like(ys, xlim[1]), ys], axis=-1)[:-1],
            np.stack([xs[::-1], np.full_like(xs, ylim[1])], axis=-1)[:-1],
            np.stack([np.full_like(ys, xlim[0]), ys[::-1]], axis=-1),
        ]
    )  # closed: (4 * tiles + 1, 2)
    points = np.asarray(embedding.apply(jnp.asarray(perimeter)))
    return [points[i : i + 2] for i in range(len(points) - 1)]


def _split_along_intersection(
    pieces: list[np.ndarray], embedding: Affine, other: Affine
) -> list[np.ndarray]:
    """Split convex polygons `(N, 2)` in the coordinates of `embedding` along its
    intersection line with `other`; parallel planes leave them whole."""
    try:
        line = planes.plane_intersection(embedding, other)
    except ValueError:  # parallel planes
        return pieces

    # the line in plane coordinates: z_0 + t e
    to_plane = embedding.pseudoinverse()
    ends = np.asarray(to_plane.apply(line.apply(jnp.array([[0.0], [1.0]]))))
    point, direction = ends[0], ends[1] - ends[0]
    return [
        part for piece in pieces for part in _split_polygon(piece, point, direction)
    ]


def _split_polygon(polygon: np.ndarray, point: np.ndarray, direction: np.ndarray):
    """The parts of a convex polygon `(N, 2)` on either side of a line."""
    offsets = polygon - point
    side = direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]
    tolerance = 1e-6 * np.abs(direction).max() * np.abs(offsets).max()
    side = np.where(np.abs(side) <= tolerance, 0.0, side)
    if np.all(side >= 0) or np.all(side <= 0):
        return [polygon]

    left, right = [], []
    for k in range(len(polygon)):
        here, there = polygon[k], polygon[(k + 1) % len(polygon)]
        side_here, side_there = side[k], side[(k + 1) % len(polygon)]
        if side_here >= 0:
            left.append(here)
        if side_here <= 0:
            right.append(here)
        if side_here * side_there < 0:  # the edge crosses the line
            crossing = here + (there - here) * side_here / (side_here - side_there)
            left.append(crossing)
            right.append(crossing)
    return [np.array(left), np.array(right)]


def plot_line_3d(ax: Axes3D, line: Affine, *, tlim=(-1, 1), **kwargs):
    r"""Draw the segment `tlim` of a line `(1,) -> (3,)`, $t \mapsto p + t\,d$.

    E.g. `planes.plane_intersection`. `kwargs` go to `ax.plot`.
    """
    _check_map(line, input_dim=1, name='line')
    points = np.asarray(line.apply(jnp.array([[tlim[0]], [tlim[1]]])))  # (2, 3)
    return ax.plot(*points.T, **kwargs)


def multiview_axes(
    *,
    lim=(-3, 3),
    frame_names: Sequence[str] = LATENT_NAMES,
    plane_names: Sequence[str] | None = None,
    plane_colors: Sequence[Any] | None = None,
    ticks: bool = False,
    figsize=(plot.SINGLE_COLUMN, plot.SINGLE_COLUMN),
    suptitle: str = '',
) -> tuple[Axes3D, tuple[Axes, Axes, Axes]]:
    """A 3D axes and three 2D panels on the planes of a frame; return both.

    Panels come in the order of `planes.frame_planes`, $(q_1, q_2)$,
    $(q_1, q_3)$, $(q_2, q_3)$, labelled by `frame_names` of $q_1, q_2, q_3$
    and titled by `plane_names`; their frames are drawn in `plane_colors`. All
    axes cover the symmetric range `lim` with equal aspect; the multiview
    plotting functions keep these limits and labels. Without `ticks`, no axes
    has ticks or tick labels, nor the 3D axes a grid (`hide_ticks_3d`).
    """
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(2, 2)
    top = fig.add_subplot(grid[0, 0])
    front = fig.add_subplot(grid[1, 0], sharex=top)
    side = fig.add_subplot(grid[1, 1], sharey=front)
    ax3d = fig.add_subplot(grid[0, 1], projection='3d')
    assert isinstance(ax3d, Axes3D)
    axs = (top, front, side)

    for k, (ax, (i, j)) in enumerate(zip(axs, FRAME_PAIRS)):
        ax.set(
            xlim=lim,
            ylim=lim,
            aspect='equal',
            xlabel=frame_names[i],
            ylabel=frame_names[j],
        )
        plot.set_spines(ax)  # a closed frame, coloured by the plane
        color = None if plane_colors is None else plane_colors[k]
        if plane_names is not None:
            ax.set_title(plane_names[k])
        if color is not None:
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
        if not ticks:
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    ax3d.set(
        xlim=lim,
        ylim=lim,
        zlim=lim,
        xlabel=LATENT_NAMES[0],
        ylabel=LATENT_NAMES[1],
        zlabel=LATENT_NAMES[2],
    )
    ax3d.set_aspect('equal')
    if not ticks:
        hide_ticks_3d(ax3d)
    fig.suptitle(suptitle)
    return ax3d, axs


def plot_traces_multiview(
    ax3d: Axes3D,
    axs: Sequence[Axes],
    traces,
    *,
    frame=None,
    mark_endpoints=True,
    **kwargs,
) -> None:
    """Draw (T, 3) or (B, T, 3) traces in 3D and projected onto the frame planes.

    `plot_traces_3d` on `ax3d` and `plot_traces_panels` on `axs`; `kwargs`
    go to both.
    """
    with _preserved_view(ax3d):
        plot_traces_3d(ax3d, traces, mark_endpoints=mark_endpoints, **kwargs)
    plot_traces_panels(
        axs, traces, frame=frame, mark_endpoints=mark_endpoints, **kwargs
    )


def plot_traces_panels(
    axs: Sequence[Axes],
    traces,
    *,
    frame=None,
    mark_endpoints=True,
    **kwargs,
) -> None:
    r"""Draw (T, 3) or (B, T, 3) traces projected onto the frame planes.

    `frame` `(3, 3)` defaults to the identity. The projections are the plane
    coordinates $z = [q_i, q_j]^\top x$ (`planes.plane_coordinates`), drawn
    with `plot.plot_traces_2d` (`kwargs`). Use it alone when the 3D traces are
    drawn otherwise, e.g. as `trace_pieces` sorted together with planes.
    """
    values = jnp.asarray(plot._trace_batch(traces, 3))
    embeddings = planes.frame_planes(jnp.eye(3) if frame is None else frame)
    for k, ax in enumerate(axs):
        coordinates = planes.plane_coordinates(embeddings.select(k), values)
        with _preserved_view(ax):
            plot.plot_traces_2d(
                ax, np.asarray(coordinates), mark_endpoints=mark_endpoints, **kwargs
            )


def plot_dyn_linear_multiview(
    axs: Sequence[Axes],
    fields: Affine,
    *,
    duration: float | None = None,
    panel_kwargs: Sequence[Mapping[str, Any]] | None = None,
    **kwargs,
) -> list:
    """Draw a batch `(3,)` of `(2,) -> (2,)` maps, one per panel; return the artists.

    `fields` are dynamics in the coordinates of each frame plane, e.g.
    `planes.slice_dynamics(dynamics, planes.frame_planes(frame))` or
    `planes.lift_dynamics`.
    Panels are drawn with `plot.plot_dyn_linear` over their current limits;
    `kwargs` are shared and `panel_kwargs` (one dictionary per panel) override
    them. For trails (the default mode), panels share `duration`, by default
    `trail.shared_duration_affine` of all panels. Non-finite maps are skipped.
    """
    if fields.batch_shape != (len(axs),):
        raise ValueError(
            f'expected fields with batch ({len(axs)},); got {fields.batch_shape}'
        )
    if panel_kwargs is not None and len(panel_kwargs) != len(axs):
        raise ValueError(f'expected {len(axs)} panel_kwargs; got {len(panel_kwargs)}')

    options = [
        dict[str, Any](xlim=ax.get_xlim(), ylim=ax.get_ylim())
        | kwargs
        | ({} if panel_kwargs is None else dict(panel_kwargs[k]))
        for k, ax in enumerate(axs)
    ]
    finite = [
        k
        for k in range(len(axs))
        if np.all(np.isfinite(fields.coefficients[k]))
        and np.all(np.isfinite(fields.bias[k]))
    ]

    if kwargs.get('mode', plot.DEFAULT_DYN_MODE) == 'trails':
        if duration is None:
            duration = trail.shared_duration_affine(
                [
                    (fields.select(k), options[k]['xlim'], options[k]['ylim'])
                    for k in finite
                ],
                **{key: kwargs[key] for key in ('length', 'quantile') if key in kwargs},
            )
        for k in finite:
            options[k].setdefault('duration', duration)

    artists = []
    for k in finite:
        with _preserved_view(axs[k]):
            artists.append(plot.plot_dyn_linear(axs[k], fields.select(k), **options[k]))
    return artists


@contextlib.contextmanager
def _preserved_view(ax) -> Iterator[None]:
    """Restore the limits, labels and aspect of `ax` (2D or 3D) after drawing."""
    names = ('x', 'y', 'z') if isinstance(ax, Axes3D) else ('x', 'y')
    saved = {
        **{f'{n}lim': getattr(ax, f'get_{n}lim')() for n in names},
        **{f'{n}label': getattr(ax, f'get_{n}label')() for n in names},
    }
    aspect = ax.get_aspect()
    yield
    ax.set(**saved)
    ax.set_aspect(aspect)


def _check_map(affine: Affine, *, input_dim: int, name: str) -> None:
    if (
        affine.batch_shape
        or affine.input_shape != (input_dim,)
        or affine.output_dim != 3
    ):
        raise ValueError(
            f'expected an unbatched {name} ({input_dim},) -> (3,); got batch '
            f'{affine.batch_shape}, {affine.input_shape} -> ({affine.output_dim},)'
        )
