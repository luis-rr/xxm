r"""Trails: short pieces of 2D affine dynamics, drawn to show flow strength.

A trail follows a seed point $z$ under a flow $\varphi_s$ of the map
$f(z) = Cz + d$ over a short range of times $s$, measured in steps. Seeds sit
on a (jittered) grid, every trail of a figure shares one duration, so trail
length shows the strength of the motion and curvature its rotation. Trails
fade in from transparent (earliest) to solid (latest) and end in a head.

Two interpolations both pass exactly through the integer iterates,
$\varphi_n = f^n$, with negative $n$ the inverse map:

- `'log'`, for diagonalizable $C = V \operatorname{diag}(\mu) V^{-1}$:
  $\varphi_s(z) = C^s z + g_s(C)\, d$ with $g_s(\mu) = (\mu^s - 1)/(\mu - 1)$,
  $g_s(1) = s$ and $\mu^s = e^{s \log \mu}$ (principal log). It is one natural
  interpolation between iterates, not the unique continuous trajectory of a
  discrete map; it needs no fixed point and satisfies
  $\varphi_s \circ \varphi_t = \varphi_{s+t}$.
- `'discrete'`: the integer iterates only.

`'log'` falls back to `'discrete'` when an eigenvalue is real and
non-positive, when the eigenvectors are poorly conditioned, or when the
interpolated maps are not real to tolerance: limitations of the principal
log, not proof that no real interpolation exists. Strength is measured by
the one-step displacement $\|f(z) - z\|$ for both.

The module has three layers:

- data: `Grid` (seed positions), `jitter`, `AffineFlow` (positions, speeds,
  trail times) and `Trails` (clipping, segments, lengths, heads, iterates);
- settings: heuristic estimators (`estimate_*`, `panel_speeds`,
  `shared_duration`, `shared_duration_affine`) that `TrailLayout.resolve` combines with explicit
  overrides into one concrete layout;
- drawing: `plot_trail_lines`, `plot_trail_iterates`, `plot_trail_heads`.

`plot_dyn_trails` composes them for one panel.
"""

import dataclasses
from typing import Any, Literal, NamedTuple, Self

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import (
    Collection,
    LineCollection,
    PathCollection,
    PolyCollection,
)
from matplotlib.colors import to_rgba
from matplotlib.transforms import Affine2D

from xxm.core.affine import Affine

Interpolation = Literal['log', 'discrete']

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Grid:
    """Regular grid of `num_points` per axis over the box `xlim × ylim`.

    With `cell_centred`, the box is split into `num_points × num_points` equal
    cells and each point is a cell centre, so the outermost points lie half a
    spacing inside the border; otherwise the outermost points lie on it.
    """

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    num_points: int
    cell_centred: bool = True

    def __post_init__(self) -> None:
        _check_box(self.xlim, self.ylim)
        minimum = 1 if self.cell_centred else 2
        if self.num_points < minimum:
            raise ValueError(
                f'num_points must be at least {minimum}; got {self.num_points}'
            )

    @property
    def spacing(self) -> np.ndarray:
        """Distance `(2,)` between neighbouring points along each axis."""
        extent = np.array(
            [self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]], dtype=float
        )
        return extent / (self.num_points if self.cell_centred else self.num_points - 1)

    @property
    def cell(self) -> float:
        """Smaller spacing: the unit of trail lengths measured in spacings."""
        return float(np.min(self.spacing))

    @property
    def extent(self) -> float:
        """Smaller side of the box: the size of a square panel, in data units."""
        return float(min(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]))

    def points(self, margin: int = 0) -> np.ndarray:
        """Grid points `(N, 2)`, with `margin` extra rows and columns outside the box.

        The margin points continue the grid at the same spacing on each side:
        `num_points + 2 * margin` points per axis.
        """
        if margin < 0:
            raise ValueError(f'margin must be non-negative; got {margin}')
        offset = 0.5 if self.cell_centred else 0.0
        steps = np.arange(-margin, self.num_points + margin) + offset
        spacing = self.spacing
        xx, yy = np.meshgrid(
            self.xlim[0] + steps * spacing[0], self.ylim[0] + steps * spacing[1]
        )
        return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def jitter(points, scale, *, seed: int = 0) -> np.ndarray:
    """Points `(N, 2)` moved uniformly within ±`scale`/2 (scalar or per axis)."""
    points = np.asarray(points, dtype=float)
    noise = np.random.default_rng(seed).random(points.shape) - 0.5
    return points + noise * np.asarray(scale, dtype=float)


class Trails(NamedTuple):
    """Trails `points (N, M, 2)` at increasing `times (M,)`; NaN where clipped.

    `interpolation` is that of the flow they came from (`AffineFlow.trails`).
    """

    points: np.ndarray
    times: np.ndarray
    interpolation: Interpolation

    def select(self, mask) -> Self:
        """The trails selected by a boolean `(N,)` mask or index array."""
        return self._replace(points=np.asarray(self.points)[mask])

    def clip(self, xlim, ylim, *, margin: float = 0.05) -> Self:
        """Samples outside the box, padded by `margin` of its size, set to NaN."""
        _check_box(xlim, ylim)
        pad = margin * np.array([xlim[1] - xlim[0], ylim[1] - ylim[0]])
        low = np.array([xlim[0], ylim[0]]) - pad
        high = np.array([xlim[1], ylim[1]]) + pad
        with np.errstate(invalid='ignore'):
            inside = np.all((self.points >= low) & (self.points <= high), axis=-1)
        return self._replace(points=np.where(inside[..., None], self.points, np.nan))

    def segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Consecutive-sample segments `(K, 2, 2)` with both ends finite.

        Also returns each segment's progress `(K,)` along its trail, at its
        later end: from `1 / (M - 1)` to 1.
        """
        points = _checked_points(self.points)
        progress = np.linspace(0.0, 1.0, points.shape[1])[1:]
        segments = np.stack([points[:, :-1], points[:, 1:]], axis=2)  # (N, M-1, 2, 2)
        valid = np.all(np.isfinite(segments), axis=(-2, -1))  # (N, M-1)
        return segments[valid], np.broadcast_to(progress, valid.shape)[valid]

    def lengths(self) -> np.ndarray:
        """Length `(N,)` of each trail along its finite segments, in data units."""
        points = _checked_points(self.points)
        steps = np.linalg.norm(np.diff(points, axis=1), axis=-1)  # (N, M-1)
        return np.where(np.isfinite(steps), steps, 0.0).sum(axis=1)

    def heads(self) -> tuple[np.ndarray, np.ndarray]:
        """Latest finite point of each trail and the unit direction of its last step.

        Returns positions `(K, 2)` for the `K` trails with a finite point, and
        directions `(K, 2)`, NaN where the trail has fewer than two
        consecutive finite samples or does not move.
        """
        points = _checked_points(self.points)
        finite = np.all(np.isfinite(points), axis=-1)  # (N, M)
        has_finite = np.any(finite, axis=1)
        rows = np.arange(len(points))
        last = _latest_finite_index(finite)
        previous = np.maximum(last - 1, 0)

        step = points[rows, last] - points[rows, previous]
        norm = np.linalg.norm(step, axis=-1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            directions = np.where((last > 0)[:, None] & (norm > 0), step / norm, np.nan)
        return points[rows, last][has_finite], directions[has_finite]

    def iterates(
        self, *, latest: bool = True, atol: float = 1e-9
    ) -> tuple[np.ndarray, np.ndarray]:
        """Finite samples at integer times, i.e. exact iterates of the map.

        Returns points `(K, 2)` and their progress `(K,)` along the trail, from
        0 (earliest time) to 1 (latest). All samples of a `'discrete'` trail
        are iterates; `'log'` trails have them where `times` hit integers.
        With `latest=False`, each trail's latest finite sample (where `heads`
        puts its head) is left out.
        """
        points = _checked_points(self.points)
        times = np.asarray(self.times, dtype=float)
        span = times[-1] - times[0]
        progress = (times - times[0]) / span if span > 0 else np.ones_like(times)

        finite = np.all(np.isfinite(points), axis=-1)  # (N, M)
        keep = finite & (np.abs(times - np.rint(times)) <= atol)
        if not latest:
            keep[np.arange(len(points)), _latest_finite_index(finite)] = False
        return points[keep], np.broadcast_to(progress, keep.shape)[keep]


@dataclasses.dataclass(frozen=True, eq=False)
class AffineFlow:
    r"""Flow $\varphi_s$ of a 2D affine map $f(z) = Cz + d$ (module docstring).

    Build with `from_affine`. `interpolation` is the effective method after
    any fallback. A map with non-finite entries gives NaN positions.
    """

    coefficients: np.ndarray  # (2, 2)
    bias: np.ndarray  # (2,)
    interpolation: Interpolation
    invertible: bool
    eigenvalues: np.ndarray | None = None  # (2,) complex, 'log' only
    eigenvectors: np.ndarray | None = None  # (2, 2) complex, 'log' only
    imag_tol: float = 1e-6

    @classmethod
    def from_affine(
        cls,
        affine: Affine | tuple[Any, Any],
        *,
        interpolation: Interpolation = 'log',
        cond_max: float = 1e6,
        imag_tol: float = 1e-6,
    ) -> Self:
        r"""Flow of an unbatched `(2,) -> (2,)` map, or of `(coefficients, bias)`.

        Negative times need $\operatorname{cond}(C) \le$ `cond_max`; the same
        bound applies to the eigenvectors for `'log'`.
        """
        if interpolation not in ('log', 'discrete'):
            raise ValueError(
                f"interpolation must be 'log' or 'discrete'; got {interpolation!r}"
            )
        if isinstance(affine, Affine):
            if affine.batch_shape:
                raise ValueError(f'expected an unbatched map; got {affine.batch_shape}')
            coefficients, bias = affine.coefficients_flat, affine.bias
        else:
            coefficients, bias = affine
        coefficients = np.asarray(coefficients, dtype=float)
        bias = np.asarray(bias, dtype=float)
        if coefficients.shape != (2, 2) or bias.shape != (2,):
            raise ValueError(
                'expected a (2,) -> (2,) map; got coefficients '
                f'{coefficients.shape} and bias {bias.shape}'
            )

        finite = bool(np.isfinite(coefficients).all() and np.isfinite(bias).all())
        if not finite:
            return cls(coefficients, bias, 'discrete', invertible=False)

        invertible = bool(np.linalg.cond(coefficients) <= cond_max)
        discrete = cls(coefficients, bias, 'discrete', invertible=invertible)
        if interpolation == 'discrete':
            return discrete

        eigenvalues, eigenvectors = np.linalg.eig(coefficients)
        eigenvalues = eigenvalues.astype(complex)
        eigenvectors = eigenvectors.astype(complex)
        real = np.abs(np.imag(eigenvalues)) <= imag_tol * np.maximum(
            np.abs(eigenvalues), 1.0
        )
        if np.any(real & (np.real(eigenvalues) <= 0)):
            return discrete
        if np.linalg.cond(eigenvectors) > cond_max:
            return discrete

        flow = cls(
            coefficients,
            bias,
            'log',
            invertible=invertible,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            imag_tol=imag_tol,
        )
        probe = np.array([-0.5, 0.5]) if invertible else np.array([0.5])
        if not flow._is_real(probe):
            return discrete
        return flow

    @property
    def rotation(self) -> float:
        """Largest |arg μ| per step, in radians; 0 for `'discrete'`."""
        if self.eigenvalues is None:
            return 0.0
        return float(np.abs(np.angle(self.eigenvalues)).max())

    def step(self, points) -> np.ndarray:
        """One step $f(z)$ of points `(..., 2)`."""
        return np.asarray(points, dtype=float) @ self.coefficients.T + self.bias

    def positions(self, points, times) -> np.ndarray:
        r"""$\varphi_s(z)$ for points `(N, 2)` and times `(M,)`, shape `(N, M, 2)`.

        `'discrete'` flows accept integer times only. Negative times need an
        invertible map.
        """
        points = np.asarray(points, dtype=float)
        times = np.asarray(times, dtype=float)
        if points.ndim != 2 or points.shape[-1] != 2 or times.ndim != 1:
            raise ValueError(
                f'expected points (N, 2) and times (M,); got {points.shape}, {times.shape}'
            )
        if not np.isfinite(self.coefficients).all() or not np.isfinite(self.bias).all():
            return np.full((len(points), len(times), 2), np.nan)
        if np.any(times < 0) and not self.invertible:
            raise ValueError(
                'negative times need an invertible map; run forward instead'
            )
        if self.interpolation == 'log':
            return self._log_positions(points, times)
        return self._discrete_positions(points, times)

    def _log_positions(self, points: np.ndarray, times: np.ndarray) -> np.ndarray:
        powers, gains = self._log_maps(times)
        real_powers = self._real(powers)  # (M, 2, 2)
        real_gains = self._real(gains)  # (M, 2, 2)
        moved = np.einsum('mij,nj->nmi', real_powers, points)
        return moved + np.einsum('mij,j->mi', real_gains, self.bias)[None]

    def _log_maps(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Complex $C^s$ and $g_s(C)$ for each time, shapes `(M, 2, 2)`."""
        assert self.eigenvalues is not None and self.eigenvectors is not None
        logs = np.log(self.eigenvalues)  # principal branch, (2,)
        exponents = np.outer(times, logs)  # (M, 2)
        unit = logs == 0
        gains = np.where(
            unit,
            times[:, None],
            _expm1(exponents) / np.where(unit, 1.0, _expm1(logs)),
        )
        vectors = self.eigenvectors
        inverse = np.linalg.inv(vectors)

        def apply(values):  # V diag(values) V^{-1}
            return np.einsum('ij,mj,jk->mik', vectors, values, inverse)

        return apply(np.exp(exponents)), apply(gains)

    def _real(self, values: np.ndarray) -> np.ndarray:
        scale = 1.0 + np.abs(values).max()
        if np.abs(np.imag(values)).max() > self.imag_tol * scale:
            raise RuntimeError('interpolated maps are not real; use discrete')
        return np.real(values)

    def _is_real(self, times: np.ndarray) -> bool:
        return all(
            np.abs(np.imag(m)).max() <= self.imag_tol * (1.0 + np.abs(m).max())
            for m in self._log_maps(times)
        )

    def _discrete_positions(self, points: np.ndarray, times: np.ndarray) -> np.ndarray:
        steps = np.rint(times).astype(int)
        if not np.allclose(steps, times):
            raise ValueError('discrete flows accept integer times only')

        orbit = {0: points}
        current = points
        for k in range(1, max(int(steps.max()), 0) + 1):
            current = self.step(current)
            orbit[k] = current
        current = points
        for k in range(-1, min(int(steps.min()), 0) - 1, -1):
            current = np.linalg.solve(self.coefficients, (current - self.bias).T).T
            orbit[k] = current
        return np.stack([orbit[int(k)] for k in steps], axis=1)

    def speeds(self, points) -> np.ndarray:
        r"""One-step displacements $\|f(z) - z\|$ of points `(N, 2)`, shape `(N,)`."""
        points = np.asarray(points, dtype=float)
        return np.linalg.norm(self.step(points) - points, axis=-1)

    def trail_times(
        self, duration: float, num_samples: int = 16, *, seed_position: float = 1.0
    ) -> np.ndarray:
        r"""Increasing times `(M,)` of trails `duration` steps long, seeds at 0.

        `seed_position` $p \in [0, 1]$ is where the seed lies along its trail:
        the trail runs backward from the seed for $p\,\tau$ steps and forward
        for $(1 - p)\,\tau$, i.e. over $s \in [-p\tau, (1 - p)\tau]$. So 0 puts
        the seed at the start, 0.5 in the middle and 1 at the end. Since
        $\varphi_s$ is defined for both signs, one evaluation over this range
        is the backward and forward runs concatenated. `'discrete'` flows get
        the integers covering the range, with the duration rounded to at least
        one step, and ignore `num_samples`.
        """
        if duration <= 0:
            raise ValueError(f'duration must be positive; got {duration}')
        if not 0.0 <= seed_position <= 1.0:
            raise ValueError(f'seed_position must be in [0, 1]; got {seed_position}')

        if self.interpolation == 'discrete':
            count = max(1, round(duration))
            start = -round(seed_position * count)
            return np.arange(start, start + count + 1, dtype=float)

        if num_samples < 2:
            raise ValueError(f'num_samples must be at least 2; got {num_samples}')
        start = -seed_position * duration
        return np.linspace(start, start + duration, num_samples)

    def trails(self, points, times) -> Trails:
        """Trails of seed `points` `(N, 2)` at strictly increasing `times` `(M,)`."""
        times = np.asarray(times, dtype=float)
        if np.any(np.diff(times) <= 0):
            raise ValueError('times must be strictly increasing')
        return Trails(self.positions(points, times), times, self.interpolation)


# ---------------------------------------------------------------------------
# Settings: heuristic estimates, resolved into one concrete layout
# ---------------------------------------------------------------------------

DEFAULT_SEED_SPACING = 12.0  # points between neighbouring seeds
DEFAULT_LENGTH = 0.2  # trail length, as a fraction of the panel size
DEFAULT_QUANTILE = 0.5  # ... reached at this quantile of the speeds
DEFAULT_MIN_LENGTH = 1.0  # shortest trail drawn, in points
DEFAULT_HEAD_SIZE = 5.0  # arrowhead length, in points
_PROBE_POINTS = 16  # probe grid per axis for speed statistics


def axes_size_points(ax: Axes) -> float:
    """Side of a square panel drawn in `ax`, in points: its smaller dimension.

    Measured from the current axes position, which layout engines (e.g.
    constrained layout) may still adjust at draw time.
    """
    extent = ax.get_window_extent()
    return float(min(extent.width, extent.height)) * 72 / ax.figure.dpi


def panel_speeds(flow: AffineFlow, xlim, ylim) -> np.ndarray:
    """One-step speeds on a probe grid, in panel sizes per step, shape `(K,)`.

    The panel size is the smaller side of `xlim × ylim`. In these units speeds
    from panels with different limits are comparable: for a linear field
    centred near its fixed point, they don't depend on the limits.
    """
    probes = Grid(xlim, ylim, _PROBE_POINTS)
    return flow.speeds(probes.points()) / probes.extent


def estimate_duration(
    speeds,
    *,
    length: float = DEFAULT_LENGTH,
    quantile: float = DEFAULT_QUANTILE,
) -> float:
    """Duration in steps making trails at the `quantile` speed `length` long.

    `length` is in the units of `speeds` per step: a fraction of the panel
    size for `panel_speeds`. Returns 1 when no speed is finite and positive.
    """
    values = np.asarray(speeds, dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return 1.0
    return float(length / np.quantile(values, quantile))


def shared_duration(
    panels,
    *,
    length: float = DEFAULT_LENGTH,
    quantile: float = DEFAULT_QUANTILE,
) -> float:
    """One duration for several panels, from their pooled `panel_speeds`.

    `panels` holds `(flow, xlim, ylim)` triples. Passing the result as
    `duration` to each panel makes trail lengths comparable across them.
    """
    pooled = [panel_speeds(flow, xlim, ylim) for flow, xlim, ylim in panels]
    if not pooled:
        return 1.0
    return estimate_duration(np.concatenate(pooled), length=length, quantile=quantile)


def shared_duration_affine(
    panels,
    *,
    length: float = DEFAULT_LENGTH,
    quantile: float = DEFAULT_QUANTILE,
) -> float:
    """`shared_duration` of panels given by their maps rather than their flows.

    `panels` holds `(affine, xlim, ylim)` triples of unbatched `(2,) -> (2,)`
    maps (`AffineFlow.from_affine`); maps with non-finite entries are skipped.
    """
    return shared_duration(
        [
            (AffineFlow.from_affine(affine), xlim, ylim)
            for affine, xlim, ylim in panels
            if np.all(np.isfinite(affine.coefficients))
            and np.all(np.isfinite(affine.bias))
        ],
        length=length,
        quantile=quantile,
    )


def estimate_num_points(
    panel_size: float, *, seed_spacing: float = DEFAULT_SEED_SPACING
) -> int:
    """Seeds per axis for one seed every `seed_spacing` points (at least 2)."""
    return max(2, round(panel_size / seed_spacing))


def estimate_seed_margin(flow: AffineFlow, grid: Grid, *, duration: float) -> int:
    """Seed rows and columns outside the panel: the longest trail, in spacings.

    Trails seeded that far outside can still reach the panel. Measured on the
    seeds inside the panel, rounded up and capped at `grid.num_points`.
    """
    longest = duration * _finite_max(flow.speeds(grid.points())) / grid.cell
    return min(int(np.ceil(longest)), grid.num_points)


def estimate_num_samples(
    flow: AffineFlow,
    grid: Grid,
    *,
    duration: float,
    seed_margin: int,
    per_spacing: float = 4.0,
    per_radian: float = 4.0,
    bounds: tuple[int, int] = (4, 64),
) -> int:
    """Samples per trail for smooth curves.

    `per_spacing` samples per seed spacing along the longest trail (seeds
    outside the panel included) and `per_radian` per radian of rotation over
    `duration`, clipped to `bounds`.
    """
    seeds = grid.points(seed_margin)
    longest = duration * _finite_max(flow.speeds(seeds)) / grid.cell
    needed = max(per_spacing * longest, per_radian * flow.rotation * duration) + 1
    return int(np.clip(np.ceil(needed), *bounds))


def estimate_min_length(
    grid: Grid,
    *,
    panel_size: float,
    min_length_points: float = DEFAULT_MIN_LENGTH,
) -> float:
    """Shortest trail drawn, in data units: `min_length_points` on the panel.

    A trail that barely moves (e.g. at a fixed point) shows no tail and its
    head no meaningful direction, so it is skipped with its head.
    """
    return min_length_points / panel_size * grid.extent


class TrailLayout(NamedTuple):
    """Concrete trail settings for one panel; every field is final.

    `num_points` seeds per axis inside the panel, `duration` in steps,
    `seed_margin` extra seed rows and columns outside the panel on each side,
    `num_samples` per trail, and `min_length` in data units: shorter trails
    are skipped. Build with `resolve`.
    """

    num_points: int
    duration: float
    seed_margin: int
    num_samples: int
    min_length: float

    @classmethod
    def resolve(
        cls,
        flow: AffineFlow,
        xlim,
        ylim,
        *,
        panel_size: float,
        seed_spacing: float = DEFAULT_SEED_SPACING,
        length: float = DEFAULT_LENGTH,
        quantile: float = DEFAULT_QUANTILE,
        min_length_points: float = DEFAULT_MIN_LENGTH,
        cell_centred: bool = True,
        num_points: int | None = None,
        duration: float | None = None,
        seed_margin: int | None = None,
        num_samples: int | None = None,
        min_length: float | None = None,
    ) -> Self:
        """Concrete layout of one panel, estimating every field left as None.

        `panel_size` is the panel's side in points (`axes_size_points`). The
        other base settings feed the estimators: `seed_spacing` sets
        `num_points`, `length` and `quantile` set `duration`, and
        `min_length_points` sets `min_length`. Fields are resolved in
        dependency order, so an override (e.g. a shared `duration`) is used by
        the estimates that depend on it.
        """
        if num_points is None:
            num_points = estimate_num_points(panel_size, seed_spacing=seed_spacing)
        grid = Grid(xlim, ylim, num_points, cell_centred=cell_centred)
        if duration is None:
            duration = estimate_duration(
                panel_speeds(flow, xlim, ylim), length=length, quantile=quantile
            )
        if seed_margin is None:
            seed_margin = estimate_seed_margin(flow, grid, duration=duration)
        if num_samples is None:
            num_samples = estimate_num_samples(
                flow, grid, duration=duration, seed_margin=seed_margin
            )
        if min_length is None:
            min_length = estimate_min_length(
                grid, panel_size=panel_size, min_length_points=min_length_points
            )
        return cls(num_points, duration, seed_margin, num_samples, min_length)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# arrowhead outline in points, tip at the origin pointing along +x
_ARROWHEAD = np.array([[0.0, 0.0], [-1.0, 0.45], [-0.65, 0.0], [-1.0, -0.45]])


def plot_trail_lines(
    ax: Axes,
    trails: Trails,
    *,
    color: Any = 'k',
    linewidth: float = 0.8,
    fade: float = 1.0,
    taper: bool = True,
    zorder: float = 2.0,
) -> LineCollection:
    """Trail lines as one `LineCollection`, leaving the axis limits unchanged.

    `'log'` trails fade in from transparent (`_plot_faded_lines`);
    `'discrete'` trails are faint straight connectors between iterates, which
    are not trajectories (`_plot_connector_lines`). Segments with a NaN end
    are dropped.
    """
    segments, progress = trails.segments()
    if trails.interpolation == 'log':
        lines = _plot_faded_lines(
            segments,
            progress,
            color=color,
            linewidth=linewidth,
            fade=fade,
            taper=taper,
            zorder=zorder,
        )
    else:
        lines = _plot_connector_lines(
            segments, color=color, linewidth=linewidth, zorder=zorder
        )
    ax.add_collection(lines, autolim=False)
    return lines


def plot_trail_iterates(
    ax: Axes,
    trails: Trails,
    *,
    color: Any = 'k',
    size: float = 3.0,
    fade: float = 1.0,
    latest: bool = True,
    zorder: float = 2.0,
) -> PathCollection | None:
    """Dots of diameter `size` (points) at the exact iterates of the trails.

    Alpha rises from 0.2 at the earliest iterate to 1 at the latest.
    `latest=False` leaves out each trail's latest point, e.g. under a head.
    Returns None when there are none.
    """
    points, progress = trails.iterates(latest=latest)
    colors = _faded_colors(color, progress, fade=fade, floor=0.2)
    return _scatter(ax, points, colors=colors, size=size, zorder=zorder)


def plot_trail_heads(
    ax: Axes,
    trails: Trails,
    *,
    style: Literal['arrow', 'dot'] = 'arrow',
    color: Any = 'k',
    size: float = DEFAULT_HEAD_SIZE,
    overshoot: float = 0.0,
    zorder: float = 2.0,
) -> Collection | None:
    """A marker of `size` points at the latest point of each trail.

    `'arrow'` draws arrowheads with their tip on the point, along the trail's
    last step (`Trails.heads`); trails without a direction get none. The
    orientation assumes equal data aspect, as set by `plot_dyn_trails`.
    `overshoot` (points) moves arrowheads forward along the trail, so a
    flat-ended line under them does not blunt the tip: about 1.2 line widths
    suffices for `_ARROWHEAD`. `'dot'` draws dots of diameter `size`. Returns
    None when nothing is drawn.
    """
    positions, directions = trails.heads()
    if style == 'arrow':
        return _plot_arrowheads(
            ax,
            positions,
            directions,
            color=color,
            size=size,
            overshoot=overshoot,
            zorder=zorder,
        )
    if style == 'dot':
        colors = np.tile(np.asarray(to_rgba(color)), (len(positions), 1))
        return _scatter(ax, positions, colors=colors, size=size, zorder=zorder)
    raise ValueError(f"style must be 'arrow' or 'dot'; got {style!r}")


def plot_dyn_trails(
    ax: Axes,
    affine: Affine,
    *,
    xlim=(-3, 3),
    ylim=(-3, 3),
    layout: TrailLayout | None = None,
    panel_size: float | None = None,
    seed_spacing: float = DEFAULT_SEED_SPACING,
    length: float = DEFAULT_LENGTH,
    quantile: float = DEFAULT_QUANTILE,
    min_length_points: float = DEFAULT_MIN_LENGTH,
    num_points: int | None = None,
    duration: float | None = None,
    seed_margin: int | None = None,
    num_samples: int | None = None,
    min_length: float | None = None,
    cell_centred: bool = True,
    jitter_fraction: float = 0.3,
    seed: int = 0,
    seed_position: float = 1.0,
    interpolation: Interpolation = 'log',
    color: Any = 'k',
    linewidth: float = 0.8,
    fade: float = 1.0,
    taper: bool = True,
    head: Literal['arrow', 'dot'] | None = 'arrow',
    head_size: float = DEFAULT_HEAD_SIZE,
    head_overshoot: float | None = None,
    iterates: bool | None = None,
    iterate_size: float = 2.5,
    label: str | None = None,
    zorder: float = 2.0,
) -> list[Artist]:
    """Trails of one `(2,) -> (2,)` map over the box `xlim × ylim`.

    **Layout.** A given `layout` is used as is. Otherwise `TrailLayout.resolve`
    estimates it from `panel_size` (default: `axes_size_points(ax)`) and the
    base settings `seed_spacing`, `length`, `quantile` and
    `min_length_points`; the overrides `num_points`, `duration`,
    `seed_margin`, `num_samples` and `min_length` replace single estimates.
    Pass `duration` (e.g. from `shared_duration`) to compare trail lengths
    across panels.

    **Seeds** lie on a `Grid` (`cell_centred`, plus `seed_margin` rows and
    columns outside the box), jittered by `jitter_fraction` of the spacing,
    at `seed_position` along their trails (`AffineFlow.trail_times`; 1: at the
    ends). Trails shorter than `min_length` are skipped entirely, head
    included.

    **Drawing**: `plot_trail_lines` (labelled `label`), then
    `plot_trail_iterates` (by default only for `'discrete'` trails, without
    the points under heads) and `plot_trail_heads` (`head=None` for none;
    `head_overshoot` defaults to 1.2 line widths, keeping tips sharp). Heads
    come from the full trails: a head outside the axes is cut off like any
    artist, and its trail is still drawn where it crosses them. Sets the
    limits and equal aspect, leaving labels to the caller.
    """
    flow = AffineFlow.from_affine(affine, interpolation=interpolation)
    ax.set(xlim=xlim, ylim=ylim, aspect='equal')
    if layout is None:
        layout = TrailLayout.resolve(
            flow,
            xlim,
            ylim,
            panel_size=axes_size_points(ax) if panel_size is None else panel_size,
            seed_spacing=seed_spacing,
            length=length,
            quantile=quantile,
            min_length_points=min_length_points,
            cell_centred=cell_centred,
            num_points=num_points,
            duration=duration,
            seed_margin=seed_margin,
            num_samples=num_samples,
            min_length=min_length,
        )

    grid = Grid(xlim, ylim, layout.num_points, cell_centred=cell_centred)
    seeds = jitter(
        grid.points(layout.seed_margin), jitter_fraction * grid.spacing, seed=seed
    )
    times = flow.trail_times(
        layout.duration, layout.num_samples, seed_position=seed_position
    )
    trails = flow.trails(seeds, times)
    trails = trails.select(trails.lengths() >= layout.min_length)

    lines = plot_trail_lines(
        ax,
        # guard against far-away samples (backward runs expand); axes clip the rest
        trails.clip(xlim, ylim, margin=1.0),
        color=color,
        linewidth=linewidth,
        fade=fade,
        taper=taper,
        zorder=zorder,
    )
    if label is not None:
        lines.set_label(label)
    artists: list[Artist | None] = [lines]
    if iterates if iterates is not None else trails.interpolation == 'discrete':
        artists.append(
            plot_trail_iterates(
                ax,
                trails,
                color=color,
                size=iterate_size,
                fade=fade,
                latest=head is None,
                zorder=zorder,
            )
        )
    if head is not None:
        artists.append(
            plot_trail_heads(
                ax,
                trails,
                style=head,
                color=color,
                size=head_size,
                overshoot=1.2 * linewidth if head_overshoot is None else head_overshoot,
                zorder=zorder,
            )
        )
    return [artist for artist in artists if artist is not None]


def _plot_faded_lines(
    segments: np.ndarray,
    progress: np.ndarray,
    *,
    color: Any,
    linewidth: float,
    fade: float,
    taper: bool,
    zorder: float,
) -> LineCollection:
    """Segments whose alpha grows as `progress ** fade`, widths tapering up to `linewidth`."""
    widths = linewidth * (0.25 + 0.75 * progress) if taper else linewidth
    return LineCollection(
        list(segments),
        colors=_faded_colors(color, progress, fade=fade),
        linewidths=widths,
        capstyle='butt',
        zorder=zorder,
    )


def _plot_connector_lines(
    segments: np.ndarray, *, color: Any, linewidth: float, zorder: float
) -> LineCollection:
    """Faint, thin straight segments of uniform color."""
    rgba = to_rgba(color)
    return LineCollection(
        list(segments),
        colors=[(*rgba[:3], 0.3 * rgba[3])],
        linewidths=0.6 * linewidth,
        zorder=zorder,
    )


def _plot_arrowheads(
    ax: Axes,
    positions: np.ndarray,
    directions: np.ndarray,
    *,
    color: Any,
    size: float,
    overshoot: float,
    zorder: float,
) -> PolyCollection | None:
    """Arrowheads of `size` points along unit `directions`, tips `overshoot`
    points ahead of `positions`.

    Positions without a finite direction are skipped.
    """
    oriented = np.all(np.isfinite(directions), axis=-1)
    if not oriented.any():
        return None
    heads = PolyCollection(
        list(_arrowhead_outlines(directions[oriented], size, overshoot)),
        offsets=positions[oriented],
        offset_transform=ax.transData,
        facecolors=[to_rgba(color)],
        edgecolors='none',
        zorder=zorder,
    )
    # outlines are in points, robust to the figure dpi
    heads.set_transform(Affine2D().scale(1 / 72) + ax.figure.dpi_scale_trans)
    ax.add_collection(heads, autolim=False)
    return heads


def _arrowhead_outlines(
    directions: np.ndarray, size: float, overshoot: float = 0.0
) -> np.ndarray:
    """`_ARROWHEAD` scaled to `size`, moved `overshoot` ahead and rotated along
    unit `directions` `(K, 2)`; shape `(K, V, 2)`, in points."""
    outline = size * _ARROWHEAD + np.array([overshoot, 0.0])
    cos, sin = directions.T
    rotations = np.stack([np.stack([cos, -sin], -1), np.stack([sin, cos], -1)], -2)
    return np.einsum('kij,vj->kvi', rotations, outline)


def _faded_colors(
    color: Any, progress: np.ndarray, *, fade: float, floor: float = 0.0
) -> np.ndarray:
    """RGBA `(K, 4)` of `color` with alpha scaled by `floor + (1 - floor) progress ** fade`."""
    colors = np.tile(np.asarray(to_rgba(color)), (len(progress), 1))
    colors[:, 3] *= floor + (1 - floor) * np.asarray(progress) ** fade
    return colors


def _scatter(
    ax: Axes, points: np.ndarray, *, colors: np.ndarray, size: float, zorder: float
) -> PathCollection | None:
    """Dots of diameter `size` (points), leaving the view limits unchanged."""
    if not len(points):
        return None
    autoscale = ax.get_autoscale_on()
    ax.set_autoscale_on(False)
    try:
        return ax.scatter(*points.T, s=size**2, c=colors, linewidths=0, zorder=zorder)
    finally:
        ax.set_autoscale_on(autoscale)


def _expm1(values: np.ndarray) -> np.ndarray:
    """Complex $e^x - 1$, accurate for small $|x|$."""
    small = np.abs(values) < 1e-5
    series = values * (1 + values / 2 * (1 + values / 3))
    return np.where(small, series, np.exp(values) - 1)


def _finite_max(values, default: float = 0.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else default


def _checked_points(points) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f'expected trail points (N, M, 2); got {points.shape}')
    return points


def _latest_finite_index(finite: np.ndarray) -> np.ndarray:
    """Index `(N,)` of the latest True entry of each row of `finite` `(N, M)`."""
    return finite.shape[1] - 1 - np.argmax(finite[:, ::-1], axis=1)


def _check_box(xlim, ylim) -> None:
    if xlim[0] >= xlim[1] or ylim[0] >= ylim[1]:
        raise ValueError(f'expected increasing limits; got {xlim}, {ylim}')
