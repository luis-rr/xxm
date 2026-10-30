"""Finite rectilinear grids of continuous probability distributions.

Each cell stores its *absolute* probability mass. Mass outside the finite
spatial grid is not redistributed; it is the residual to one, up to numerical
integration error. Cell probabilities are obtained by tensor-product
Gauss-Legendre quadrature, with order 1 corresponding to the midpoint rule.

At most three spatial dimensions are supported. Structural batch dimensions
precede the grid axes; temporal grids reserve the last underlying batch axis
for time, following the TemporalGaussian convention.
"""

import typing

import jax
import jax.core
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .. import batch
from ..lengths import Lengths
from .gaussian import Gaussian, TemporalGaussian

MAX_GRID_DIM = 3


def _broadcast_edges(
    edges: tuple[jax.Array, ...], batch_shape: tuple[int, ...]
) -> tuple[jax.Array, ...]:
    """Accept shared 1D edges or edges aligned to the structural batch."""
    if not 1 <= len(edges) <= MAX_GRID_DIM:
        raise ValueError(f'Grid dimension must be between 1 and {MAX_GRID_DIM}.')

    result = []

    for edge in edges:
        edge = jnp.asarray(edge)

        if edge.ndim == 1:
            edge = jnp.broadcast_to(edge, (*batch_shape, edge.shape[0]))

        elif edge.shape[:-1] != batch_shape:
            raise ValueError('Grid edges must be 1D or batch-aligned.')

        if edge.shape[-1] < 2:
            raise ValueError('Each grid dimension must contain at least one cell.')

        # Host-side validation when values are concrete. Inside jit, callers
        # must provide strictly increasing finite grid edges.
        if not isinstance(edge, jax.core.Tracer):
            values = np.asarray(edge)
            if not np.all(np.isfinite(values)) or not np.all(
                np.diff(values, axis=-1) > 0
            ):
                raise ValueError('Grid edges must be finite and strictly increasing.')

        result.append(edge)

    return tuple(result)


def _quadrature(order: int, dim: int, dtype) -> tuple[jax.Array, jax.Array]:
    """Product Gauss-Legendre nodes and weights on [-1, 1]^dim."""

    if not isinstance(order, int) or order < 1:
        raise ValueError('quadrature_order must be a positive static integer.')

    nodes, weights = np.polynomial.legendre.leggauss(order)
    node_mesh = np.meshgrid(*([nodes] * dim), indexing='ij')
    weight_mesh = np.meshgrid(*([weights] * dim), indexing='ij')
    points = np.stack(node_mesh, axis=-1).reshape(-1, dim)
    products = np.prod(np.stack(weight_mesh, axis=-1), axis=-1).reshape(-1)

    return jnp.asarray(points, dtype=dtype), jnp.asarray(products, dtype=dtype)


def _gaussian_masses(
    dist: Gaussian,
    edges: tuple[jax.Array, ...],
    quadrature_order: int,
) -> jax.Array:
    """Integrate a batched Gaussian over all spatial cells."""
    dim = len(edges)
    batch_shape = dist.batch_shape
    grid_shape = tuple(edge.shape[-1] - 1 for edge in edges)
    nodes, weights = _quadrature(quadrature_order, dim, dist.mean.dtype)
    num_nodes = nodes.shape[0]

    coordinates = []
    log_volume = jnp.zeros((*batch_shape, *grid_shape), dtype=dist.mean.dtype)
    for axis, edge in enumerate(edges):
        width = jnp.diff(edge, axis=-1)
        center = (edge[..., :-1] + edge[..., 1:]) / 2
        shape = (*batch_shape, *(1,) * axis, grid_shape[axis], *(1,) * (dim - axis - 1))
        center = center.reshape(shape)
        width = width.reshape(shape)
        # Quadrature nodes in the axis-specific interval.
        coordinates.append(
            center[..., None]
            + width[..., None]
            * nodes[:, axis].reshape((1,) * (len(shape)) + (num_nodes,))
            / 2
        )
        log_volume = log_volume + jnp.log(width)

    points = jnp.stack(jnp.broadcast_arrays(*coordinates), axis=-1)
    log_prob = dist.log_prob(points)  # (*B, *G, num_nodes)
    log_weights = jnp.log(weights).reshape((1,) * (log_prob.ndim - 1) + (num_nodes,))
    log_mass = jsp.special.logsumexp(log_prob + log_weights, axis=-1)
    log_mass = log_mass + log_volume - dim * jnp.log(2.0)

    return jnp.exp(log_mass)


def _selected_moments(
    posterior: TemporalGaussian,
    variables: tuple[int, ...] | None,
    grid_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Select true Gaussian marginals before discretization."""
    if posterior.dist.batch_shape != (*posterior.batch_shape, posterior.max_steps):
        raise ValueError('Temporal Gaussian must have batch shape (*B, T).')

    original_dim = posterior.variable_dim
    if variables is None:
        variables = tuple(range(original_dim))
    variables = tuple(variables)
    if len(variables) != grid_dim:
        raise ValueError('variables must select one coordinate per grid dimension.')
    if len(set(variables)) != len(variables) or any(
        not isinstance(v, int) or v < 0 or v >= original_dim for v in variables
    ):
        raise ValueError('variables must be distinct valid coordinate indices.')

    indices = jnp.asarray(variables, dtype=jnp.int32)
    means = jnp.take(posterior.dist.mean, indices, axis=-1)
    covariances = jnp.take(
        jnp.take(posterior.dist.covariance, indices, axis=-2), indices, axis=-1
    )
    return means, covariances


def _temporal_parameters(
    posterior: TemporalGaussian,
    variables: tuple[int, ...] | None,
    grid_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Replace padded Gaussian parameters by proper neutral Gaussians."""
    means, covariances = _selected_moments(posterior, variables, grid_dim)
    valid = posterior.lengths.to_mask(posterior.max_steps).values
    means = jnp.where(valid[..., None], means, 0)
    identity = jnp.eye(grid_dim, dtype=covariances.dtype)
    covariances = jnp.where(valid[..., None, None], covariances, identity)
    return means, covariances


class GriddedDistribution(typing.NamedTuple):
    """Absolute cell probabilities on a finite rectilinear grid.

    mass: (*B, G1, ..., GD)
    edges[d]: (*B, Gd + 1), one per spatial dimension

    The original continuous distribution may have nonzero probability outside
    the grid. `outside_mass` returns its estimated residual without clipping
    or renormalization. Quadrature error can make this residual slightly
    negative; validate integration accuracy separately when that occurs.
    """

    mass: jax.Array
    edges: tuple[jax.Array, ...]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.mass.shape[: -len(self.edges)]

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return self.mass.shape[-len(self.edges) :]

    @property
    def variable_dim(self) -> int:
        return len(self.edges)

    def validate_shapes(self) -> None:
        if not 1 <= self.variable_dim <= MAX_GRID_DIM:
            raise ValueError('Unsupported spatial grid dimension.')
        if self.mass.ndim < self.variable_dim:
            raise ValueError('The mass array has too few dimensions.')
        for edge, size in zip(self.edges, self.grid_shape):
            if edge.shape != (*self.batch_shape, size + 1):
                raise ValueError('Grid edges must match mass and batch shapes.')

    def density(self) -> jax.Array:
        """Cell-average PDF values, i.e. mass divided by cell volume."""
        result = self.mass

        for axis, edge in enumerate(self.edges):
            widths = jnp.diff(edge, axis=-1)
            shape = (
                *self.batch_shape,
                *(1,) * axis,
                self.grid_shape[axis],
                *(1,) * (self.variable_dim - axis - 1),
            )
            result = result / widths.reshape(shape)

        return result

    def outside_mass(self) -> jax.Array:
        """Residual mass outside the grid, including quadrature error."""
        axes = tuple(range(self.mass.ndim - self.variable_dim, self.mass.ndim))
        return 1.0 - jnp.sum(self.mass, axis=axes)

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    def get(self, index: batch.SelT | None = None) -> typing.Self:
        return batch.get(self, index)

    @classmethod
    def from_gaussian(
        cls,
        dist: Gaussian,
        edges: tuple[jax.Array, ...],
        *,
        quadrature_order: int = 3,
    ) -> typing.Self:
        """Integrate a Gaussian (or batch) into finite spatial cells.

        `edges` may be globally shared 1D arrays or arrays batch-aligned to
        the Gaussian. Does not condition on the rectangle or renormalize.
        """
        edges = _broadcast_edges(edges, dist.batch_shape)

        if len(edges) != dist.variable_dim:
            raise ValueError('Grid dimension must match Gaussian variable dimension.')

        mass = _gaussian_masses(dist, edges, quadrature_order)

        return cls(mass, edges)

    @classmethod
    def from_gaussian_occupancy(
        cls,
        posterior: TemporalGaussian,
        edges: tuple[jax.Array, ...],
        *,
        variables: tuple[int, ...] | None = None,
        quadrature_order: int = 3,
    ) -> typing.Self:
        """Time-average Gaussian marginal cell probabilities without a T-grid.

        Each independent sequence is averaged over its *valid* time steps.
        Requires at least one valid timestep per sequence.
        """
        bshape = posterior.batch_shape

        if posterior.max_steps < 1:
            raise ValueError('At least one stored timestep is required.')

        edges = _broadcast_edges(edges, bshape)
        means, covariances = _temporal_parameters(posterior, variables, len(edges))
        time_axis = len(bshape)
        grid_shape = tuple(edge.shape[-1] - 1 for edge in edges)
        initial = jnp.zeros((*bshape, *grid_shape), dtype=means.dtype)
        valid = posterior.lengths.to_mask(posterior.max_steps).values

        def add_time(t, total):
            mean = jax.lax.dynamic_index_in_dim(
                means, t, axis=time_axis, keepdims=False
            )
            cov = jax.lax.dynamic_index_in_dim(
                covariances, t, axis=time_axis, keepdims=False
            )
            at_t = cls.from_gaussian(
                Gaussian.from_moments(mean, cov),
                edges,
                quadrature_order=quadrature_order,
            ).mass
            active = jax.lax.dynamic_index_in_dim(
                valid, t, axis=time_axis, keepdims=False
            )
            return total + jnp.where(
                active.reshape((*bshape, *(1,) * len(grid_shape))), at_t, 0
            )

        total = jax.lax.fori_loop(0, posterior.max_steps, add_time, initial)
        lengths = posterior.lengths.values
        # Zero-length sequences have no occupancy distribution: return NaN.
        divisor = lengths.reshape((*bshape, *(1,) * len(grid_shape)))
        return cls(total / jnp.where(divisor > 0, divisor, jnp.nan), edges)


# TODO: Generalize to time bins. Store shared bin edges in steps (K + 1,) and
# valid-step counts (*B, K); each bin's mass is the mean of per-step masses
# over its valid steps, accumulated during the scan. Width 1 recovers the
# per-step grid and a single bin recovers `from_gaussian_occupancy`, which it
# would replace. `mean_time` must then weight bins by counts. This reduces
# memory and plotting cost (decisive for 2D over time), not integration cost.
# Bins should be narrow relative to the motion of the mean, or within-bin
# drift is shown as spread.
class TemporalGriddedDistribution(typing.NamedTuple):
    """Time-indexed grids with contiguous valid-prefix lengths.

    dist.mass: (*B, T, G1, ..., GD)
    dist.edges[d]: (*B, T, Gd + 1)
    lengths.values: (*B,)

    Spatial edges must be identical at every time in a sequence, so time
    averaging describes a single fixed spatial partition. The constructor
    `from_gaussian` guarantees this property.
    """

    dist: GriddedDistribution
    lengths: Lengths

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.lengths.batch_shape

    @property
    def max_steps(self) -> int:
        return self.dist.mass.shape[len(self.batch_shape)]

    @property
    def variable_dim(self) -> int:
        return self.dist.variable_dim

    def validate_shapes(self) -> None:
        if self.dist.batch_shape != (*self.batch_shape, self.max_steps):
            raise ValueError('Temporal grid must have batch shape (*B, T).')
        self.dist.validate_shapes()

    def density(self) -> jax.Array:
        """Cell-averaged density at each valid time step."""
        return self.dist.density()

    def outside_mass(self) -> jax.Array:
        """Residual mass per time step; padded times are not distributions."""
        return self.dist.outside_mass()

    def mean_time(self) -> GriddedDistribution:
        """Mean mass over the genuine temporal prefix of each sequence."""
        bshape = self.batch_shape
        time_axis = len(bshape)
        mass = jnp.sum(self.lengths.clear(self.dist.mass), axis=time_axis)
        divisor = self.lengths.values.reshape((*bshape, *(1,) * self.variable_dim))
        mass = mass / jnp.where(divisor > 0, divisor, jnp.nan)
        edges = tuple(jnp.take(edge, 0, axis=time_axis) for edge in self.dist.edges)
        return GriddedDistribution(mass, edges)

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    def get(self, index: batch.SelT | None = None) -> typing.Self:
        return batch.get(self, index)

    @classmethod
    def from_gaussian(
        cls,
        posterior: TemporalGaussian,
        edges: tuple[jax.Array, ...],
        *,
        variables: tuple[int, ...] | None = None,
        quadrature_order: int = 3,
    ) -> typing.Self:
        """Discretize Gaussian marginal posteriors at every valid timestep.

        Cross-time covariances are deliberately not used. Padded time steps
        store zero mass and should be excluded using `lengths`.
        """
        bshape = posterior.batch_shape
        if posterior.max_steps < 1:
            raise ValueError('At least one stored timestep is required.')
        edges = _broadcast_edges(edges, bshape)
        means, covariances = _temporal_parameters(posterior, variables, len(edges))
        time_axis = len(bshape)
        steps = posterior.max_steps
        valid = posterior.lengths.to_mask(steps).values

        def at_time(_, t):
            mean = jax.lax.dynamic_index_in_dim(
                means, t, axis=time_axis, keepdims=False
            )
            cov = jax.lax.dynamic_index_in_dim(
                covariances, t, axis=time_axis, keepdims=False
            )
            mass = GriddedDistribution.from_gaussian(
                Gaussian.from_moments(mean, cov),
                edges,
                quadrature_order=quadrature_order,
            ).mass
            active = jax.lax.dynamic_index_in_dim(
                valid, t, axis=time_axis, keepdims=False
            )
            mass = jnp.where(active.reshape((*bshape, *(1,) * len(edges))), mass, 0)
            return None, mass

        _, stacked = jax.lax.scan(at_time, None, jnp.arange(steps))
        mass = jnp.moveaxis(stacked, 0, time_axis)
        time_edges = tuple(
            jnp.broadcast_to(edge[..., None, :], (*bshape, steps, edge.shape[-1]))
            for edge in edges
        )
        return cls(GriddedDistribution(mass, time_edges), posterior.lengths)

    def at_time(self, index: int) -> GriddedDistribution:
        """Return the spatial distribution at one time step.

        Selects the same time index across all batch elements.
        Padded time steps contain zero mass and are not valid distributions.
        """
        if not -self.max_steps <= index < self.max_steps:
            raise IndexError('Time index out of bounds.')

        axis = len(self.batch_shape)
        mass = jnp.take(self.dist.mass, index, axis=axis)
        edges = tuple(jnp.take(edge, index, axis=axis) for edge in self.dist.edges)
        return GriddedDistribution(mass, edges)
