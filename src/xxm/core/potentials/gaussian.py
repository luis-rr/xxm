"""Gaussian log potentials in canonical quadratic form."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from xxm.core import batch
from xxm.core.dists.gaussian import (
    Gaussian,
    GaussianChainMarginals,
    LinearGaussian,
    PairedGaussian,
)
from xxm.core.lengths import Lengths


def _precision_and_log_det(
    covariance: jax.Array,  # (..., N, N)
) -> tuple[jax.Array, jax.Array]:
    """Convert covariance to precision and log determinant."""

    if covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError('covariance must have shape (..., N, N)')

    variable_dim = covariance.shape[-1]

    cholesky = jnp.linalg.cholesky(covariance)

    identity = jnp.broadcast_to(
        jnp.eye(variable_dim, dtype=covariance.dtype),
        covariance.shape,
    )

    precision = jsp_linalg.cho_solve(
        (cholesky, True),
        identity,
    )

    log_det = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )

    return precision, log_det


class GaussianPotential(typing.NamedTuple):
    r"""Gaussian potential in canonical form.

    $$\log \phi(u) = -\frac12 u^\top J u + h^\top u + c.$$

    `precision_blocks` stores $J$, `information_vectors` stores $h$,
    and `log_constant` stores $c$. Leading dimensions are batch dimensions.
    """

    precision_blocks: jax.Array  # (..., N, N)
    information_vectors: jax.Array  # (..., N)
    log_constant: jax.Array  # (...)

    def normalize(self) -> Gaussian:
        """Return the normalized Gaussian distribution defined by this potential.

        Requires a positive-definite precision matrix.
        """
        return Gaussian.from_canonical(
            precision=self.precision_blocks,
            information=self.information_vectors,
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape shared by the canonical parameters."""
        precision_shape = self.precision_blocks.shape[:-2]
        information_shape = self.information_vectors.shape[:-1]
        log_constant_shape = self.log_constant.shape
        batch.require_same_shape(precision_shape, information_shape, log_constant_shape)
        return precision_shape

    @property
    def variable_dim(self) -> int:
        """Dimension of the potential's variable."""
        return self.precision_blocks.shape[-1]

    @classmethod
    def from_moments(
        cls,
        gaussian: Gaussian,
    ) -> GaussianPotential:
        r"""Convert Gaussian moments $(\mu,\Sigma)$ to canonical parameters $(J,h,c)$."""
        if gaussian.mean.ndim < 1:
            raise ValueError('mean must have shape (..., N)')

        if gaussian.covariance.ndim < 2:
            raise ValueError('covariance must have shape (..., N, N)')

        d = gaussian.mean.shape[-1]
        batch_shape = gaussian.mean.shape[:-1]

        if d < 1:
            raise ValueError('mean must contain at least one variable dimension')

        if gaussian.covariance.shape != batch_shape + (d, d):
            raise ValueError(
                'mean and covariance must have shapes (..., N) and (..., N, N) '
                'with matching leading dimensions'
            )

        precision, log_det_covariance = _precision_and_log_det(gaussian.covariance)

        information = jnp.einsum(
            '...ij,...j->...i',
            precision,
            gaussian.mean,
        )

        quadratic = jnp.sum(gaussian.mean * information, axis=-1)

        return cls(
            precision_blocks=precision,
            information_vectors=information,
            log_constant=(
                -0.5 * quadratic
                - 0.5 * log_det_covariance
                - 0.5 * d * jnp.log(2.0 * jnp.pi)
            ),
        )

    @classmethod
    def from_linear_likelihood(
        cls,
        model: LinearGaussian,
        observations: jax.Array,
    ) -> GaussianPotential:
        """Construct the potential over inputs induced by a linear Gaussian likelihood."""
        if model.input_ndim != 1:
            raise ValueError(
                'Gaussian potentials require a vector-shaped linear input; '
                f'got {model.input_shape}'
            )

        if observations.shape[-1] != model.output_dim:
            raise ValueError(
                'observations must have trailing output dimension '
                f'{model.output_dim}; got {observations.shape}'
            )

        batch_shape = model.batch_shape
        query_shape = observations.shape[:-1]

        coefficients = model.affine.coefficients_flat

        cholesky = jnp.linalg.cholesky(model.covariance)

        whitened_coefficients = jsp_linalg.solve_triangular(
            cholesky,
            coefficients,
            lower=True,
        )

        precision_block = jnp.einsum(
            '...nd,...ne->...de',
            whitened_coefficients,
            whitened_coefficients,
        )

        whitened_residuals = Gaussian(
            mean=model.affine.bias,
            covariance=model.covariance,
        ).whiten(observations)

        information_vectors = jnp.einsum(
            '...nd,...n->...d',
            batch.align_array(
                whitened_coefficients,
                batch_shape,
                query_shape,
            ),
            whitened_residuals,
        )

        quadratic_terms = jnp.sum(
            whitened_residuals**2,
            axis=-1,
        )

        log_det_covariance = 2.0 * jnp.sum(
            jnp.log(
                jnp.diagonal(
                    cholesky,
                    axis1=-2,
                    axis2=-1,
                )
            ),
            axis=-1,
        )

        log_det_covariance = batch.align_array(
            log_det_covariance,
            batch_shape,
            query_shape,
        )

        return cls(
            precision_blocks=batch.align_array(
                precision_block,
                batch_shape,
                query_shape,
            ),
            information_vectors=information_vectors,
            log_constant=-0.5
            * (
                quadratic_terms
                + log_det_covariance
                + model.output_dim * jnp.log(2.0 * jnp.pi)
            ),
        )

    @classmethod
    def from_local_quadratic(
        cls,
        point: jax.Array,  # (..., D)
        log_value: jax.Array,  # (...)
        gradient: jax.Array,  # (..., D)
        precision: jax.Array,  # (..., D, D)
    ) -> GaussianPotential:
        r"""Construct a local quadratic potential around `point`.

        $$\log \phi(u) \approx a + g^\top(u-u_0)
        -\frac12 (u-u_0)^\top J(u-u_0).$$

        `point` stores $u_0$, `log_value` stores $a$, `gradient` stores $g$,
        and `precision` stores the negative Hessian $J$ at $u_0$.
        All inputs must have the same structural batch prefix.
        """
        if point.ndim < 1:
            raise ValueError('point must have shape (*B, D)')
        if (
            gradient.shape != point.shape
            or log_value.shape != point.shape[:-1]
            or precision.shape != point.shape + (point.shape[-1],)
        ):
            raise ValueError(
                'local quadratic inputs must have aligned shapes: point (*B, D), '
                'log_value (*B), gradient (*B, D), precision (*B, D, D)'
            )

        information = gradient + jnp.einsum(
            '...ij,...j->...i',
            precision,
            point,
        )

        log_constant = (
            log_value
            - jnp.sum(gradient * point, axis=-1)
            - 0.5
            * jnp.einsum(
                '...i,...ij,...j->...',
                point,
                precision,
                point,
            )
        )

        return cls(
            precision_blocks=precision,
            information_vectors=information,
            log_constant=log_constant,
        )

    def weighted_sum(self, weights: jax.Array, *, axis: int) -> typing.Self:
        """Sum log potentials over an explicit, aligned batch axis."""
        batch_shape = self.batch_shape
        return self.__class__(
            precision_blocks=batch.weighted_sum(
                self.precision_blocks, weights, batch_shape, axis
            ),
            information_vectors=batch.weighted_sum(
                self.information_vectors, weights, batch_shape, axis
            ),
            log_constant=batch.weighted_sum(
                self.log_constant, weights, batch_shape, axis
            ),
        )

    def log_potential(self, values: jax.Array) -> jax.Array:
        r"""Evaluate the log potential at aligned deterministic values."""

        expected = (*self.batch_shape, self.variable_dim)
        if values.shape != expected:
            raise ValueError(f'values must have shape {expected}; got {values.shape}')

        return (
            -0.5
            * jnp.einsum(
                '...i,...ij,...j->...',
                values,
                self.precision_blocks,
                values,
            )
            + jnp.einsum(
                '...i,...i->...',
                self.information_vectors,
                values,
            )
            + self.log_constant
        )

    def expected_log_potential(
        self,
        mean: jax.Array,
        second_moment: jax.Array,
    ) -> jax.Array:
        r"""Compute the Gaussian expectation of this log potential.

        Moment batches begin with the potential batch; query axes follow.
        `mean` stores $\mathbb{E}[u]$ and `second_moment` stores
        $\mathbb{E}[uu^\top]$.
        """
        if mean.shape[-1] != self.variable_dim:
            raise ValueError(
                f'mean must have trailing dimension {self.variable_dim}; '
                f'got {mean.shape}'
            )

        if second_moment.shape[-2:] != (
            self.variable_dim,
            self.variable_dim,
        ):
            raise ValueError(
                'second_moment must have trailing shape '
                f'{(self.variable_dim, self.variable_dim)}; '
                f'got {second_moment.shape}'
            )

        batch.require_same_shape(mean.shape[:-1], second_moment.shape[:-2])
        shape = mean.shape[:-1]
        precision = batch.align_array(self.precision_blocks, self.batch_shape, shape)
        information = batch.align_array(
            self.information_vectors, self.batch_shape, shape
        )
        constant = batch.align_array(self.log_constant, self.batch_shape, shape)

        return (
            -0.5
            * jnp.einsum(
                '...ij,...ij->...',
                precision,
                second_moment,
            )
            + jnp.einsum(
                '...i,...i->...',
                information,
                mean,
            )
            + constant
        )

    def expected_log_potential_broadcast(
        self, mean: jax.Array, second_moment: jax.Array
    ) -> jax.Array:
        """Evaluate all moment tuples with receiver batch axes first."""
        return self.expected_log_potential(
            jnp.broadcast_to(mean, self.batch_shape + mean.shape),
            jnp.broadcast_to(second_moment, self.batch_shape + second_moment.shape),
        )

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)

    def scale(
        self,
        weights: jax.Array,
    ) -> typing.Self:
        """
        Scale each aligned log potential by `weights`.

        A zero weight produces the neutral factor.
        """
        if weights.shape != self.batch_shape:
            raise ValueError('weights must match the potential batch shape')

        weights = weights.astype(self.precision_blocks.dtype)

        return type(self)(
            precision_blocks=(self.precision_blocks * weights[..., None, None]),
            information_vectors=(self.information_vectors * weights[..., None]),
            log_constant=self.log_constant * weights,
        )

    def add(self, other: typing.Self) -> typing.Self:
        """Add aligned log potentials."""
        batch.require_same_shape(self.batch_shape, other.batch_shape)
        return jax.tree.map(jnp.add, self, other)

    def sub(self, other: typing.Self) -> typing.Self:
        """Subtract aligned log potentials."""
        batch.require_same_shape(self.batch_shape, other.batch_shape)
        return jax.tree.map(jnp.subtract, self, other)


class GaussianPairPotential(typing.NamedTuple):
    r"""Pairwise Gaussian potential in canonical form.

    $$\log f(x_0,x_1) = -\frac12x_0^\top J_{00}x_0
    -x_1^\top J_{10}x_0 -\frac12x_1^\top J_{11}x_1
    +h_0^\top x_0+h_1^\top x_1+c.$$

    `lower_precision` stores the lower off-diagonal block $J_{10}$; its
    transpose is the upper block $J_{01}$.

    Leading dimensions are treated as batch dimensions for independent potentials.
    """

    left_precision: jax.Array  # (..., N, N)
    right_precision: jax.Array  # (..., N, N)
    lower_precision: jax.Array  # (..., N, N)
    left_information: jax.Array  # (..., N)
    right_information: jax.Array  # (..., N)
    log_constant: jax.Array  # (...)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape shared by the pair-potential parameters."""
        left_precision_shape = self.left_precision.shape[:-2]
        right_precision_shape = self.right_precision.shape[:-2]
        lower_precision_shape = self.lower_precision.shape[:-2]
        left_information_shape = self.left_information.shape[:-1]
        right_information_shape = self.right_information.shape[:-1]
        log_constant_shape = self.log_constant.shape
        batch.require_same_shape(
            left_precision_shape,
            right_precision_shape,
            lower_precision_shape,
            left_information_shape,
            right_information_shape,
            log_constant_shape,
        )
        return left_precision_shape

    @property
    def variable_dim(self) -> int:
        """Dimension of each variable in the pair."""
        return self.left_precision.shape[-1]

    @classmethod
    def from_linear_conditional(
        cls,
        lin_gaussian: LinearGaussian,
    ) -> GaussianPairPotential:
        r"""Convert $p(x_1\mid x_0)$ into a joint pair potential over $(x_0,x_1)$."""
        if lin_gaussian.input_ndim != 1:
            raise ValueError(
                'Gaussian pair potentials require vector-shaped inputs; '
                f'got {lin_gaussian.input_shape}'
            )

        if (
            lin_gaussian.affine.coefficients.shape[-2]
            != lin_gaussian.affine.coefficients.shape[-1]
        ):
            raise ValueError('matrix must have shape (..., N, N)')

        n = lin_gaussian.affine.coefficients.shape[-1]
        batch_shape = lin_gaussian.affine.coefficients.shape[:-2]

        if n < 1:
            raise ValueError('matrix must contain at least one variable dimension')

        if lin_gaussian.affine.bias.shape != batch_shape + (n,):
            raise ValueError(
                'matrix and bias must have shapes (..., N, N) and (..., N) '
                'with matching leading dimensions'
            )

        if lin_gaussian.covariance.shape != batch_shape + (n, n):
            raise ValueError(
                'matrix and covariance must both have shape (..., N, N) '
                'with matching leading dimensions'
            )

        precision, log_det_covariance = _precision_and_log_det(lin_gaussian.covariance)

        matrix_t = jnp.swapaxes(
            lin_gaussian.affine.coefficients,
            -1,
            -2,
        )

        precision_matrix = precision @ lin_gaussian.affine.coefficients

        precision_bias = jnp.einsum(
            '...ij,...j->...i',
            precision,
            lin_gaussian.affine.bias,
        )

        return cls(
            left_precision=matrix_t @ precision_matrix,
            right_precision=precision,
            lower_precision=-precision_matrix,
            left_information=-jnp.einsum(
                '...ij,...j->...i',
                matrix_t,
                precision_bias,
            ),
            right_information=precision_bias,
            log_constant=(
                -0.5
                * jnp.sum(
                    lin_gaussian.affine.bias * precision_bias,
                    axis=-1,
                )
                - 0.5 * log_det_covariance
                - 0.5 * n * jnp.log(2.0 * jnp.pi)
            ),
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def weighted_sum(self, weights: jax.Array, *, axis: int) -> typing.Self:
        """Sum log potentials over an explicit, aligned batch axis."""
        batch_shape = self.batch_shape
        return self.__class__(
            left_precision=batch.weighted_sum(
                self.left_precision, weights, batch_shape, axis
            ),
            right_precision=batch.weighted_sum(
                self.right_precision, weights, batch_shape, axis
            ),
            lower_precision=batch.weighted_sum(
                self.lower_precision, weights, batch_shape, axis
            ),
            left_information=batch.weighted_sum(
                self.left_information, weights, batch_shape, axis
            ),
            right_information=batch.weighted_sum(
                self.right_information, weights, batch_shape, axis
            ),
            log_constant=batch.weighted_sum(
                self.log_constant, weights, batch_shape, axis
            ),
        )

    def log_potential(
        self,
        left: jax.Array,
        right: jax.Array,
    ) -> jax.Array:
        r"""Evaluate the pair log potential at aligned deterministic values."""

        expected = (*self.batch_shape, self.variable_dim)

        if left.shape != expected or right.shape != expected:
            raise ValueError(
                f'left and right must have shape {expected}; '
                f'got {left.shape} and {right.shape}'
            )

        return (
            -0.5
            * jnp.einsum(
                '...i,...ij,...j->...',
                left,
                self.left_precision,
                left,
            )
            - jnp.einsum(
                '...i,...ij,...j->...',
                right,
                self.lower_precision,
                left,
            )
            - 0.5
            * jnp.einsum(
                '...i,...ij,...j->...',
                right,
                self.right_precision,
                right,
            )
            + jnp.einsum(
                '...i,...i->...',
                self.left_information,
                left,
            )
            + jnp.einsum(
                '...i,...i->...',
                self.right_information,
                right,
            )
            + self.log_constant
        )

    def expected_log_potential(
        self,
        posterior: PairedGaussian,
    ) -> jax.Array:
        r"""Compute the expected pair log potential from aligned joint moments.

        The posterior and potential batches must be aligned.
        """
        if posterior.left_dim != self.variable_dim:
            raise ValueError(
                f'left variable dimension must be {self.variable_dim}; '
                f'got {posterior.left_dim}'
            )

        if posterior.right_dim != self.variable_dim:
            raise ValueError(
                f'right variable dimension must be {self.variable_dim}; '
                f'got {posterior.right_dim}'
            )
        batch.require_same_shape(self.batch_shape, posterior.batch_shape)

        left_mean = posterior.left.mean
        right_mean = posterior.right.mean

        left_second = posterior.left.covariance + jnp.einsum(
            '...i,...j->...ij',
            left_mean,
            left_mean,
        )

        right_second = posterior.right.covariance + jnp.einsum(
            '...i,...j->...ij',
            right_mean,
            right_mean,
        )

        # PairedGaussian stores Cov(right, left).
        left_right_moment = jnp.swapaxes(
            posterior.cross_covariance,
            -1,
            -2,
        ) + jnp.einsum(
            '...i,...j->...ij',
            left_mean,
            right_mean,
        )

        return (
            -0.5
            * jnp.einsum(
                '...ij,...ij->...',
                self.left_precision,
                left_second,
            )
            - jnp.einsum(
                '...ij,...ji->...',
                self.lower_precision,
                left_right_moment,
            )
            - 0.5
            * jnp.einsum(
                '...ij,...ij->...',
                self.right_precision,
                right_second,
            )
            + jnp.einsum(
                '...i,...i->...',
                self.left_information,
                left_mean,
            )
            + jnp.einsum(
                '...i,...i->...',
                self.right_information,
                right_mean,
            )
            + self.log_constant
        )

    def expected_log_potential_broadcast(self, posterior: PairedGaussian) -> jax.Array:
        """Evaluate all joint moments with receiver batch axes first."""
        posterior_batch = posterior.batch_shape
        potential = self.broadcast(
            posterior_batch,
            axis=len(self.batch_shape),
        )
        posterior = posterior.broadcast(self.batch_shape)
        return potential.expected_log_potential(posterior)

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)

    def scale(
        self,
        weights: jax.Array,
    ) -> typing.Self:
        """
        Scale each aligned pair log potential by `weights`.

        A zero weight produces the neutral factor.
        """
        if weights.shape != self.batch_shape:
            raise ValueError('weights must match the potential batch shape')

        weights = weights.astype(self.left_precision.dtype)

        matrix_weights = weights[..., None, None]
        vector_weights = weights[..., None]

        return type(self)(
            left_precision=self.left_precision * matrix_weights,
            right_precision=self.right_precision * matrix_weights,
            lower_precision=self.lower_precision * matrix_weights,
            left_information=self.left_information * vector_weights,
            right_information=self.right_information * vector_weights,
            log_constant=self.log_constant * weights,
        )

    def add(self, other: typing.Self) -> typing.Self:
        """Add aligned Gaussian pair log potentials."""
        batch.require_same_shape(self.batch_shape, other.batch_shape)
        return jax.tree.map(jnp.add, self, other)

    def sub(self, other: typing.Self) -> typing.Self:
        """Subtract aligned Gaussian pair log potentials."""
        batch.require_same_shape(self.batch_shape, other.batch_shape)
        return jax.tree.map(jnp.subtract, self, other)


class GaussianChainPotentials(typing.NamedTuple):
    initial: GaussianPotential
    pairs: GaussianPairPotential
    lengths: Lengths

    def expected_log_potential(
        self,
        posterior: GaussianChainMarginals,
    ) -> jax.Array:
        """Expected sum of initial and pair log potentials."""

        if posterior.max_steps == 0:
            return jnp.zeros_like(self.initial.log_constant)

        mean = posterior.means[..., 0, :]
        second = (
            posterior.covariances[..., 0, :, :]
            + mean[..., :, None] * mean[..., None, :]
        )

        initial = self.initial.expected_log_potential(mean, second)
        initial = jnp.where(self.lengths.values > 0, initial, 0)

        pairs = self.pairs.expected_log_potential(posterior.paired_marginals())
        pairs = self.lengths.transitions().clear(pairs)

        return initial + jnp.sum(pairs, axis=-1)
