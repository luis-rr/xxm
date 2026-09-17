r"""Gaussian chain inference using canonical form and sequential block elimination.

The canonical potential is

$$\log f(x) = -\frac12 x^\top J x + h^\top x + c,$$

where $J$ is symmetric positive definite. The corresponding normalized Gaussian has mean
$J^{-1}h$ and covariance $J^{-1}$. The log normalizer is $\log \int f(x)\, dx$.
"""

from __future__ import annotations

import math
import typing

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from xxm.core import _batch
from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian, PairedGaussian


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

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape shared by the canonical parameters."""
        precision_shape = self.precision_blocks.shape[:-2]
        information_shape = self.information_vectors.shape[:-1]
        log_constant_shape = self.log_constant.shape
        assert precision_shape == information_shape == log_constant_shape
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

        residuals = observations - _batch.align_array(
            model.affine.bias,
            batch_shape,
            query_shape,
        )

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

        whitened_residuals = jsp_linalg.solve_triangular(
            _batch.align_array(
                cholesky,
                batch_shape,
                query_shape,
            ),
            residuals[..., None],
            lower=True,
        )[..., 0]

        information_vectors = jnp.einsum(
            '...nd,...n->...d',
            _batch.align_array(
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

        log_det_covariance = _batch.align_array(
            log_det_covariance,
            batch_shape,
            query_shape,
        )

        return cls(
            precision_blocks=_batch.align_array(
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
            precision_blocks=_batch.weighted_sum(
                self.precision_blocks, weights, batch_shape, axis
            ),
            information_vectors=_batch.weighted_sum(
                self.information_vectors, weights, batch_shape, axis
            ),
            log_constant=_batch.weighted_sum(
                self.log_constant, weights, batch_shape, axis
            ),
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

        _batch.require_same(mean.shape[:-1], second_moment.shape[:-2])
        shape = mean.shape[:-1]
        precision = _batch.align_array(self.precision_blocks, self.batch_shape, shape)
        information = _batch.align_array(
            self.information_vectors, self.batch_shape, shape
        )
        constant = _batch.align_array(self.log_constant, self.batch_shape, shape)

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

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = _batch.selection(index, len(self.batch_shape))
        return self.__class__(
            precision_blocks=self.precision_blocks[index],
            information_vectors=self.information_vectors[index],
            log_constant=self.log_constant[index],
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            precision_blocks=_batch.broadcast_array(self.precision_blocks, shape, axis),
            information_vectors=_batch.broadcast_array(
                self.information_vectors, shape, axis
            ),
            log_constant=_batch.broadcast_array(self.log_constant, shape, axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            precision_blocks=jnp.squeeze(self.precision_blocks, axis=axes),
            information_vectors=jnp.squeeze(self.information_vectors, axis=axes),
            log_constant=jnp.squeeze(self.log_constant, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            precision_blocks=jnp.take(self.precision_blocks, permutation, axis=axis),
            information_vectors=jnp.take(
                self.information_vectors, permutation, axis=axis
            ),
            log_constant=jnp.take(self.log_constant, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            precision_blocks=jnp.moveaxis(self.precision_blocks, source, destination),
            information_vectors=jnp.moveaxis(
                self.information_vectors, source, destination
            ),
            log_constant=jnp.moveaxis(self.log_constant, source, destination),
        )


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
        assert (
            left_precision_shape
            == right_precision_shape
            == lower_precision_shape
            == left_information_shape
            == right_information_shape
            == log_constant_shape
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
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            left_precision=_batch.broadcast_array(self.left_precision, shape, axis),
            right_precision=_batch.broadcast_array(self.right_precision, shape, axis),
            lower_precision=_batch.broadcast_array(self.lower_precision, shape, axis),
            left_information=_batch.broadcast_array(self.left_information, shape, axis),
            right_information=_batch.broadcast_array(
                self.right_information, shape, axis
            ),
            log_constant=_batch.broadcast_array(self.log_constant, shape, axis),
        )

    def weighted_sum(self, weights: jax.Array, *, axis: int) -> typing.Self:
        """Sum log potentials over an explicit, aligned batch axis."""
        batch_shape = self.batch_shape
        return self.__class__(
            left_precision=_batch.weighted_sum(
                self.left_precision, weights, batch_shape, axis
            ),
            right_precision=_batch.weighted_sum(
                self.right_precision, weights, batch_shape, axis
            ),
            lower_precision=_batch.weighted_sum(
                self.lower_precision, weights, batch_shape, axis
            ),
            left_information=_batch.weighted_sum(
                self.left_information, weights, batch_shape, axis
            ),
            right_information=_batch.weighted_sum(
                self.right_information, weights, batch_shape, axis
            ),
            log_constant=_batch.weighted_sum(
                self.log_constant, weights, batch_shape, axis
            ),
        )

    def expected_log_potential(
        self,
        posterior: PairedGaussian,
    ) -> jax.Array:
        r"""Compute the expected pair log potential from aligned joint moments.

        Posterior batches begin with the potential batch; query axes follow.
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

        batch_shape = self.batch_shape
        shape = posterior.batch_shape
        potential = GaussianPairPotential(
            left_precision=_batch.align_array(self.left_precision, batch_shape, shape),
            right_precision=_batch.align_array(
                self.right_precision, batch_shape, shape
            ),
            lower_precision=_batch.align_array(
                self.lower_precision, batch_shape, shape
            ),
            left_information=_batch.align_array(
                self.left_information, batch_shape, shape
            ),
            right_information=_batch.align_array(
                self.right_information, batch_shape, shape
            ),
            log_constant=_batch.align_array(self.log_constant, batch_shape, shape),
        )

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
                potential.left_precision,
                left_second,
            )
            - jnp.einsum(
                '...ij,...ji->...',
                potential.lower_precision,
                left_right_moment,
            )
            - 0.5
            * jnp.einsum(
                '...ij,...ij->...',
                potential.right_precision,
                right_second,
            )
            + jnp.einsum(
                '...i,...i->...',
                potential.left_information,
                left_mean,
            )
            + jnp.einsum(
                '...i,...i->...',
                potential.right_information,
                right_mean,
            )
            + potential.log_constant
        )

    def expected_log_potential_broadcast(self, posterior: PairedGaussian) -> jax.Array:
        """Evaluate all joint moments with receiver batch axes first."""
        return self.expected_log_potential(posterior.broadcast(self.batch_shape))

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = _batch.selection(index, len(self.batch_shape))
        return self.__class__(
            left_precision=self.left_precision[index],
            right_precision=self.right_precision[index],
            lower_precision=self.lower_precision[index],
            left_information=self.left_information[index],
            right_information=self.right_information[index],
            log_constant=self.log_constant[index],
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            left_precision=jnp.squeeze(self.left_precision, axis=axes),
            right_precision=jnp.squeeze(self.right_precision, axis=axes),
            lower_precision=jnp.squeeze(self.lower_precision, axis=axes),
            left_information=jnp.squeeze(self.left_information, axis=axes),
            right_information=jnp.squeeze(self.right_information, axis=axes),
            log_constant=jnp.squeeze(self.log_constant, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            left_precision=jnp.take(self.left_precision, permutation, axis=axis),
            right_precision=jnp.take(self.right_precision, permutation, axis=axis),
            lower_precision=jnp.take(self.lower_precision, permutation, axis=axis),
            left_information=jnp.take(self.left_information, permutation, axis=axis),
            right_information=jnp.take(self.right_information, permutation, axis=axis),
            log_constant=jnp.take(self.log_constant, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            left_precision=jnp.moveaxis(self.left_precision, source, destination),
            right_precision=jnp.moveaxis(self.right_precision, source, destination),
            lower_precision=jnp.moveaxis(self.lower_precision, source, destination),
            left_information=jnp.moveaxis(self.left_information, source, destination),
            right_information=jnp.moveaxis(self.right_information, source, destination),
            log_constant=jnp.moveaxis(self.log_constant, source, destination),
        )


class GaussianChain(typing.NamedTuple):
    r"""Represents a Gaussian chain with block-tridiagonal structure in canonical form.

    $$\log f(x)=-\frac12x^\top Jx+h^\top x+c.$$

    Here $J$ is symmetric positive definite and block tridiagonal,
    $h$ is the information vector, and $c$ is `log_constant`.

    The lower block convention is $B_t=J_{t+1,t}$, so the upper block is
    $B_t^\top$.

    `information_vectors[t]` contains the block of $h$ associated with $x_t$.

    Leading batch dimensions index independent chains; time and variable
    dimensions are intrinsic.
    """

    diagonal_precision_blocks: jax.Array  # (*B, T, N, N)
    lower_precision_blocks: jax.Array  # (*B, T - 1, N, N)
    information_vectors: jax.Array  # (*B, T, N)
    log_constant: jax.Array  # (*B,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading independent-chain dimensions."""
        return self.diagonal_precision_blocks.shape[:-3]

    def forward_backward(self) -> tuple[GaussianChainMarginals, jax.Array]:
        """Infer each chain independently, preserving the batch prefix."""
        if self.num_steps < 1:
            raise ValueError('Gaussian chain inference requires at least one time step')
        batch = self.batch_shape
        t, d = self.num_steps, self.variable_dim
        if (
            self.diagonal_precision_blocks.shape != batch + (t, d, d)
            or self.lower_precision_blocks.shape != batch + (t - 1, d, d)
            or self.information_vectors.shape != batch + (t, d)
            or self.log_constant.shape != batch
        ):
            raise ValueError(
                'Gaussian chain fields must have aligned batch, time, and variable dimensions'
            )
        if not batch:
            return self._forward_backward_single()
        n, t, d = math.prod(batch), self.num_steps, self.variable_dim
        flat = GaussianChain(
            diagonal_precision_blocks=self.diagonal_precision_blocks.reshape(
                n, t, d, d
            ),
            lower_precision_blocks=self.lower_precision_blocks.reshape(n, t - 1, d, d),
            information_vectors=self.information_vectors.reshape(n, t, d),
            log_constant=self.log_constant.reshape(n),
        )
        posterior, log_normalizer = jax.vmap(GaussianChain._forward_backward_single)(
            flat
        )
        return GaussianChainMarginals(
            means=posterior.means.reshape(batch + (t, d)),
            covariances=posterior.covariances.reshape(batch + (t, d, d)),
            cross_covariances=posterior.cross_covariances.reshape(
                batch + (t - 1, d, d)
            ),
        ), log_normalizer.reshape(batch)

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = _batch.selection(index, len(self.batch_shape))
        return self.__class__(
            diagonal_precision_blocks=self.diagonal_precision_blocks[index],
            lower_precision_blocks=self.lower_precision_blocks[index],
            information_vectors=self.information_vectors[index],
            log_constant=self.log_constant[index],
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            diagonal_precision_blocks=_batch.broadcast_array(
                self.diagonal_precision_blocks, shape, axis
            ),
            lower_precision_blocks=_batch.broadcast_array(
                self.lower_precision_blocks, shape, axis
            ),
            information_vectors=_batch.broadcast_array(
                self.information_vectors, shape, axis
            ),
            log_constant=_batch.broadcast_array(self.log_constant, shape, axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            diagonal_precision_blocks=jnp.squeeze(
                self.diagonal_precision_blocks, axis=axes
            ),
            lower_precision_blocks=jnp.squeeze(self.lower_precision_blocks, axis=axes),
            information_vectors=jnp.squeeze(self.information_vectors, axis=axes),
            log_constant=jnp.squeeze(self.log_constant, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            diagonal_precision_blocks=jnp.take(
                self.diagonal_precision_blocks, permutation, axis=axis
            ),
            lower_precision_blocks=jnp.take(
                self.lower_precision_blocks, permutation, axis=axis
            ),
            information_vectors=jnp.take(
                self.information_vectors, permutation, axis=axis
            ),
            log_constant=jnp.take(self.log_constant, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            diagonal_precision_blocks=jnp.moveaxis(
                self.diagonal_precision_blocks, source, destination
            ),
            lower_precision_blocks=jnp.moveaxis(
                self.lower_precision_blocks, source, destination
            ),
            information_vectors=jnp.moveaxis(
                self.information_vectors, source, destination
            ),
            log_constant=jnp.moveaxis(self.log_constant, source, destination),
        )

    @classmethod
    def from_pair_potentials(
        cls,
        initial_potential: GaussianPotential,  # batch *B
        pair_potentials: GaussianPairPotential,  # batch (*B, T - 1)
    ) -> GaussianChain:
        """Construct chains from potential batches `*B` and `(*B, T - 1)`."""

        if pair_potentials.variable_dim != initial_potential.variable_dim:
            raise ValueError(
                'initial_potential and pair_potentials must have the same variable dimension. '
                f'Got {initial_potential.variable_dim} and {pair_potentials.variable_dim}'
            )

        batch_shape = initial_potential.batch_shape
        if (
            len(pair_potentials.batch_shape) != len(batch_shape) + 1
            or pair_potentials.batch_shape[:-1] != batch_shape
        ):
            raise ValueError(
                'pair potentials must have batch shape (*B, T - 1) matching the initial potential'
            )
        num_steps = pair_potentials.batch_shape[-1] + 1
        latent_dim = initial_potential.variable_dim
        dtype = initial_potential.precision_blocks.dtype

        diagonal = jnp.zeros(
            batch_shape + (num_steps, latent_dim, latent_dim),
            dtype=dtype,
        )
        diagonal = diagonal.at[..., 0, :, :].add(initial_potential.precision_blocks)
        diagonal = diagonal.at[..., :-1, :, :].add(pair_potentials.left_precision)
        diagonal = diagonal.at[..., 1:, :, :].add(pair_potentials.right_precision)

        information_vectors = jnp.zeros(
            batch_shape + (num_steps, latent_dim),
            dtype=dtype,
        )
        information_vectors = information_vectors.at[..., 0, :].add(
            initial_potential.information_vectors
        )
        information_vectors = information_vectors.at[..., :-1, :].add(
            pair_potentials.left_information
        )
        information_vectors = information_vectors.at[..., 1:, :].add(
            pair_potentials.right_information
        )

        log_constant = initial_potential.log_constant + jnp.sum(
            pair_potentials.log_constant, axis=-1
        )

        return cls(
            diagonal_precision_blocks=diagonal,
            lower_precision_blocks=pair_potentials.lower_precision,
            information_vectors=information_vectors,
            log_constant=log_constant,
        )

    @property
    def num_steps(self) -> int:
        """Number of time steps $T$."""
        return self.diagonal_precision_blocks.shape[-3]

    @property
    def variable_dim(self) -> int:
        """Dimension of each continuous state."""
        return self.diagonal_precision_blocks.shape[-1]

    def add_local_potential(
        self,
        potential: GaussianPotential,
    ) -> GaussianChain:
        """Add one unary Gaussian potential at each time step."""

        expected_batch_shape = self.batch_shape + (self.num_steps,)

        if potential.batch_shape != expected_batch_shape:
            raise ValueError(
                'potential must contain exactly one potential per time step; '
                f'expected leading shape {expected_batch_shape}, '
                f'got {potential.batch_shape}'
            )

        if potential.variable_dim != self.variable_dim:
            raise ValueError(
                'potential variable dimension must match chain variable dimension; '
                f'expected {self.variable_dim}, got {potential.variable_dim}'
            )

        return GaussianChain(
            diagonal_precision_blocks=(
                self.diagonal_precision_blocks + potential.precision_blocks
            ),
            lower_precision_blocks=self.lower_precision_blocks,
            information_vectors=(
                self.information_vectors + potential.information_vectors
            ),
            log_constant=(self.log_constant + jnp.sum(potential.log_constant, axis=-1)),
        )

    def log_potential(
        self,
        latent: jax.Array,
    ) -> jax.Array:
        r"""Compute $\log f(x)$ for a latent trajectory."""

        expected_shape = self.batch_shape + (self.num_steps, self.variable_dim)

        if latent.shape != expected_shape:
            raise ValueError(
                f'latent must have shape {expected_shape}. Got shape {latent.shape}'
            )

        diagonal_terms = jnp.einsum(
            '...ti,...tij,...tj->...',
            latent,
            self.diagonal_precision_blocks,
            latent,
        )

        cross_terms = jnp.einsum(
            '...ti,...tij,...tj->...',
            latent[..., 1:, :],
            self.lower_precision_blocks,
            latent[..., :-1, :],
        )

        linear_terms = jnp.sum(latent * self.information_vectors, axis=(-2, -1))

        return -0.5 * diagonal_terms - cross_terms + linear_terms + self.log_constant

    def dense_precision(self) -> jax.Array:
        """Construct the dense precision matrix for reference and testing.

        This helper is intentionally dense and should not be used by the main
        Gaussian-chain inference path.
        """

        diagonal_blocks = self.diagonal_precision_blocks
        lower_blocks = self.lower_precision_blocks

        time_steps, variable_dim, _ = diagonal_blocks.shape[-3:]
        dense_size = time_steps * variable_dim

        dense = jnp.zeros(
            self.batch_shape + (dense_size, dense_size),
            dtype=diagonal_blocks.dtype,
        )

        for t in range(time_steps):
            start = t * variable_dim
            stop = (t + 1) * variable_dim

            dense = dense.at[..., start:stop, start:stop].set(
                diagonal_blocks[..., t, :, :]
            )

            if t < time_steps - 1:
                next_start = (t + 1) * variable_dim
                next_stop = (t + 2) * variable_dim
                lower_block = lower_blocks[..., t, :, :]

                dense = dense.at[
                    ...,
                    start:stop,
                    next_start:next_stop,
                ].set(jnp.swapaxes(lower_block, -1, -2))

                dense = dense.at[
                    ...,
                    next_start:next_stop,
                    start:stop,
                ].set(lower_block)

        return dense

    def _forward_elimination(
        self,
    ) -> _GaussianChainFactorization:
        """Perform forward block elimination for a Gaussian chain."""

        initial_carry = _ForwardEliminationCarry.from_canonical(
            self.diagonal_precision_blocks[0],
            self.information_vectors[0],
        )

        def step(
            carry: _ForwardEliminationCarry,
            inputs: tuple[jax.Array, jax.Array, jax.Array],
        ):
            lower_precision, next_precision, next_information = inputs

            result = carry.eliminate_next(
                lower_precision,
                next_precision,
                next_information,
            )

            return result.next_carry, result

        final_carry, results = jax.lax.scan(
            step,
            initial_carry,
            (
                self.lower_precision_blocks,
                self.diagonal_precision_blocks[1:],
                self.information_vectors[1:],
            ),
        )

        precision_cholesky_factors = jnp.concatenate(
            [
                initial_carry.cholesky[None],
                results.next_carry.cholesky,
            ],
        )

        return _GaussianChainFactorization(
            precision_cholesky_factors=precision_cholesky_factors,
            conditional_mean=results.conditional_mean,
            terminal_mean=final_carry.mean_offset,
        )

    def _forward_backward_single(
        self,
    ) -> tuple[GaussianChainMarginals, jax.Array]:
        """Compute moments and log normalizer for a Gaussian chain."""

        factorization = self._forward_elimination()

        means = factorization.backward_means()

        covariances, cross_covariances = factorization.backward_covariances()

        log_det_precision = jnp.sum(
            jax.vmap(_log_det_from_cholesky)(factorization.precision_cholesky_factors)
        )

        quadratic_term = jnp.sum(self.information_vectors * means)

        total_dimension = self.num_steps * self.variable_dim

        log_normalizer = (
            self.log_constant
            + 0.5 * quadratic_term
            - 0.5 * log_det_precision
            + 0.5
            * total_dimension
            * jnp.log(
                jnp.array(
                    2.0 * jnp.pi,
                    dtype=self.diagonal_precision_blocks.dtype,
                )
            )
        )

        posterior = GaussianChainMarginals(
            means=means,
            covariances=covariances,
            cross_covariances=cross_covariances,
        )

        return posterior, log_normalizer


class _ForwardEliminationCarry(typing.NamedTuple):
    r"""Effective precision Cholesky factor and offset $J^{-1}h$ at one step."""

    cholesky: jax.Array
    mean_offset: jax.Array

    @classmethod
    def from_canonical(
        cls,
        precision: jax.Array,
        information: jax.Array,
    ) -> typing.Self:
        """Construct the elimination carry from effective canonical parameters."""
        cholesky = jnp.linalg.cholesky(precision)

        return cls(
            cholesky=cholesky,
            mean_offset=_solve_from_cholesky(
                cholesky,
                information,
            ),
        )

    def eliminate_next(
        self,
        lower_precision: jax.Array,
        next_precision: jax.Array,
        next_information: jax.Array,
    ) -> _ForwardEliminationResult:
        r"""Eliminate $x_t$ and retain its conditional mean as a function of $x_{t+1}$."""
        mean_coefficient = -_solve_from_cholesky(
            self.cholesky,
            lower_precision.T,
        )

        conditional_mean = Affine(
            coefficients=mean_coefficient,
            bias=self.mean_offset,
        )

        effective_precision = next_precision + lower_precision @ mean_coefficient

        effective_information = next_information - lower_precision @ self.mean_offset

        next_carry = self.__class__.from_canonical(
            effective_precision,
            effective_information,
        )

        return _ForwardEliminationResult(
            next_carry=next_carry,
            conditional_mean=conditional_mean,
        )


class _ForwardEliminationResult(typing.NamedTuple):
    """Next elimination carry and the eliminated state's conditional mean map."""

    next_carry: _ForwardEliminationCarry
    conditional_mean: Affine


class _GaussianChainFactorization(typing.NamedTuple):
    """State produced by sequential Gaussian-chain block elimination."""

    precision_cholesky_factors: jax.Array
    conditional_mean: Affine
    terminal_mean: jax.Array  # (D,)

    def backward_means(self) -> jax.Array:
        """Compute marginal means by backward recursion."""

        def step(next_mean, conditional_mean):
            mean = conditional_mean.apply(next_mean)
            return mean, mean

        _, means_rest = jax.lax.scan(
            step,
            self.terminal_mean,
            self.conditional_mean,
            reverse=True,
        )

        return jnp.concatenate(
            [
                means_rest,
                self.terminal_mean[None],
            ],
        )

    def backward_covariances(
        self,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute marginal and adjacent cross-covariances by backward recursion."""
        cholesky_factors = self.precision_cholesky_factors
        coefficients = self.conditional_mean.coefficients

        last_covariance = _inverse_from_cholesky(cholesky_factors[-1])

        def step(next_covariance, inputs):
            cholesky, coefficient = inputs

            conditional_covariance = _inverse_from_cholesky(cholesky)

            covariance = (
                conditional_covariance + coefficient @ next_covariance @ coefficient.T
            )

            cross_covariance = coefficient @ next_covariance

            return covariance, (
                covariance,
                cross_covariance,
            )

        _, (covariances_rest, cross_covariances) = jax.lax.scan(
            step,
            last_covariance,
            (
                cholesky_factors[:-1],
                coefficients,
            ),
            reverse=True,
        )

        covariances = jnp.concatenate(
            [
                covariances_rest,
                last_covariance[None],
            ],
        )

        return covariances, cross_covariances


def _log_det(covariance: jax.Array) -> jax.Array:
    """Compute a positive-definite covariance's log determinant via Cholesky."""
    cholesky = jnp.linalg.cholesky(covariance)

    return 2.0 * jnp.sum(
        jnp.log(
            jnp.diagonal(
                cholesky,
                axis1=-2,
                axis2=-1,
            )
        ),
        axis=-1,
    )


def _conditional_log_det(
    covariance: jax.Array,
    next_covariance: jax.Array,
    cross_covariance: jax.Array,
) -> jax.Array:
    r"""Compute $\log\det\operatorname{Cov}(x_{t+1}\mid x_t)$ from pair moments.

    `cross_covariance` stores $\operatorname{Cov}(x_t,x_{t+1})$.
    """
    cholesky = jnp.linalg.cholesky(
        covariance,
    )

    solved = jsp_linalg.cho_solve(
        (cholesky, True),
        cross_covariance,
    )

    conditional_covariance = (
        next_covariance - jnp.swapaxes(cross_covariance, -1, -2) @ solved
    )

    # Remove small numerical asymmetries before Cholesky.
    conditional_covariance = 0.5 * (
        conditional_covariance + jnp.swapaxes(conditional_covariance, -1, -2)
    )

    return _log_det(
        conditional_covariance,
    )


class GaussianChainMarginals(typing.NamedTuple):
    r"""
    Marginal moments of the normalized Gaussian chain distribution.

    Under the normalized chain distribution $q(x)$:

    * `means[t]` stores $\mathbb{E}_q[x_t]$.
    * `covariances[t]` stores $\operatorname{Cov}_q(x_t)$.
    * `cross_covariances[t]` stores $\operatorname{Cov}_q(x_t,x_{t+1})$.
    """

    means: jax.Array  # (*B, T, D)
    covariances: jax.Array  # (*B, T, D, D)
    cross_covariances: jax.Array  # (*B, T - 1, D, D)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading independent-chain dimensions."""
        return self.means.shape[:-2]

    @property
    def num_steps(self) -> int:
        """Number of time steps."""
        return self.means.shape[-2]

    @property
    def variable_dim(self) -> int:
        """Dimension of each Gaussian variable."""
        return self.means.shape[-1]

    def entropy(self) -> jax.Array:
        """Entropy per chain, preserving the batch prefix."""
        if self.num_steps == 0:
            return jnp.zeros(self.batch_shape, dtype=self.means.dtype)
        initial_log_det = _log_det(self.covariances[..., 0, :, :])
        conditional_log_dets = _conditional_log_det(
            self.covariances[..., :-1, :, :],
            self.covariances[..., 1:, :, :],
            self.cross_covariances,
        )
        joint_log_det = initial_log_det + jnp.sum(conditional_log_dets, axis=-1)
        joint_dim = self.num_steps * self.variable_dim
        return 0.5 * (joint_dim * (1.0 + jnp.log(2.0 * jnp.pi)) + joint_log_det)

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = _batch.selection(index, len(self.batch_shape))
        return self.__class__(
            means=self.means[index],
            covariances=self.covariances[index],
            cross_covariances=self.cross_covariances[index],
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            means=_batch.broadcast_array(self.means, shape, axis),
            covariances=_batch.broadcast_array(self.covariances, shape, axis),
            cross_covariances=_batch.broadcast_array(
                self.cross_covariances, shape, axis
            ),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            means=jnp.squeeze(self.means, axis=axes),
            covariances=jnp.squeeze(self.covariances, axis=axes),
            cross_covariances=jnp.squeeze(self.cross_covariances, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            means=jnp.take(self.means, permutation, axis=axis),
            covariances=jnp.take(self.covariances, permutation, axis=axis),
            cross_covariances=jnp.take(self.cross_covariances, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            means=jnp.moveaxis(self.means, source, destination),
            covariances=jnp.moveaxis(self.covariances, source, destination),
            cross_covariances=jnp.moveaxis(self.cross_covariances, source, destination),
        )

    def raw_second_moments(self) -> jax.Array:
        r"""Return raw second moments $\mathbb{E}_q[x_t x_t^\top]$."""
        extra = jnp.einsum(
            '...ti,...tj->...tij',
            self.means,
            self.means,
        )

        return self.covariances + extra

    def raw_cross_moments(self) -> jax.Array:
        r"""Return raw cross moments $\mathbb{E}_q[x_t x_{t+1}^\top]$ for $T-1$ pairs."""
        extra = jnp.einsum(
            '...ti,...tj->...tij',
            self.means[..., :-1, :],
            self.means[..., 1:, :],
        )

        return self.cross_covariances + extra

    def paired_marginals(self) -> PairedGaussian:
        """Return adjacent marginals $(x_t, x_{t+1})$."""
        return PairedGaussian(
            left=Gaussian(
                mean=self.means[..., :-1, :],
                covariance=self.covariances[..., :-1, :, :],
            ),
            right=Gaussian(
                mean=self.means[..., 1:, :],
                covariance=self.covariances[..., 1:, :, :],
            ),
            cross_covariance=jnp.swapaxes(
                self.cross_covariances,
                -2,
                -1,
            ),
        )

    def expected_log_potential(
        self,
        chain: GaussianChain,
    ) -> jax.Array:
        r"""Compute the expected chain log potential $\mathbb{E}_q[\log f(x)]$."""
        _batch.require_same(self.batch_shape, chain.batch_shape)
        if chain.num_steps != self.num_steps:
            raise ValueError('chain and posterior must have the same number of steps')

        if chain.variable_dim != self.means.shape[-1]:
            raise ValueError('chain and posterior variable dimensions must match')

        second_moments = self.raw_second_moments()
        cross_moments = self.raw_cross_moments()

        unary_quadratic = jnp.einsum(
            '...tij,...tij->...',
            chain.diagonal_precision_blocks,
            second_moments,
        )

        pair_quadratic = jnp.einsum(
            '...tij,...tji->...',
            chain.lower_precision_blocks,
            cross_moments,
        )

        linear = jnp.einsum(
            '...ti,...ti->...',
            chain.information_vectors,
            self.means,
        )

        return -0.5 * unary_quadratic - pair_quadratic + linear + chain.log_constant


def _solve_from_cholesky(
    cholesky_factor: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """Solve an SPD system from its lower-triangular Cholesky factor."""
    intermediate = jsp_linalg.solve_triangular(
        cholesky_factor,
        rhs,
        lower=True,
    )

    return jsp_linalg.solve_triangular(
        cholesky_factor.T,
        intermediate,
        lower=False,
    )


def _inverse_from_cholesky(
    cholesky_factor: jax.Array,
) -> jax.Array:
    """Compute an SPD inverse from its Cholesky factor."""
    identity = jnp.eye(
        cholesky_factor.shape[0],
        dtype=cholesky_factor.dtype,
    )

    return _solve_from_cholesky(
        cholesky_factor,
        identity,
    )


def _log_det_from_cholesky(
    cholesky_factor: jax.Array,
) -> jax.Array:
    """Compute an SPD log determinant from its Cholesky factor."""
    return 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky_factor)))
