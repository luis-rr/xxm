"""Gaussian distributions, conditionals, and paired moment representations."""

import typing

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.data import Sequences
from xxm.core.dists.additive import AffineDist, LocDist
from xxm.core.lengths import Lengths


class FullGaussianNoise(typing.NamedTuple):
    r"""Zero-mean Gaussian noise.

    $$\epsilon \sim \mathcal{N}(0, \Sigma).$$

    `covariance` stores $\Sigma$. Leading dimensions are batch dimensions.
    """

    covariance: jax.Array  # (..., N, N)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.covariance.shape[:-2]

    @property
    def variable_dim(self) -> int:
        """Dimension of the noise variable."""
        return self.covariance.shape[-1]

    @property
    def dtype(self) -> jnp.dtype:
        """Data type."""
        return self.covariance.dtype

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

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        """Convert to a different data type."""
        return self._replace(covariance=self.covariance.astype(dtype))

    @classmethod
    def identity(
        cls,
        dim: int,
        batch_shape: tuple[int, ...] = (),
        dtype: jax.typing.DTypeLike | None = None,
    ) -> typing.Self:
        """Unit noise with identity covariance and batch `batch_shape`."""
        return cls(
            jnp.broadcast_to(jnp.eye(dim, dtype=dtype), batch_shape + (dim, dim))
        )

    def permute_variables(self, permutation: jax.Array) -> typing.Self:
        """Reorder variable dimensions globally or independently by batch."""
        permutation = batch.broadcast_trailing(
            permutation,
            self.batch_shape,
            (self.variable_dim,),
        )

        row_indices = jnp.broadcast_to(
            permutation[..., :, None],
            self.covariance.shape,
        )

        covariance = jnp.take_along_axis(
            self.covariance,
            row_indices,
            axis=-2,
        )

        column_indices = jnp.broadcast_to(
            permutation[..., None, :],
            self.covariance.shape,
        )

        return self._replace(
            covariance=jnp.take_along_axis(covariance, column_indices, axis=-1)
        )

    def reorient_variables(self, signs: jax.Array) -> typing.Self:
        """Flip variable signs globally or independently by batch."""
        signs = batch.broadcast_trailing(
            signs,
            self.batch_shape,
            (self.variable_dim,),
        )
        return self._replace(
            covariance=signs[..., :, None] * self.covariance * signs[..., None, :]
        )

    def with_floor(
        self,
        floor: float,
        reference: 'FullGaussianNoise | None' = None,
    ) -> typing.Self:
        """Symmetrize and add an isotropic floor relative to a reference scale.

        `floor` is a dimensionless fraction of the average marginal variance of
        `reference`, which defaults to this noise.
        """
        reference = self if reference is None else reference

        covariance = 0.5 * (self.covariance + jnp.swapaxes(self.covariance, -2, -1))

        scale = jnp.mean(reference.variance, axis=-1)
        scale = jnp.where(scale > 0.0, scale, 1.0)

        identity = jnp.eye(self.variable_dim, dtype=covariance.dtype)

        return self._replace(
            covariance=covariance + floor * scale[..., None, None] * identity
        )

    def whiten(
        self,
        values: jax.Array,  # (*B, *Q, N)
    ) -> jax.Array:  # (*B, *Q, N)
        r"""Map values to $L^{-1}\epsilon$, where $LL^\top = \Sigma$.

        Query axes `*Q` follow the receiver batch. Each batch entry is factored
        once and its query points are solved as columns of one system, so $L$
        is never broadcast to every point.
        """
        if values.shape[-1:] != (self.variable_dim,):
            raise ValueError('values must match the noise variable dimension')

        chol = jnp.linalg.cholesky(self.covariance)  # (*B, N, N)

        columns = jnp.swapaxes(
            batch.flatten_query(values, self.batch_shape),
            -1,
            -2,
        )  # (*B, N, Q)

        solved = jsp_linalg.solve_triangular(chol, columns, lower=True)

        return jnp.swapaxes(solved, -1, -2).reshape(values.shape)

    def log_det(self) -> jax.Array:  # (*B,)
        r"""Log determinant $\log|\Sigma|$."""
        chol = jnp.linalg.cholesky(self.covariance)
        return 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)

    @property
    def variance(self) -> jax.Array:  # (*B, N)
        r"""Marginal variances, the diagonal of $\Sigma$."""
        return jnp.diagonal(self.covariance, axis1=-2, axis2=-1)

    def precision(self) -> jax.Array:  # (*B, N, N)
        r"""Precision $\Sigma^{-1}$."""
        chol = jnp.linalg.cholesky(self.covariance)
        identity = jnp.broadcast_to(
            jnp.eye(self.variable_dim, dtype=self.covariance.dtype),
            self.covariance.shape,
        )
        return jsp_linalg.cho_solve((chol, True), identity)

    def entropy(self) -> jax.Array:  # (*B,)
        """Differential entropy."""
        return 0.5 * (
            self.variable_dim * (1.0 + jnp.log(2.0 * jnp.pi)) + self.log_det()
        )

    def log_prob(
        self,
        values: jax.Array,  # (*B, *Q, N)
    ) -> jax.Array:  # (*B, *Q)
        """Evaluate aligned values, with query axes following the receiver batch."""
        mahalanobis = jnp.sum(self.whiten(values) ** 2, axis=-1)
        log_det = batch.align_array(self.log_det(), self.batch_shape, values.shape[:-1])

        return -0.5 * (
            self.variable_dim * jnp.log(2.0 * jnp.pi) + log_det + mahalanobis
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Sample values with shape `(*batch_shape, *sample_shape, N)`."""
        shape = self.batch_shape + (1,) * len(sample_shape)
        return jax.random.multivariate_normal(
            key,
            mean=jnp.zeros(shape + (self.variable_dim,), dtype=self.dtype),
            cov=self.covariance.reshape(shape + (self.variable_dim, self.variable_dim)),
            shape=self.batch_shape + sample_shape,
        )

    def congruence(self, matrix: jax.Array) -> 'FullGaussianNoise':  # (..., M, N)
        r"""Noise of $M\epsilon$, with covariance $M\Sigma M^\top$."""
        return FullGaussianNoise(
            jnp.einsum('...oi,...ij,...pj->...op', matrix, self.covariance, matrix)
        )

    def expected_log_prob(self, other: 'FullGaussian') -> jax.Array:
        r"""
        Expected log density $\mathbb{E}_{\epsilon\sim\text{other}}[\log p(\epsilon)]$.

        The two objects must have aligned batches.
        """
        mahalanobis = jnp.sum(self.whiten(other.mean) ** 2, axis=-1)

        cholesky = jnp.linalg.cholesky(self.covariance)
        trace = jnp.trace(
            jsp_linalg.cho_solve((cholesky, True), other.covariance),
            axis1=-2,
            axis2=-1,
        )

        return -0.5 * (
            self.variable_dim * jnp.log(2.0 * jnp.pi)
            + self.log_det()
            + mahalanobis
            + trace
        )


class FullGaussian(LocDist[FullGaussianNoise]):
    r"""Multivariate Gaussian distribution in moment form.

    $$u \sim \mathcal{N}(\mu, \Sigma).$$

    `loc` stores $\mu$ and `noise` is $\mathcal{N}(0, \Sigma)$; `mean` and
    `covariance` expose them. Leading dimensions are batch dimensions, shared
    across all attributes.
    """

    __slots__ = ()

    @classmethod
    def from_moments(cls, mean: jax.Array, covariance: jax.Array) -> typing.Self:
        """Construct from a mean `(..., N)` and a covariance `(..., N, N)`."""
        return cls(mean, FullGaussianNoise(covariance))

    @property
    def mean(self) -> jax.Array:  # (..., N)
        r"""Mean $\mu$."""
        return self.loc

    @property
    def covariance(self) -> jax.Array:  # (..., N, N)
        r"""Covariance $\Sigma$."""
        return self.noise.covariance

    @classmethod
    def from_canonical(
        cls, precision: jax.Array, information: jax.Array
    ) -> typing.Self:
        """Normalize canonical parameters with matching leading batch dimensions."""
        if precision.shape != information.shape + (information.shape[-1],):
            raise ValueError(
                'precision and information must have matching batch and variable dimensions'
            )
        precision = 0.5 * (precision + jnp.swapaxes(precision, -1, -2))
        cholesky = jnp.linalg.cholesky(precision)
        identity = jnp.broadcast_to(
            jnp.eye(precision.shape[-1], dtype=precision.dtype), precision.shape
        )
        mean = jsp_linalg.cho_solve((cholesky, True), information[..., None])[..., 0]
        covariance = jsp_linalg.cho_solve((cholesky, True), identity)
        covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
        return cls.from_moments(mean, covariance)

    @property
    def variance(self) -> jax.Array:
        """Marginal variances, diagonal of covariance."""
        return self.noise.variance

    def _validate_affine_input(
        self,
        affine: Affine,
    ) -> None:
        batch.require_same_shape(self.batch_shape, affine.batch_shape)

        if affine.input_shape != (self.variable_dim,):
            raise ValueError(
                'Gaussian affine transformations require vector-shaped inputs; '
                f'expected input shape {(self.variable_dim,)}, '
                f'got {affine.input_shape}'
            )

    def affine_mean(
        self,
        affine: Affine,
    ) -> jax.Array:
        r"""Compute mean of $f(u)$ where $u \sim \text{self}$."""
        self._validate_affine_input(affine)
        return affine.apply(self.mean)

    def affine_covariance(
        self,
        affine: Affine,
    ) -> jax.Array:
        r"""Compute covariance of $f(u)$ where $u \sim \text{self}$."""
        self._validate_affine_input(affine)
        return self.noise.congruence(affine.coefficients).covariance

    def affine_variance(
        self,
        affine: Affine,
    ) -> jax.Array:
        r"""Compute marginal variances of $f(u)$ where $u \sim \text{self}$."""
        self._validate_affine_input(affine)

        return jnp.einsum(
            '...oi,...ij,...oj->...o',
            affine.coefficients,
            self.covariance,
            affine.coefficients,
        )

    def affine(
        self,
        affine: Affine,
    ) -> 'FullGaussian':
        r"""Distribution of $f(u)$ for $u \sim \text{self}$."""
        return FullGaussian(
            self.affine_mean(affine),
            self.noise.congruence(affine.coefficients),
        )

    def whiten(
        self,
        values: jax.Array,  # (*B, *Q, N)
    ) -> jax.Array:  # (*B, *Q, N)
        r"""Map values to $L^{-1}(u - \mu)$, where $LL^\top = \Sigma$."""
        return self.noise.whiten(self.residual(values))

    def expected_log_prob(
        self,
        other: 'FullGaussian',
    ) -> jax.Array:
        r"""
        Expected log density $\mathbb{E}_{u\sim\text{other}}[\log p(u)]$.

        The two Gaussian objects must have aligned batches.
        """
        batch.require_same_shape(self.batch_shape, other.batch_shape)
        if other.mean.shape[-1] != self.variable_dim:
            raise ValueError(
                f'mean must have trailing dimension {self.variable_dim}; '
                f'got {other.mean.shape}'
            )

        if other.covariance.shape[-2:] != (
            self.variable_dim,
            self.variable_dim,
        ):
            raise ValueError(
                'covariance must have trailing shape '
                f'{(self.variable_dim, self.variable_dim)}; '
                f'got {other.covariance.shape}'
            )

        return self.noise.expected_log_prob(other._replace(loc=other.mean - self.mean))


class FullLinearGaussian(AffineDist[FullGaussianNoise]):
    r"""Linear-Gaussian conditional distribution.

    $$v \mid u \sim \mathcal{N}(Wu + b, \Sigma).$$

    `affine` encodes $(W, b)$ and `noise` is $\mathcal{N}(0, \Sigma)$.
    Tensor-shaped inputs resolve as tensor contractions. Leading dimensions are
    batch dimensions, shared across all attributes.
    """

    __slots__ = ()

    @classmethod
    def from_components(
        cls,
        coefficients: jax.Array,  # (..., O, *input_shape)
        bias: jax.Array,  # (..., O)
        covariance: jax.Array,  # (..., O, O)
    ) -> typing.Self:
        """Construct from affine coefficients, bias, and a noise covariance."""
        return cls(Affine(coefficients, bias), FullGaussianNoise(covariance))

    @property
    def covariance(self) -> jax.Array:  # (..., O, O)
        r"""Noise covariance $\Sigma$."""
        return self.noise.covariance

    def conditional(
        self,
        values: jax.Array,  # (..., *input_shape)
    ) -> FullGaussian:
        """Conditional distribution at deterministic input."""
        mean = self.conditional_mean(values)  # (..., O)

        covariance = jnp.broadcast_to(
            batch.align_array(self.covariance, self.batch_shape, mean.shape[:-1]),
            mean.shape[:-1] + (self.output_dim, self.output_dim),
        )  # (..., O, O)

        return FullGaussian.from_moments(mean, covariance)

    def compose_output(
        self,
        affine: Affine,
    ) -> typing.Self:
        """
        Postcompose the output with $w=W'v+b'$.
        """
        if affine.input_shape != (self.output_dim,):
            raise ValueError(
                'output affine input must match the model output dimension; '
                f'expected {(self.output_dim,)}, got {affine.input_shape}'
            )

        return self.__class__(
            affine=affine.compose(self.affine),
            noise=self.noise.congruence(affine.coefficients_flat),
        )

    def expected_log_prob(
        self,
        input: FullGaussian,
        output: FullGaussian,
        input_output_covariance: jax.Array,
    ) -> jax.Array:
        r"""Compute $\mathbb{E}_q[\log p(v\mid u)]$ from joint input-output moments.

        `input_output_covariance` stores $\operatorname{Cov}_q(u,v)$ with
        trailing shape $(I,O)$; tensor-shaped inputs are flattened.

        The model, input Gaussian, output Gaussian, and cross-covariance must
        have aligned batches.
        """
        input_mean_flat = self.affine.flatten_input(
            input.mean,
        )

        shape = input_mean_flat.shape[:-1]
        batch.require_same_shape(
            shape,
            self.batch_shape,
            input.covariance.shape[:-2],
            output.batch_shape,
        )
        if input.covariance.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError('input covariance must match the flattened input size')
        if (
            output.variable_dim != self.output_dim
            or input_output_covariance.shape
            != shape + (self.input_size, self.output_dim)
        ):
            raise ValueError(
                'output and cross covariance must match the conditional dimensions'
            )
        coefficients = self.affine.coefficients_flat
        bias = self.affine.bias

        residual_mean = (
            output.mean
            - jnp.einsum(
                '...oi,...i->...o',
                coefficients,
                input_mean_flat,
            )
            - bias
        )

        projected_input_covariance = jnp.einsum(
            '...oi,...ij,...pj->...op',
            coefficients,
            input.covariance,
            coefficients,
        )

        projected_cross_covariance = jnp.einsum(
            '...oi,...ip->...op',
            coefficients,
            input_output_covariance,
        )

        residual_covariance = (
            output.covariance
            + projected_input_covariance
            - projected_cross_covariance
            - jnp.swapaxes(
                projected_cross_covariance,
                -1,
                -2,
            )
        )

        # Remove insignificant asymmetry introduced by floating-point arithmetic.
        residual_covariance = 0.5 * (
            residual_covariance
            + jnp.swapaxes(
                residual_covariance,
                -1,
                -2,
            )
        )

        return self.noise.expected_log_prob(
            FullGaussian.from_moments(residual_mean, residual_covariance)
        )

    def expected_log_prob_broadcast(
        self,
        input: FullGaussian,
        output: FullGaussian,
        input_output_covariance: jax.Array,
    ) -> jax.Array:
        """Evaluate joint moments with receiver batch axes first.

        Cross covariance stores Cov(input, output), with trailing shape (I, O).
        """
        # Gaussian input means may retain the conditional's tensor input shape.
        input = FullGaussian.from_moments(
            self.affine.flatten_input(input.mean), input.covariance
        )
        query_shape = input.batch_shape
        batch.require_same_shape(
            query_shape,
            output.batch_shape,
            input_output_covariance.shape[:-2],
        )

        model = self.broadcast(
            query_shape,
            axis=len(self.batch_shape),
        )
        input = input.broadcast(self.batch_shape)
        output = output.broadcast(self.batch_shape)
        input_output_covariance = jnp.broadcast_to(
            input_output_covariance,
            self.batch_shape + input_output_covariance.shape,
        )
        if self.input_ndim != 1:
            input = input._replace(loc=self.affine.unflatten_input(input.mean))
        return model.expected_log_prob(
            input,
            output,
            input_output_covariance,
        )


GaussianNoise = FullGaussianNoise
"""Gaussian noise of any covariance structure."""

Gaussian = FullGaussian
"""A Gaussian of any covariance structure."""

LinearGaussian = FullLinearGaussian
"""A linear-Gaussian conditional with noise of any covariance structure."""


class PairedGaussian(typing.NamedTuple):
    r"""Joint Gaussian distribution over pair of random variables.

    `left` and `right` are marginal distributions. `cross_covariance` is
    $\operatorname{Cov}(\text{right}, \text{left})$. Leading dimensions are batch dimensions,
    shared across all attributes.
    """

    left: FullGaussian
    right: FullGaussian
    cross_covariance: jax.Array  # (..., R, L)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        left_shape = self.left.batch_shape
        right_shape = self.right.batch_shape
        cross_shape = self.cross_covariance.shape[:-2]

        batch.require_same_shape(left_shape, right_shape)
        batch.require_same_shape(left_shape, cross_shape)
        assert self.cross_covariance.shape[-2:] == (
            self.right_dim,
            self.left_dim,
        )

        return left_shape

    @property
    def left_dim(self) -> int:
        """Dimension of left variable."""
        return self.left.variable_dim

    @property
    def right_dim(self) -> int:
        """Dimension of right variable."""
        return self.right.variable_dim

    @property
    def variable_dim(self) -> int:
        """Total dimension of concatenated pair."""
        return self.left_dim + self.right_dim

    @property
    def dtype(self) -> jnp.dtype:
        """Common dtype implied by both marginals and the cross-covariance."""
        return jnp.result_type(
            self.left.dtype,
            self.right.dtype,
            self.cross_covariance,
        )

    @property
    def mean(self) -> jax.Array:
        r"""Mean of concatenated variable $[\text{left}, \text{right}]$."""
        return jnp.concatenate([self.left.mean, self.right.mean], axis=-1)

    @property
    def covariance(self) -> jax.Array:
        r"""Covariance of concatenated variable $[\text{left}, \text{right}]$."""
        left_right_covariance = jnp.swapaxes(self.cross_covariance, -2, -1)

        top = jnp.concatenate([self.left.covariance, left_right_covariance], axis=-1)

        bottom = jnp.concatenate(
            [self.cross_covariance, self.right.covariance], axis=-1
        )

        return jnp.concatenate([top, bottom], axis=-2)

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        """Convert to a different data type."""
        return self._replace(
            left=self.left.astype(dtype),
            right=self.right.astype(dtype),
            cross_covariance=self.cross_covariance.astype(dtype),
        )

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


# TODO: Consider a generic `TemporalDist[D]` over any distribution, which would
# also cover `TemporalPoisson`.
class TemporalGaussian(typing.NamedTuple):
    """Time-indexed Gaussian distributions with contiguous valid prefixes.

    dist.mean:        (*B, T, *C, D)
    dist.covariance:  (*B, T, *C, D, D)
    lengths.values:   (*B,)

    *B contains structural batch dimensions.
    T is time.
    *C contains additional Gaussian batch dimensions, e.g. states.
    D is the Gaussian variable dimension.

    Generic batch operations affect only *B.
    """

    dist: FullGaussian
    lengths: Lengths

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.lengths.batch_shape

    @property
    def time_axis(self) -> int:
        return len(self.batch_shape)

    @property
    def max_steps(self) -> int:
        return self.dist.batch_shape[self.time_axis]

    @property
    def variable_dim(self) -> int:
        return self.dist.variable_dim

    def validate(self) -> None:
        """Validate static shapes without inspecting traced lengths."""
        if self.dist.mean.ndim < len(self.batch_shape) + 2:
            raise ValueError('Gaussian mean must have shape (*B, T, *C, D)')

        trailing = batch.split_prefix(
            self.dist.batch_shape,
            self.batch_shape,
            name='Gaussian batch shape',
        )

        if not trailing:
            raise ValueError('TemporalGaussian requires a time axis')

        expected = self.dist.mean.shape + (self.variable_dim,)
        if self.dist.covariance.shape != expected:
            raise ValueError(
                f'Gaussian covariance must have shape {expected}; '
                f'got {self.dist.covariance.shape}'
            )

        if not jnp.issubdtype(self.lengths.values.dtype, jnp.integer):
            raise ValueError('lengths must have an integer dtype')

    @property
    def mean(self) -> Sequences:
        """Gaussian means, with padded timesteps cleared."""
        return Sequences.from_padded(
            self.dist.mean,
            self.lengths,
        )

    @property
    def covariance(self) -> Sequences:
        """Gaussian covariance matrices, with padded timesteps cleared."""
        return Sequences.from_padded(
            self.dist.covariance,
            self.lengths,
        )

    @property
    def variance(self) -> Sequences:
        """Marginal variances, with padded timesteps cleared."""
        return Sequences.from_padded(
            self.dist.variance,
            self.lengths,
        )

    @property
    def std(self) -> Sequences:
        """Marginal standard deviations, with padded timesteps cleared."""
        return Sequences.from_padded(
            jnp.sqrt(self.dist.variance),
            self.lengths,
        )

    def select(self, index: batch.SelT) -> typing.Self:
        """Select only structural batch dimensions."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton structural batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along one structural batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        """Move an axis within the structural batch prefix."""
        return batch.move_axis(self, source, destination)

    def select_time(
        self,
        *,
        start: int,
        stop: int | None = None,
    ) -> typing.Self:
        """Select a time interval, adjusting valid-prefix lengths."""
        start, stop, _ = slice(start, stop).indices(self.max_steps)

        new_lengths = Lengths(
            jnp.clip(
                self.lengths.values - start,
                0,
                max(stop - start, 0),
            )
        )

        index = (slice(None),) * self.time_axis + (slice(start, stop),)

        return type(self)(
            dist=self.dist.select(index),
            lengths=new_lengths,
        )

    def get(self, index: batch.SelT | None = None) -> typing.Self:
        """Retrieve exactly one unbatched item."""
        return batch.get(self, index)


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
    lengths: Lengths

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading independent-chain dimensions."""
        return self.means.shape[:-2]

    @property
    def max_steps(self) -> int:
        """Number of time steps."""
        return self.means.shape[-2]

    @property
    def variable_dim(self) -> int:
        """Dimension of each Gaussian variable."""
        return self.means.shape[-1]

    def entropy_terms(self) -> jax.Array:
        r"""Return Markov entropy contributions for each time step.

        The returned terms are

        $$
        H(x_0),\;
        H(x_1 \mid x_0),\;
        \ldots,\;
        H(x_{T-1} \mid x_{T-2}),
        $$

        with shape `(*B, T)`.
        """
        if self.max_steps == 0:
            return jnp.zeros(
                (*self.batch_shape, 0),
                dtype=self.means.dtype,
            )

        constant = self.variable_dim * (1.0 + jnp.log(2.0 * jnp.pi))

        initial_log_det = _log_det(self.covariances[..., 0, :, :])

        conditional_log_dets = _conditional_log_det(
            self.covariances[..., :-1, :, :],
            self.covariances[..., 1:, :, :],
            self.cross_covariances,
        )

        initial = 0.5 * (constant + initial_log_det)

        conditionals = 0.5 * (constant + conditional_log_dets)

        return jnp.concatenate(
            [
                initial[..., None],
                conditionals,
            ],
            axis=-1,
        )

    def entropy(
        self,
    ) -> jax.Array:
        """Entropy per chain over its genuine temporal prefix."""
        terms = self.entropy_terms()
        terms = self.lengths.clear(terms)
        return jnp.sum(terms, axis=-1)

    def align(self, alignment: Affine) -> typing.Self:
        """Express posterior marginals in aligned latent coordinates."""
        return self.affine(alignment)

    def affine(self, affine: Affine) -> typing.Self:
        """Push posterior marginals through a batch-aligned affine map."""
        batch.require_same_shape(self.batch_shape, affine.batch_shape)

        if affine.input_shape != (self.variable_dim,):
            raise ValueError(
                'affine input dimension must match the posterior variable dimension'
            )

        coefficients = affine.coefficients
        means = (
            jnp.einsum('...oi,...ti->...to', coefficients, self.means)
            + affine.bias[..., None, :]
        )
        covariances = jnp.einsum(
            '...oi,...tij,...pj->...top',
            coefficients,
            self.covariances,
            coefficients,
        )
        cross_covariances = jnp.einsum(
            '...oi,...tij,...pj->...top',
            coefficients,
            self.cross_covariances,
            coefficients,
        )

        return self.__class__(
            lengths=self.lengths,
            means=means,
            covariances=covariances,
            cross_covariances=cross_covariances,
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

    def raw_second_moments(self) -> jax.Array:
        r"""Return raw second moments $\mathbb{E}_q[x_t x_t^\top]$."""
        extra = jnp.einsum(
            '...ti,...tj->...tij',
            self.means,
            self.means,
        )

        return self.covariances + extra

    def reorient_variables(
        self,
        signs: jax.Array,
    ) -> typing.Self:
        """Flip variable signs globally or independently by chain batch."""
        signs = batch.broadcast_trailing(
            signs,
            self.batch_shape,
            (self.variable_dim,),
        )

        return self._replace(
            means=(self.means * signs[..., None, :]),
            covariances=(
                signs[..., None, :, None] * self.covariances * signs[..., None, None, :]
            ),
            cross_covariances=(
                signs[..., None, :, None]
                * self.cross_covariances
                * signs[..., None, None, :]
            ),
        )

    def permute_variables(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel variable dimensions globally or independently by chain batch."""
        permutation = batch.broadcast_trailing(
            permutation,
            self.batch_shape,
            (self.variable_dim,),
        )

        mean_indices = jnp.broadcast_to(permutation[..., None, :], self.means.shape)

        means = jnp.take_along_axis(self.means, mean_indices, axis=-1)

        covariance_row_indices = jnp.broadcast_to(
            permutation[..., None, :, None],
            self.covariances.shape,
        )

        covariances = jnp.take_along_axis(
            self.covariances,
            covariance_row_indices,
            axis=-2,
        )

        covariance_column_indices = jnp.broadcast_to(
            permutation[..., None, None, :],
            self.covariances.shape,
        )

        covariances = jnp.take_along_axis(
            covariances,
            covariance_column_indices,
            axis=-1,
        )

        cross_row_indices = jnp.broadcast_to(
            permutation[..., None, :, None],
            self.cross_covariances.shape,
        )

        cross_covariances = jnp.take_along_axis(
            self.cross_covariances,
            cross_row_indices,
            axis=-2,
        )

        cross_column_indices = jnp.broadcast_to(
            permutation[..., None, None, :],
            self.cross_covariances.shape,
        )

        cross_covariances = jnp.take_along_axis(
            cross_covariances,
            cross_column_indices,
            axis=-1,
        )

        return self._replace(
            means=means,
            covariances=covariances,
            cross_covariances=cross_covariances,
        )

    def shift_variables(
        self,
        offset: jax.Array,
    ) -> typing.Self:
        """Shift all chain-variable coordinates as x_t' = x_t - offset."""
        offset = batch.broadcast_trailing(
            offset,
            self.batch_shape,
            (self.variable_dim,),
        )

        return self._replace(
            means=(self.means - offset[..., None, :]),
        )

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
            left=FullGaussian.from_moments(
                self.means[..., :-1, :],
                self.covariances[..., :-1, :, :],
            ),
            right=FullGaussian.from_moments(
                self.means[..., 1:, :],
                self.covariances[..., 1:, :, :],
            ),
            cross_covariance=jnp.swapaxes(
                self.cross_covariances,
                -2,
                -1,
            ),
        )


class GaussianMarginals(typing.Protocol):
    r"""Protocol for continuous Gaussian marginals.

    Represents marginal means, covariances, and raw moments across time under
    a Gaussian distribution $q(x)$.

    It may hold, for instance, the posterior distribution after inference.

    Independent batch dimensions precede the time and variable dimensions.
    """

    @property
    def lengths(self) -> Lengths: ...

    @property
    def max_steps(self) -> int:
        r"""Number of time steps $T$."""
        ...

    @property
    def variable_dim(self) -> int:
        r"""Dimension of the variables $D$."""
        ...

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""Shape of independent batch dimensions (*B)."""
        ...

    @property
    def means(self) -> jax.Array:  # (*B, T, D)
        r"""Marginal means $\mathbb{E}_q[x_t]$."""
        ...

    @property
    def covariances(self) -> jax.Array:  # (*B, T, D, D)
        r"""Marginal covariances $\operatorname{Cov}_q(x_t)$."""
        ...

    @property
    def cross_covariances(self) -> jax.Array:  # (*B, T-1, D, D)
        r"""Cross-covariances $\operatorname{Cov}_q(x_t,x_{t+1})$."""
        ...

    def raw_second_moments(self) -> jax.Array:  # (*B, T, D, D)
        r"""Raw second moments $\mathbb{E}_q[x_t x_t^\top]$."""
        ...

    def raw_cross_moments(self) -> jax.Array:  # (*B, T-1, D, D)
        r"""Raw cross-moments $\mathbb{E}_q[x_t x_{t+1}^\top]$."""
        ...

    def paired_marginals(self) -> PairedGaussian:
        r"""Return adjacent Gaussian marginals $q(x_t, x_{t+1})$."""
        ...
