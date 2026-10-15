"""Gaussian distributions, conditionals, and paired moment representations."""

import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as jsp_linalg

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.data import Sequences
from xxm.core.lengths import Lengths


class Gaussian(typing.NamedTuple):
    r"""Multivariate Gaussian distribution in moment form.

    $$u \sim \mathcal{N}(\mu, \Sigma).$$

    `mean` stores $\mu$ and `covariance` stores $\Sigma$. Leading dimensions
    are batch dimensions, shared across all attributes.
    """

    mean: jax.Array  # (..., N)
    covariance: jax.Array  # (..., N, N)

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
        return cls(mean=mean, covariance=covariance)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        mean_shape = self.mean.shape[:-1]
        covariance_shape = self.covariance.shape[:-2]
        batch.require_same_shape(mean_shape, covariance_shape)
        return mean_shape

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)

    def permute_variables(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Reorder variable dimensions globally or independently by batch."""
        permutation = batch.broadcast_trailing(
            permutation,
            self.batch_shape,
            (self.variable_dim,),
        )

        mean = jnp.take_along_axis(
            self.mean,
            permutation,
            axis=-1,
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

        covariance = jnp.take_along_axis(
            covariance,
            column_indices,
            axis=-1,
        )

        return self._replace(
            mean=mean,
            covariance=covariance,
        )

    def reorient_variables(
        self,
        signs: jax.Array,
    ) -> typing.Self:
        """Flip variable signs globally or independently by batch."""
        signs = batch.broadcast_trailing(
            signs,
            self.batch_shape,
            (self.variable_dim,),
        )

        return self._replace(
            mean=self.mean * signs,
            covariance=(signs[..., :, None] * self.covariance * signs[..., None, :]),
        )

    def shift_variables(
        self,
        offset: jax.Array,
    ) -> typing.Self:
        """Shift variable coordinates as x' = x - offset."""
        offset = batch.broadcast_trailing(
            offset,
            self.batch_shape,
            (self.variable_dim,),
        )

        return self._replace(
            mean=self.mean - offset,
        )

    @property
    def variable_dim(self) -> int:
        """Dimension of the Gaussian variable."""
        return self.mean.shape[-1]

    @property
    def dtype(self) -> jnp.dtype:
        """Data type."""
        return jnp.result_type(self.mean, self.covariance)

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def astype(self, dtype: jax.typing.DTypeLike) -> 'Gaussian':
        """Convert to a different data type."""
        return self._replace(
            mean=self.mean.astype(dtype),
            covariance=self.covariance.astype(dtype),
        )

    @property
    def variance(self) -> jax.Array:
        """Marginal variances, diagonal of covariance."""
        return jnp.diagonal(self.covariance, axis1=-2, axis2=-1)

    def entropy(self) -> jax.Array:
        """Differential entropy of the Gaussian distribution."""
        cholesky = jnp.linalg.cholesky(self.covariance)
        log_det = 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)),
            axis=-1,
        )
        return 0.5 * (self.variable_dim * (1.0 + jnp.log(2.0 * jnp.pi)) + log_det)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Sample values with shape `(*batch_shape, *sample_shape, D)`."""
        shape = self.batch_shape + (1,) * len(sample_shape)
        return jax.random.multivariate_normal(
            key,
            mean=self.mean.reshape(shape + (self.variable_dim,)),
            cov=self.covariance.reshape(shape + (self.variable_dim, self.variable_dim)),
            shape=self.batch_shape + sample_shape,
        )

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

        return jnp.einsum(
            '...oi,...ij,...pj->...op',
            affine.coefficients,
            self.covariance,
            affine.coefficients,
        )

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
    ) -> 'Gaussian':
        r"""Distribution of $f(u)$ for $u \sim \text{self}$."""

        return self.__class__(
            mean=self.affine_mean(affine),
            covariance=self.affine_covariance(affine),
        )

    def log_prob(
        self,
        values: jax.Array,  # (..., N)
    ) -> jax.Array:  # (...)
        """Evaluate aligned values, with query axes following the receiver batch."""
        if values.shape[-1:] != (self.variable_dim,):
            raise ValueError('values must match the Gaussian variable dimension')
        mean = batch.align_array(self.mean, self.batch_shape, values.shape[:-1])
        covariance = batch.align_array(
            self.covariance, self.batch_shape, values.shape[:-1]
        )
        residuals = values - mean  # (..., N)

        chol = jnp.linalg.cholesky(covariance)  # (..., N, N)

        # ``solve_triangular`` requires explicit matching batch dimensions.
        chol = jnp.broadcast_to(
            chol,
            residuals.shape[:-1] + (self.variable_dim, self.variable_dim),
        )

        solved = jsp.linalg.solve_triangular(
            chol,
            residuals[..., None],
            lower=True,
        )[..., 0]  # (..., N)

        mahalanobis = jnp.sum(solved**2, axis=-1)  # (...)

        log_det = 2.0 * jnp.sum(
            jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)),
            axis=-1,
        )  # (...)

        return -0.5 * (
            self.variable_dim * jnp.log(2.0 * jnp.pi) + log_det + mahalanobis
        )

    def log_prob_broadcast(self, values: jax.Array) -> jax.Array:
        """Evaluate all values with receiver batch axes first."""
        values = jnp.broadcast_to(values, self.batch_shape + values.shape)
        return self.log_prob(values)

    def mixture_mean(self, weights: jax.Array, *, axis: int) -> jax.Array:
        """Compute a mixture mean over an explicit, aligned batch axis."""
        return batch.weighted_sum(self.mean, weights, self.batch_shape, axis)

    def expected_log_prob(
        self,
        other: 'Gaussian',
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

        mean = other.mean
        covariance = other.covariance
        model_mean = self.mean
        cholesky = jnp.linalg.cholesky(self.covariance)

        residual = mean - model_mean

        whitened_residual = jsp.linalg.solve_triangular(
            cholesky,
            residual[..., None],
            lower=True,
        )[..., 0]

        mahalanobis = jnp.sum(
            whitened_residual**2,
            axis=-1,
        )

        precision_covariance = jsp.linalg.cho_solve(
            (cholesky, True),
            covariance,
        )

        trace = jnp.trace(
            precision_covariance,
            axis1=-2,
            axis2=-1,
        )

        log_det = 2.0 * jnp.sum(
            jnp.log(
                jnp.diagonal(
                    cholesky,
                    axis1=-2,
                    axis2=-1,
                )
            ),
            axis=-1,
        )

        return -0.5 * (
            self.variable_dim * jnp.log(2.0 * jnp.pi) + log_det + mahalanobis + trace
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


class LinearGaussian(typing.NamedTuple):
    r"""Linear-Gaussian conditional distribution.

    $$v \mid u \sim \mathcal{N}(Wu + b, \Sigma).$$

    `affine` encodes $(W, b)$ and `covariance` stores $\Sigma$.
    Tensor-shaped inputs resolve as tensor contractions. Leading dimensions are
    batch dimensions, shared across all attributes.
    """

    affine: Affine
    covariance: jax.Array  # (..., O, O)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        covariance_shape = self.covariance.shape[:-2]
        affine_shape = self.affine.batch_shape
        batch.require_same_shape(covariance_shape, affine_shape)
        return affine_shape

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Input shape of the affine map."""
        return self.affine.input_shape

    @property
    def input_ndim(self) -> int:
        """Number of input dimensions."""
        return self.affine.input_ndim

    @property
    def input_size(self) -> int:
        """Total size of input."""
        return self.affine.input_size

    @property
    def output_dim(self) -> int:
        """Output dimension."""
        return self.affine.output_dim

    @property
    def dtype(self) -> jnp.dtype:
        """Data type."""
        return jnp.result_type(self.affine.dtype, self.covariance)

    def reshape_input(
        self,
        input_shape: tuple[int, ...],
    ) -> typing.Self:
        """Return same model with input shape reshaped."""
        return self._replace(
            affine=self.affine.reshape_input(input_shape),
        )

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def astype(
        self,
        dtype: jax.typing.DTypeLike,
    ) -> typing.Self:
        """Convert to a different data type."""
        return self._replace(
            affine=self.affine.astype(dtype),
            covariance=self.covariance.astype(dtype),
        )

    def conditional_mean(
        self,
        values: jax.Array,
    ) -> jax.Array:
        """Mean of conditional distribution at deterministic input."""
        return self.affine.apply(values)

    def conditional(
        self,
        values: jax.Array,  # (..., *input_shape)
    ) -> Gaussian:
        """Conditional distribution at deterministic input."""
        mean = self.conditional_mean(values)  # (..., O)

        covariance = jnp.broadcast_to(
            batch.align_array(self.covariance, self.batch_shape, mean.shape[:-1]),
            mean.shape[:-1] + (self.output_dim, self.output_dim),
        )  # (..., O, O)

        return Gaussian(
            mean=mean,
            covariance=covariance,
        )

    def sample(
        self,
        key: jax.Array,
        values: jax.Array,
    ) -> jax.Array:
        """Sample from conditional distribution at deterministic input."""
        return self.conditional(values).sample(key)

    def compose_input(
        self,
        affine: Affine,
    ) -> typing.Self:
        """
        Precompose with $u=W'u'+b'$.
        """
        return self._replace(
            affine=self.affine.compose(affine),
        )

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

        covariance = jnp.einsum(
            '...oi,...ij,...pj->...op',
            affine.coefficients_flat,
            self.covariance,
            affine.coefficients_flat,
        )

        return self.__class__(
            affine=affine.compose(self.affine),
            covariance=covariance,
        )

    def expected_log_prob(
        self,
        input: Gaussian,
        output: Gaussian,
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

        noise = Gaussian(
            mean=jnp.zeros_like(self.affine.bias),
            covariance=self.covariance,
        )

        return noise.expected_log_prob(
            Gaussian(
                mean=residual_mean,
                covariance=residual_covariance,
            )
        )

    def expected_log_prob_broadcast(
        self,
        input: Gaussian,
        output: Gaussian,
        input_output_covariance: jax.Array,
    ) -> jax.Array:
        """Evaluate joint moments with receiver batch axes first.

        Cross covariance stores Cov(input, output), with trailing shape (I, O).
        """
        # Gaussian input means may retain the conditional's tensor input shape.
        input = Gaussian(self.affine.flatten_input(input.mean), input.covariance)
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
            input = input._replace(mean=self.affine.unflatten_input(input.mean))
        return model.expected_log_prob(
            input,
            output,
            input_output_covariance,
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)


class PairedGaussian(typing.NamedTuple):
    r"""Joint Gaussian distribution over pair of random variables.

    `left` and `right` are marginal distributions. `cross_covariance` is
    $\operatorname{Cov}(\text{right}, \text{left})$. Leading dimensions are batch dimensions,
    shared across all attributes.
    """

    left: Gaussian
    right: Gaussian
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

    dist: Gaussian
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
