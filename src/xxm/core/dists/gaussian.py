"""Gaussian distributions, conditionals, and paired moment representations."""

import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as jsp_linalg

from xxm.core import batch
from xxm.core.affine import Affine


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
        assert mean_shape == covariance_shape
        return mean_shape

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = batch.axis_index(source, len(self.batch_shape))
        destination = batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            mean=jnp.moveaxis(self.mean, source, destination),
            covariance=jnp.moveaxis(self.covariance, source, destination),
        )

    def permute_variables(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Reorder variable dimensions globally or independently by batch."""
        permutation = jnp.asarray(permutation)

        global_shape = (self.variable_dim,)
        batched_shape = self.batch_shape + global_shape

        if permutation.shape == global_shape:
            permutation = jnp.broadcast_to(
                permutation,
                batched_shape,
            )
        elif permutation.shape != batched_shape:
            raise ValueError(
                'permutation must have shape '
                f'{global_shape} or {batched_shape}, '
                f'got {permutation.shape}'
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
        coefficient_signs: jax.Array,
    ) -> typing.Self:
        """Flip variable signs globally or independently by batch."""
        coefficient_signs = jnp.asarray(coefficient_signs)

        global_shape = (self.variable_dim,)
        batched_shape = self.batch_shape + global_shape

        if coefficient_signs.shape == global_shape:
            coefficient_signs = jnp.broadcast_to(
                coefficient_signs,
                batched_shape,
            )
        elif coefficient_signs.shape != batched_shape:
            raise ValueError(
                'coefficient_signs must have shape '
                f'{global_shape} or {batched_shape}, '
                f'got {coefficient_signs.shape}'
            )

        return self._replace(
            mean=self.mean * coefficient_signs,
            covariance=(
                coefficient_signs[..., :, None]
                * self.covariance
                * coefficient_signs[..., None, :]
            ),
        )

    @property
    def variable_dim(self) -> int:
        """Dimension of the Gaussian variable."""
        return self.mean.shape[-1]

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        """Data type."""
        return jnp.result_type(self.mean, self.covariance)

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = batch.selection(index, len(self.batch_shape))
        return self.__class__(
            mean=self.mean[index],
            covariance=self.covariance[index],
        )

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
        batch.require_same(self.batch_shape, affine.batch_shape)

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
        batch.require_same(self.batch_shape, other.batch_shape)
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
        shape, axis = batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            mean=batch.broadcast_array(self.mean, shape, axis),
            covariance=batch.broadcast_array(self.covariance, shape, axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            mean=jnp.squeeze(self.mean, axis=axes),
            covariance=jnp.squeeze(self.covariance, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = batch.axis_index(axis, len(self.batch_shape))
        permutation = batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            mean=jnp.take(self.mean, permutation, axis=axis),
            covariance=jnp.take(self.covariance, permutation, axis=axis),
        )


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
        assert covariance_shape == affine_shape
        return affine_shape

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            affine=self.affine.broadcast(shape, axis=axis),
            covariance=batch.broadcast_array(self.covariance, shape, axis),
        )

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
    def dtype(self) -> jax.typing.DTypeLike:
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
        index = batch.selection(index, len(self.batch_shape))
        return self.__class__(
            affine=self.affine.select(index),
            covariance=self.covariance[index],
        )

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
        batch.require_same(
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
        batch.require_same(
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
        axes = batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            affine=self.affine.squeeze(axes),
            covariance=jnp.squeeze(self.covariance, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = batch.axis_index(axis, len(self.batch_shape))
        permutation = batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            affine=self.affine.permute(permutation, axis=axis),
            covariance=jnp.take(self.covariance, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = batch.axis_index(source, len(self.batch_shape))
        destination = batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            affine=self.affine.move_axis(source, destination),
            covariance=jnp.moveaxis(self.covariance, source, destination),
        )


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

        assert left_shape == right_shape
        assert left_shape == cross_shape
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
    def dtype(self) -> jax.typing.DTypeLike:
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
        index = batch.selection(index, len(self.batch_shape))
        return self.__class__(
            left=self.left.select(index),
            right=self.right.select(index),
            cross_covariance=self.cross_covariance[index],
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        """Convert to a different data type."""
        return self._replace(
            left=self.left.astype(dtype),
            right=self.right.astype(dtype),
            cross_covariance=self.cross_covariance.astype(dtype),
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            left=self.left.broadcast(shape, axis=axis),
            right=self.right.broadcast(shape, axis=axis),
            cross_covariance=batch.broadcast_array(self.cross_covariance, shape, axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            left=self.left.squeeze(axes),
            right=self.right.squeeze(axes),
            cross_covariance=jnp.squeeze(self.cross_covariance, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = batch.axis_index(axis, len(self.batch_shape))
        permutation = batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            left=self.left.permute(permutation, axis=axis),
            right=self.right.permute(permutation, axis=axis),
            cross_covariance=jnp.take(self.cross_covariance, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = batch.axis_index(source, len(self.batch_shape))
        destination = batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            left=self.left.move_axis(source, destination),
            right=self.right.move_axis(source, destination),
            cross_covariance=jnp.moveaxis(self.cross_covariance, source, destination),
        )
