"""Gaussian distributions, conditionals, and paired moment representations."""

import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.core.affine import Affine


class Gaussian(typing.NamedTuple):
    r"""Multivariate Gaussian distribution in moment form.

    $$u \sim \mathcal{N}(\mu, \Sigma).$$

    `mean` stores $\mu$ and `covariance` stores $\Sigma$. Leading dimensions
    are batch dimensions, shared across all attributes.
    """

    mean: jax.Array  # (..., N)
    covariance: jax.Array  # (..., N, N)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        mean_shape = self.mean.shape[:-1]
        covariance_shape = self.covariance.shape[:-2]
        assert mean_shape == covariance_shape
        return mean_shape

    @property
    def variable_dim(self) -> int:
        """Dimension of the Gaussian variable."""
        return self.mean.shape[-1]

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        """Data type."""
        return jnp.result_type(self.mean, self.covariance)

    def select(self, index) -> 'Gaussian':
        """Index into batch dimensions."""
        return Gaussian(
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

    def sample(
        self,
        key: jax.Array,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Sample from the distribution."""
        return jax.random.multivariate_normal(
            key,
            mean=self.mean,
            cov=self.covariance,
            shape=sample_shape + self.batch_shape,
        )

    def _validate_affine_input(
        self,
        affine: Affine,
    ) -> None:
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
        """Evaluate log densities with aligned/broadcast-compatible batch dimensions."""
        residuals = values - self.mean  # (..., N)

        chol = jnp.linalg.cholesky(self.covariance)  # (..., N, N)

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

    def log_prob_broadcast(
        self,
        values: jax.Array,  # (..., N)
    ) -> jax.Array:  # (..., *batch_shape)
        """Evaluate every value against every batched distribution."""
        values = values.reshape(
            values.shape[:-1] + (1,) * len(self.batch_shape) + (self.variable_dim,)
        )  # (..., 1, ..., 1, N)

        return self.log_prob(values)

    def mixture_mean(self, weights: jax.Array) -> jax.Array:
        """Mean of mixture over last batch dimension."""
        return jnp.sum(
            weights[..., :, None] * self.mean,
            axis=-2,
        )

    def expected_log_prob(
        self,
        other: 'Gaussian',
    ) -> jax.Array:
        r"""
        Expected log density $\mathbb{E}_{u\sim\text{other}}[\log p(u)]$.
        """
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

        batch_shape = jnp.broadcast_shapes(
            self.batch_shape,
            other.mean.shape[:-1],
            other.covariance.shape[:-2],
        )

        mean = jnp.broadcast_to(
            other.mean,
            batch_shape + (self.variable_dim,),
        )

        covariance = jnp.broadcast_to(
            other.covariance,
            batch_shape + (self.variable_dim, self.variable_dim),
        )

        model_mean = jnp.broadcast_to(
            self.mean,
            batch_shape + (self.variable_dim,),
        )

        cholesky = jnp.broadcast_to(
            jnp.linalg.cholesky(self.covariance),
            batch_shape + (self.variable_dim, self.variable_dim),
        )

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
            affine=self.affine.input_reshape(input_shape),
        )

    def select(self, index) -> typing.Self:
        """Index into the batch dimensions."""
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
            self.covariance,
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

    def add_covariance_jitter(
        self,
        jitter: float,
    ) -> typing.Self:
        """Add isotropic jitter to output covariance for numerical stability."""
        identity = jnp.eye(
            self.output_dim,
            dtype=self.covariance.dtype,
        )

        return self._replace(
            covariance=self.covariance + jitter * identity,
        )

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
        """
        input_mean_flat = self.affine.input_flatten(
            input.mean,
        )

        coefficients = self.affine.coefficients_flat

        residual_mean = (
            output.mean
            - jnp.einsum(
                '...oi,...i->...o',
                coefficients,
                input_mean_flat,
            )
            - self.affine.bias
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
        r"""Evaluate every moment tuple against every batched model.

        `input_output_covariance` stores $\operatorname{Cov}_q(u,v)$ with
        trailing shape $(I,O)$, as in `expected_log_prob`.
        """
        extra = (1,) * len(self.batch_shape)

        # TODO is there a joint re-shape + broadcast method hiding in here?

        input_mean = input.mean.reshape(
            input.mean.shape[: -self.input_ndim] + extra + self.input_shape
        )

        input_covariance = input.covariance.reshape(
            input.covariance.shape[:-2] + extra + (self.input_size, self.input_size)
        )

        output_mean = output.mean.reshape(
            output.mean.shape[:-1] + extra + (self.output_dim,)
        )

        output_covariance = output.covariance.reshape(
            output.covariance.shape[:-2] + extra + (self.output_dim, self.output_dim)
        )

        input_output_covariance = input_output_covariance.reshape(
            input_output_covariance.shape[:-2]
            + extra
            + (self.input_size, self.output_dim)
        )

        return self.expected_log_prob(
            input=Gaussian(
                mean=input_mean,
                covariance=input_covariance,
            ),
            output=Gaussian(
                mean=output_mean,
                covariance=output_covariance,
            ),
            input_output_covariance=input_output_covariance,
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

    def select(self, index) -> typing.Self:
        """Index into the batch dimensions."""
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
