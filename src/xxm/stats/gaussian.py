import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.core.affine import Affine


class Gaussian(typing.NamedTuple):
    """A multivariate Gaussian distribution in moment form.

    Leading dimensions are batch dimensions and must be shared between attributes.
    """

    mean: jax.Array  # (..., N)
    covariance: jax.Array  # (..., N, N)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        mean_shape = self.mean.shape[:-1]
        covariance_shape = self.covariance.shape[:-2]
        assert mean_shape == covariance_shape
        return mean_shape

    @property
    def variable_dim(self) -> int:
        return self.mean.shape[-1]

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        return jnp.result_type(self.mean, self.covariance)

    def select(self, index) -> 'Gaussian':
        """Index into the batch dimensions of the distribution."""
        return Gaussian(
            mean=self.mean[index],
            covariance=self.covariance[index],
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> 'Gaussian':
        return self._replace(
            mean=self.mean.astype(dtype),
            covariance=self.covariance.astype(dtype),
        )

    @property
    def variance(self) -> jax.Array:
        return jnp.diagonal(self.covariance, axis1=-2, axis2=-1)

    def sample(
        self,
        key: jax.Array,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        return jax.random.multivariate_normal(
            key,
            mean=self.mean,
            cov=self.covariance,
            shape=sample_shape + self.batch_shape,
        )

    def affine_mean(
        self,
        affine: Affine,
    ) -> jax.Array:
        """
        Calculate the mean of the distribution of y when:
            y = A x + b, x ~ N(self.mean, self.covariance)
        """
        return affine.apply(self.mean)

    def affine_covariance(
        self,
        affine: Affine,
    ) -> jax.Array:
        """
        Calculate the covariance of the distribution of y when:
            y = A x + b, x ~ N(self.mean, self.covariance)
        """

        covariance = jnp.einsum(
            '...oi,...ij,...pj->...op',
            affine.coefficients,
            self.covariance,
            affine.coefficients,
        )
        return covariance

    def affine_variance(
        self,
        affine: Affine,
    ) -> jax.Array:
        """
        Calculate the variance (diagonal of the covariance) of the distribution of y when:
            y = A x + b, x ~ N(self.mean, self.covariance)
        """
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
        """Distribution of ``y = A x + b`` for ``x ~ self``."""

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

        return -0.5 * (self.variable_dim * jnp.log(2.0 * jnp.pi) + log_det + mahalanobis)

    def log_prob_broadcast(
        self,
        values: jax.Array,  # (..., N)
    ) -> jax.Array:  # (..., *batch_shape)
        """Evaluate every value against every batched distribution."""
        values = values.reshape(
            values.shape[:-1] + (1,) * len(self.batch_shape) + (self.variable_dim,)
        )  # (..., 1, ..., 1, N)

        return self.log_prob(values)


class LinearGaussian(typing.NamedTuple):
    """A linear Gaussian model:

        y | x ~ N(A x + b, Q)

    Leading dimensions are batch dimensions and must be shared between attributes.
    """

    affine: Affine
    covariance: jax.Array  # (..., O, O)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        covariance_shape = self.covariance.shape[:-2]
        affine_shape = self.affine.batch_shape
        assert covariance_shape == affine_shape
        return affine_shape

    @property
    def input_dim(self) -> int:
        return self.affine.input_dim

    @property
    def output_dim(self) -> int:
        return self.affine.output_dim

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        return jnp.result_type(self.affine.dtype, self.covariance)

    def select(self, index) -> 'LinearGaussian':
        """Index into the batch dimensions."""
        return self.__class__(
            affine=self.affine.select(index),
            covariance=self.covariance[index],
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> 'LinearGaussian':
        return self._replace(
            affine=self.affine.astype(dtype),
            covariance=self.covariance.astype(dtype),
        )

    def conditional_mean(self, values: jax.Array) -> jax.Array:
        """Conditional mean for deterministic input values."""
        return self.affine.apply(values)

    def conditional(
        self,
        values: jax.Array,  # (..., I)
    ) -> Gaussian:
        """Conditional output distribution for deterministic inputs."""
        mean = self.conditional_mean(values)  # (..., O)

        covariance = jnp.broadcast_to(
            self.covariance,
            mean.shape[:-1] + (self.affine.output_dim, self.affine.output_dim),
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
        return self.conditional(values).sample(key)

    def add_covariance_jitter(
        self,
        jitter: float,
    ) -> 'LinearGaussian':
        """Add isotropic jitter to the output covariance."""
        identity = jnp.eye(
            self.affine.output_dim,
            dtype=self.covariance.dtype,
        )

        return self._replace(
            covariance=self.covariance + jitter * identity,
        )
