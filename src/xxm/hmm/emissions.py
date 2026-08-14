import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .core import Posterior


class PoissonEmissions(typing.NamedTuple):
    rates: jax.Array  # (K, N)

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        # Returns (T, K)
        x = observations[:, None, :]  # (T, 1, N)
        rates = self.rates[None, :, :]  # (1, K, N)

        return jnp.sum(
            x * jnp.log(rates) - rates - jsp.special.gammaln(x + 1),
            axis=-1,
        )

    def fit_params(self, observations: jax.Array, posterior: Posterior) -> 'PoissonEmissions':
        return self.__class__(posterior.weighted_means(observations))

    def permute(
        self,
        permutation: jax.Array,
    ) -> 'PoissonEmissions':
        return PoissonEmissions(
            rates=self.rates[permutation],
        )

    def sample(
        self,
        key: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        return jax.random.poisson(
            key,
            self.rates[state],
        )


class GaussianEmissions(typing.NamedTuple):
    means: jax.Array  # (K, N)
    covariances: jax.Array  # (K, N, N)

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        # (T, K, N)
        residuals = observations[:, None, :] - self.means[None, :, :]

        # (K, N, N)
        chol = jnp.linalg.cholesky(self.covariances)

        # Solve L y = x for every (t, k)
        solved = jsp.linalg.solve_triangular(
            chol[None, :, :, :],
            residuals[..., None],
            lower=True,
        )[..., 0]

        mahalanobis = jnp.sum(solved**2, axis=-1)  # (T, K)

        log_det = 2 * jnp.sum(
            jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)),
            axis=-1,
        )  # (K,)

        n_dims = observations.shape[-1]

        return -0.5 * (n_dims * jnp.log(2 * jnp.pi) + log_det[None, :] + mahalanobis)

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> 'GaussianEmissions':

        state_marginals = posterior.state_marginals  # (T, K)
        state_counts = state_marginals.sum(axis=0)  # (K,)

        means = state_marginals.T @ observations / state_counts[:, None]  # (K, N)

        residuals = observations[:, None, :] - means[None, :, :]  # (T,K,N)

        covariances = (
            jnp.einsum(
                'tk,tki,tkj->kij',
                state_marginals,
                residuals,
                residuals,
            )
            / state_counts[:, None, None]
        )

        return GaussianEmissions(
            means=means,
            covariances=covariances,
        )

    def sample(
        self,
        key: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        return jax.random.multivariate_normal(
            key,
            self.means[state],
            self.covariances[state],
        )

    def permute(
        self,
        permutation: jax.Array,
    ) -> 'GaussianEmissions':
        return GaussianEmissions(
            means=self.means[permutation],
            covariances=self.covariances[permutation],
        )
