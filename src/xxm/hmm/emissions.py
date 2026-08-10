import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp


def _kmeans(
    observations: jax.Array,
    num_states: int,
    key: jax.Array,
    num_iters: int = 20,
) -> jax.Array:
    """Return hard K-means assignments with shape (T,)."""
    initial_indices = jax.random.choice(
        key,
        observations.shape[0],
        shape=(num_states,),
        replace=False,
    )
    initial_centers = observations[initial_indices]

    def step(_, centers):
        distances = jnp.sum(
            (observations[:, None, :] - centers[None, :, :]) ** 2,
            axis=-1,
        )
        assignments = jnp.argmin(distances, axis=1)

        weights = jax.nn.one_hot(assignments, num_states)
        counts = weights.sum(axis=0)

        new_centers = weights.T @ observations / jnp.maximum(counts[:, None], 1)

        # Keep the old center if a cluster is empty.
        return jnp.where(
            (counts > 0)[:, None],
            new_centers,
            centers,
        )

    centers = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        initial_centers,
    )

    distances = jnp.sum(
        (observations[:, None, :] - centers[None, :, :]) ** 2,
        axis=-1,
    )
    return jnp.argmin(distances, axis=1)


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

    def m_step(
        self,
        observations: jax.Array,
        state_marginals: jax.Array,  # (T, K)
    ) -> 'PoissonEmissions':

        state_counts = state_marginals.sum(axis=0)  # (K,)

        rates = state_marginals.T @ observations / state_counts[:, None]

        return PoissonEmissions(rates=rates)

    @classmethod
    def initialize(
        cls,
        observations: jax.Array,
        num_states: int,
        key: jax.Array,
    ) -> 'PoissonEmissions':
        assignments = _kmeans(
            observations,
            num_states,
            key,
        )

        weights = jax.nn.one_hot(assignments, num_states)
        counts = weights.sum(axis=0)

        rates = weights.T @ observations / jnp.maximum(counts[:, None], 1)

        rates = jnp.maximum(rates, 1e-8)

        return cls(rates=rates)

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

    def m_step(
        self,
        observations: jax.Array,
        state_marginals: jax.Array,  # (T, K)
    ) -> 'GaussianEmissions':
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

    @classmethod
    def initialize(
        cls,
        observations: jax.Array,
        num_states: int,
        key: jax.Array,
    ) -> 'GaussianEmissions':
        assignments = _kmeans(
            observations,
            num_states,
            key,
        )

        weights = jax.nn.one_hot(assignments, num_states)
        counts = weights.sum(axis=0)

        means = weights.T @ observations / jnp.maximum(counts[:, None], 1)

        residuals = observations[:, None, :] - means[None, :, :]

        covariances = jnp.einsum(
            'tk,tki,tkj->kij',
            weights,
            residuals,
            residuals,
        ) / jnp.maximum(counts[:, None, None], 1)

        covariances += 1e-6 * jnp.eye(observations.shape[-1])[None, :, :]

        return cls(
            means=means,
            covariances=covariances,
        )

    def permute(
        self,
        permutation: jax.Array,
    ) -> 'GaussianEmissions':
        return GaussianEmissions(
            means=self.means[permutation],
            covariances=self.covariances[permutation],
        )
