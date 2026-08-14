import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.hmm.core import Posterior
from xxm.stats import gaussian


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
        states: jax.Array,
    ) -> jax.Array:
        return jax.random.poisson(
            key,
            self.rates[states],
        )


class GaussianEmissions(typing.NamedTuple):
    means: jax.Array  # (K, N)
    covariances: jax.Array  # (K, N, N)

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:

        return gaussian.log_likelihoods(
            observations=observations,
            means=self.means[None, :, :],
            covariances=self.covariances,
        )

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self:
        means, covariances = gaussian.fit_weighted(
            observations,
            posterior.state_marginals,
        )

        return self._replace(
            means=means,
            covariances=covariances,
        )

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        return jax.random.multivariate_normal(
            key,
            self.means[states],
            self.covariances[states],
        )

    def permute(
        self,
        permutation: jax.Array,
    ) -> 'GaussianEmissions':
        return GaussianEmissions(
            means=self.means[permutation],
            covariances=self.covariances[permutation],
        )


class ARGaussianEmissions(typing.NamedTuple):
    coefficients: jax.Array  # (K, L, N, N)
    biases: jax.Array  # (K, N)
    covariances: jax.Array  # (K, N, N)

    @property
    def lag(self) -> int:
        return self.coefficients.shape[1]

    @property
    def num_states(self) -> int:
        return self.coefficients.shape[0]

    @property
    def num_dims(self) -> int:
        return self.coefficients.shape[-1]

    def _lagged_observations(
        self,
        observations: jax.Array,
    ) -> jax.Array:
        """Return histories ordered from lag 1 to lag L."""
        return jnp.stack(
            [
                observations[self.lag - i - 1 : observations.shape[0] - i - 1]
                for i in range(self.lag)
            ],
            axis=1,
        )  # (T-L, L, N)

    def conditional_means(self, observations: jax.Array) -> jax.Array:
        history = self._lagged_observations(observations)  # (T-L, L, N)

        return (
            jnp.einsum(
                'klnm,tlm->tkn',
                self.coefficients,
                history,
            )
            + self.biases
        )  # (T-L, K, N)

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        log_likelihoods = gaussian.log_likelihoods(
            observations=observations[self.lag :],
            means=self.conditional_means(observations),
            covariances=self.covariances,
        )  # (T-L, K)

        # Pad the first L time steps with zeros to match the shape of the input observations.
        # An alternative would be to have _to_chain work with a chain with T-L time steps,
        # but that would require an AR-specific inference instead of using the HMM one.
        padding = jnp.zeros(
            (self.lag, log_likelihoods.shape[1]),
            dtype=log_likelihoods.dtype,
        )

        return jnp.concatenate([padding, log_likelihoods], axis=0)

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self:
        history = self._lagged_observations(observations)  # (T-L, L, N)
        current = observations[self.lag :]  # (T-L, N)

        num_samples, lag, n = history.shape

        predictors = history.reshape(num_samples, lag * n)  # (T-L, L*N)

        # slice to account for the zero-padded log likelihoods (T-L, K)
        weights = posterior.state_marginals[self.lag :]

        coefficients, biases, covariances = gaussian.fit_weighted_linear(
            inputs=predictors,
            outputs=current,
            weights=weights,
            ridge=1e-6,
        )

        # (K, N, L*N) -> (K, N, L, N) -> (K, L, N, N)
        weights = coefficients.reshape(-1, n, lag, n)
        weights = jnp.transpose(weights, (0, 2, 1, 3))

        return self.__class__(
            coefficients=weights,
            biases=biases,
            covariances=covariances,
        )

    def permute(self, permutation: jax.Array) -> typing.Self:
        """Return a copy with states reordered by ``permutation``."""
        return self.__class__(
            coefficients=self.coefficients[permutation],
            biases=self.biases[permutation],
            covariances=self.covariances[permutation],
        )

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        def step(carry, state):
            history, key = carry

            key, key_observation = jax.random.split(key)

            mean = (
                jnp.einsum(
                    'lnm,lm->n',
                    self.coefficients[state],
                    history,
                )
                + self.biases[state]
            )

            observation = jax.random.multivariate_normal(
                key_observation,
                mean=mean,
                cov=self.covariances[state],
            )

            new_history = jnp.concatenate(
                [
                    observation[None, :],
                    history[:-1],
                ],
                axis=0,
            )

            return (new_history, key), observation

        initial_history = jnp.zeros(
            (self.lag, self.num_dims),
            dtype=self.biases.dtype,
        )

        _, observations = jax.lax.scan(
            step,
            (initial_history, key),
            states,
        )

        return observations
