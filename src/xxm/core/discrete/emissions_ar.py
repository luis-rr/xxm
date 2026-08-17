r"""
Autoregressive emission models for hidden Markov models.

For state \(z_t=k\), the conditional predictor is

\[
\eta_t^{(k)}
=
b_k + \sum_{\ell=1}^{L} A_{k\ell} y_{t-\ell}.
\]

Gaussian emissions use
\[
    \(y_t \sim \mathcal{N}(\eta_t^{(k)}, \Sigma_k)\)
\]
while Poisson emissions use
\[
    \(y_t \sim \operatorname{Poisson}(\exp(\eta_t^{(k)}))\).
\]

Inference and fitting use the conditional likelihood given the first \(L\)
observations, so their emission log likelihoods are set to zero. Sampling
instead starts from an all-zero history, providing a simple initial condition
without introducing a separate initial-observation model.
"""

import typing

import jax
import jax.numpy as jnp

from xxm.hmm.core import Posterior
from xxm.stats import gaussian, poisson


def lagged_observations(
    observations: jax.Array,
    lag: int,
    num_dims: int,
) -> jax.Array:
    """Return histories ordered from lag 1 to lag L."""
    return jnp.stack(
        [observations[lag - i - 1 : observations.shape[0] - i - 1] for i in range(lag)],
        axis=1,
    )  # (T-L, L, N)


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

    def conditional_means(self, observations: jax.Array) -> jax.Array:
        history = lagged_observations(
            observations,
            lag=self.lag,
            num_dims=self.num_dims,
        )  # (T-L, L, N)

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
        history = lagged_observations(
            observations,
            lag=self.lag,
            num_dims=self.num_dims,
        )  # (T-L, L, N)
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


class ARPoissonEmissions(typing.NamedTuple):
    coefficients: jax.Array  # (K, L, N, N)
    biases: jax.Array  # (K, N)

    @property
    def lag(self) -> int:
        return self.coefficients.shape[1]

    @property
    def num_states(self) -> int:
        return self.coefficients.shape[0]

    @property
    def num_dims(self) -> int:
        return self.coefficients.shape[-1]

    def log_rates(self, observations: jax.Array) -> jax.Array:
        history = lagged_observations(
            observations,
            lag=self.lag,
            num_dims=self.num_dims,
        )  # (T-L, L, N)

        return (
            jnp.einsum(
                'klnm,tlm->tkn',
                self.coefficients,
                history,
            )
            + self.biases
        )

    def rates(self, observations: jax.Array) -> jax.Array:
        return jnp.exp(self.log_rates(observations))

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        log_likelihoods = poisson.log_likelihoods(
            observations=observations[self.lag :],
            log_rates=self.log_rates(observations),
        )

        padding = jnp.zeros(
            (self.lag, self.num_states),
            dtype=log_likelihoods.dtype,
        )

        return jnp.concatenate([padding, log_likelihoods], axis=0)

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self:
        history = lagged_observations(
            observations,
            lag=self.lag,
            num_dims=self.num_dims,
        )  # (T-L, L, N)

        current = observations[self.lag :]  # (T-L, N)

        num_samples, lag, n = history.shape

        predictors = history.reshape(
            num_samples,
            lag * n,
        )  # (T-L, L*N)

        state_weights = posterior.state_marginals[self.lag :]  # (T-L, K)

        # (K, L, N, N) -> (K, N, L, N) -> (K, N, L*N)
        readout = jnp.transpose(
            self.coefficients,
            (0, 2, 1, 3),
        ).reshape(
            self.num_states,
            n,
            lag * n,
        )

        coefficients, biases = poisson.fit_weighted_linear(
            inputs=predictors,
            outputs=current,
            weights=state_weights,
            coefficients=readout,
            bias=self.biases,
        )

        # (K, N, L*N) -> (K, N, L, N) -> (K, L, N, N)
        coefficients = coefficients.reshape(
            self.num_states,
            n,
            lag,
            n,
        )
        coefficients = jnp.transpose(
            coefficients,
            (0, 2, 1, 3),
        )

        return self.__class__(
            coefficients=coefficients,
            biases=biases,
        )

    def permute(self, permutation: jax.Array) -> typing.Self:
        """Return a copy with states reordered by ``permutation``."""
        return self.__class__(
            coefficients=self.coefficients[permutation],
            biases=self.biases[permutation],
        )

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        def step(carry, state):
            history, key = carry

            key, key_observation = jax.random.split(key)

            log_rate = (
                jnp.einsum(
                    'lnm,lm->n',
                    self.coefficients[state],
                    history,
                )
                + self.biases[state]
            )

            observation = jax.random.poisson(
                key_observation,
                jnp.exp(log_rate),
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
