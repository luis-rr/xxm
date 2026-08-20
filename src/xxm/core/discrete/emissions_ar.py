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

from xxm.core.discrete.chain import DiscretePotential
from xxm.hmm.core import Posterior
from xxm.stats import gaussian_fit, poisson_fit
from xxm.stats.gaussian import Gaussian, LinearGaussian
from xxm.stats.poisson import LinearPoisson, Poisson


def lagged_observations(observations: jax.Array, max_lag: int) -> jax.Array:  # (T-L, L, N)
    """Return histories ordered from lag 1 to lag L."""

    return jnp.stack(
        [observations[max_lag - i - 1 : observations.shape[0] - i - 1] for i in range(max_lag)],
        axis=1,
    )


def flatten_ar_coefficients(
    coefficients: jax.Array,  # (..., L, O, I)
) -> jax.Array:  # (..., O, L*I)
    """Flatten lagged AR coefficients into a standard affine input dimension."""
    coefficients = jnp.swapaxes(
        coefficients,
        -3,
        -2,
    )  # (..., O, L, I)

    return coefficients.reshape(
        coefficients.shape[:-2] + (coefficients.shape[-2] * coefficients.shape[-1],)
    )  # (..., O, L*I)


ConditionalModelT = typing.TypeVar(
    'ConditionalModelT',
    LinearGaussian,
    LinearPoisson,
)


def _fit_ar_model(
    model: ConditionalModelT,
    inputs: jax.Array,
    outputs: jax.Array,
    weights: jax.Array,
) -> ConditionalModelT:
    if isinstance(model, LinearGaussian):
        return gaussian_fit.linear_from_pairs_weighted(
            inputs=inputs,
            outputs=outputs,
            weights=weights,
            ridge=1e-6,
        )

    if isinstance(model, LinearPoisson):
        return poisson_fit.linear_from_pairs_weighted(
            inputs=inputs,
            outputs=outputs,
            weights=weights,
            initial_affine=model.affine,
        )

    typing.assert_never(model)


class AREmissions(typing.NamedTuple, typing.Generic[ConditionalModelT]):
    """State-dependent autoregressive emissions."""

    model: ConditionalModelT  # K-batched, input dimension L*N

    @property
    def num_states(self) -> int:
        return self.model.batch_shape[0]

    @property
    def output_dim(self) -> int:
        return self.model.output_dim

    @property
    def max_lag(self) -> int:
        return self.model.input_dim // self.output_dim

    def predictors(
        self,
        observations: jax.Array,  # (T, N)
    ) -> jax.Array:  # (T-L, L*N)
        """Construct flattened autoregressive predictors."""
        history = lagged_observations(observations, self.max_lag)  # (T-L, L, N)

        return history.reshape(
            history.shape[0],
            self.max_lag * self.output_dim,
        )  # (T-L, L*N)

    @typing.overload
    def conditional(
        self: 'AREmissions[LinearGaussian]',
        observations: jax.Array,
    ) -> Gaussian: ...

    @typing.overload
    def conditional(
        self: 'AREmissions[LinearPoisson]',
        observations: jax.Array,
    ) -> Poisson: ...

    @typing.overload
    def conditional(
        self: 'AREmissions[ConditionalModelT]',
        observations: jax.Array,
    ) -> Gaussian | Poisson: ...

    def conditional(
        self,
        observations: jax.Array,
    ) -> Gaussian | Poisson:
        """Conditional distribution for each time point and state."""
        predictors = self.predictors(observations)
        return self.model.conditional(predictors[:, None, :])

    def log_likelihoods(
        self,
        observations: jax.Array,  # (T, N)
    ) -> jax.Array:  # (T, K)
        conditional = self.conditional(observations)

        log_probs = conditional.log_prob(observations[self.max_lag :, None, :])  # (T-L, K)

        padding = jnp.zeros(
            (self.max_lag, self.num_states),
            dtype=log_probs.dtype,
        )  # (L, K)

        return jnp.concatenate(
            [padding, log_probs],
            axis=0,
        )  # (T, K)

    def get_potential(
        self,
        observations: jax.Array,  # (T, N)
    ) -> DiscretePotential:
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self,
        observations: jax.Array,  # (T, N)
        posterior: Posterior,
    ) -> typing.Self:
        predictors = self.predictors(observations)  # (T-L, L*N)
        current = observations[self.max_lag :]  # (T-L, N)
        weights = posterior.state_marginals[self.max_lag :]  # (T-L, K)

        model = _fit_ar_model(
            model=self.model,
            inputs=predictors,
            outputs=current,
            weights=weights,
        )
        return self._replace(model=model)

    def permute(
        self,
        permutation: jax.Array,  # (K,)
    ) -> typing.Self:
        return self._replace(
            model=self.model.select(permutation),
        )

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,  # (T,)
    ) -> jax.Array:  # (T, N)
        def step(carry, state):
            history, key = carry  # (L, N)

            key, key_observation = jax.random.split(key)

            predictors = history.reshape(
                self.max_lag * self.output_dim,
            )  # (L*N,)

            conditional = self.model.select(state).conditional(predictors)

            observation = conditional.sample(key_observation)  # (N,)

            new_history = jnp.concatenate(
                [
                    observation[None, :],
                    history[:-1],
                ],
                axis=0,
            )  # (L, N)

            return (new_history, key), observation

        initial_history = jnp.zeros(
            (self.max_lag, self.output_dim),
            dtype=self.model.dtype,
        )  # (L, N)

        _, observations = jax.lax.scan(
            step,
            (initial_history, key),
            states,
        )

        return observations
