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

from xxm.core.chains.discrete import DiscretePotential
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.posteriors import DiscretePosterior


def lagged_observations(
    observations: jax.Array, num_lags: int
) -> jax.Array:  # (T-L, L, N)
    """Return histories ordered from lag 1 to lag L."""

    return jnp.stack(
        [
            observations[num_lags - i - 1 : observations.shape[0] - i - 1]
            for i in range(num_lags)
        ],
        axis=1,
    )


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


class AREmissions(
    typing.NamedTuple,
    typing.Generic[ConditionalModelT],
):
    r"""
    State-dependent autoregressive emissions with structured lagged inputs.

    The conditional model maps predictors of shape ``(L, N)`` to outputs of shape
    ``(N,)``, where ``L`` is the number of lags and ``N`` the observationdimension.

    Predictors are ordered from most recent to oldest observation, so
    ``x_t = (y_{t-1}, ..., y_{t-L})``.

    For an affine conditional model, coefficients therefore have shape ``(K, N, L, N)``.
    """

    model: ConditionalModelT  # K-batched, input shape (L, N)

    @property
    def num_states(self) -> int:
        return self.model.batch_shape[0]

    @property
    def output_dim(self) -> int:
        return self.model.output_dim

    @property
    def num_lags(self) -> int:
        if len(self.model.input_shape) != 2:
            raise ValueError(
                'autoregressive emissions require model input shape (L, N); '
                f'got {self.model.input_shape}'
            )

        num_lags, input_dim = self.model.input_shape

        if input_dim != self.output_dim:
            raise ValueError(
                'autoregressive input variable dimension must match output dimension; '
                f'got input shape {self.model.input_shape} '
                f'and output dimension {self.output_dim}'
            )

        return num_lags

    def predictors(self, observations: jax.Array) -> jax.Array:  # (T-L, L, N)
        """Construct autoregressive predictors ordered from lag 1 to lag L."""
        return lagged_observations(
            observations,
            self.num_lags,
        )

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

    def conditional(self, observations: jax.Array) -> Gaussian | Poisson:
        """Conditional distribution for each time point and state."""
        predictors = self.predictors(
            observations,
        )  # (T-L, L, N)

        return self.model.conditional(predictors[:, None, ...])  # (T-L, K)

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:  # (T, K)
        conditional = self.conditional(observations)

        log_probs = conditional.log_prob(
            observations[self.num_lags :, None, :]
        )  # (T-L, K)

        padding = jnp.zeros(
            (self.num_lags, self.num_states),
            dtype=log_probs.dtype,
        )  # (L, K)

        return jnp.concatenate(
            [
                padding,
                log_probs,
            ],
            axis=0,
        )  # (T, K)

    def compute_potential(self, observations: jax.Array) -> DiscretePotential:
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self,
        observations: jax.Array,  # (T, N)
        posterior: DiscretePosterior,
    ) -> typing.Self:
        predictors = self.predictors(
            observations,
        )  # (T-L, L, N)

        current = observations[self.num_lags :]  # (T-L, N)

        weights = posterior.state_probs[self.num_lags :]  # (T-L, K)

        model = _fit_ar_model(
            model=self.model,
            inputs=predictors,
            outputs=current,
            weights=weights,
        )

        return self._replace(
            model=model,
        )

    def permute(self, permutation: jax.Array) -> typing.Self:
        return self._replace(
            model=self.model.select(permutation),
        )

    def sample(self, key: jax.Array, states: jax.Array) -> jax.Array:  # (T, N)
        def step(carry, state):
            history, key = carry  # (L, N)

            key, key_observation = jax.random.split(key)

            conditional = self.model.select(state).conditional(history)

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
            (
                self.num_lags,
                self.output_dim,
            ),
            dtype=self.model.dtype,
        )  # (L, N)

        _, observations = jax.lax.scan(
            step,
            (initial_history, key),
            states,
        )

        return observations
