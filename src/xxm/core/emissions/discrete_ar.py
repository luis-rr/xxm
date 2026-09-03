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

    The conditional model maps predictors of shape ``(L, N)`` to outputs of
    shape ``(N,)``. Predictors are ordered from most recent to oldest,
    ``(y[t-1], ..., y[t-L])``. For an affine conditional model, coefficients
    therefore have shape ``(K, N, L, N)``.

    Likelihood and fitting methods receive a chronological sequence whose first
    ``L`` observations are fixed conditioning history. Only the remaining
    observations have associated latent states.

    Autonomous sampling uses an all-zero history. Conditional continuation
    sampling accepts an explicit chronological history, ordered from oldest
    to most recent.
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
                'autoregressive input variable dimension must match output '
                f'dimension; got input shape {self.model.input_shape} '
                f'and output dimension {self.output_dim}'
            )

        return num_lags

    def predictors(
        self,
        observations: jax.Array,
    ) -> jax.Array:  # (T-L, L, N)
        """Construct predictors ordered from most recent to oldest."""
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

    def conditional(
        self,
        observations: jax.Array,
    ) -> Gaussian | Poisson:
        """Conditional distribution for each modeled time point and state."""
        predictors = self.predictors(
            observations,
        )  # (T-L, L, N)

        return self.model.conditional(
            predictors[:, None, ...],
        )  # (T-L, K)

    def log_likelihoods(
        self,
        observations: jax.Array,
    ) -> jax.Array:  # (T-L, K)
        """Emission log likelihoods conditional on the initial history."""
        conditional = self.conditional(
            observations,
        )

        return conditional.log_prob(observations[self.num_lags :, None, :])

    def compute_potential(
        self,
        observations: jax.Array,
    ) -> DiscretePotential:
        """Construct potentials only for observations after the history."""
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self,
        observations: jax.Array,
        posterior: DiscretePosterior,
    ) -> typing.Self:
        """Fit AR parameters conditional on the initial observation history."""
        predictors = self.predictors(
            observations,
        )  # (T-L, L, N)

        current = observations[self.num_lags :]  # (T-L, N)

        model = _fit_ar_model(
            model=self.model,
            inputs=predictors,
            outputs=current,
            weights=posterior.state_probs,  # (T-L, K)
        )

        return self._replace(
            model=model,
        )

    def permute(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        return self._replace(
            model=self.model.select(permutation),
        )

    def sample_continuation(
        self,
        key: jax.Array,
        states: jax.Array,
        initial_history: jax.Array,
    ) -> jax.Array:
        """
        Sample a continuation conditional on an explicit observation history.

        ``initial_history`` has shape ``(L, N)`` and is chronological, from
        oldest to most recent. Only newly generated observations are returned.
        """
        if initial_history.shape != (
            self.num_lags,
            self.output_dim,
        ):
            raise ValueError(
                'initial_history must have shape '
                f'({self.num_lags}, {self.output_dim}); '
                f'got {initial_history.shape}'
            )

        # Conditional predictors are ordered most recent to oldest.
        history = initial_history[::-1]

        def step(carry, state):
            history, key = carry

            key, key_observation = jax.random.split(key)

            conditional = self.model.select(state).conditional(
                history,
            )

            observation = conditional.sample(
                key_observation,
            )

            history = jnp.concatenate(
                [
                    observation[None, :],
                    history[:-1],
                ],
                axis=0,
            )

            return (history, key), observation

        _, observations = jax.lax.scan(
            step,
            (history, key),
            states,
        )

        return observations

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        """
        Sample observations autonomously from a zero prehistory.

        The zero history is a deterministic simulation boundary condition, not
        an assumption about the stationary distribution of the AR process.
        """
        initial_history = jnp.zeros(
            (
                self.num_lags,
                self.output_dim,
            ),
            dtype=self.model.dtype,
        )

        return self.sample_continuation(
            key,
            states,
            initial_history,
        )
