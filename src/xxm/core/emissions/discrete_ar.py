r"""Autoregressive emission models for state-conditional variables with history dependence.

For observation time $t=L,\ldots,T-1$ and state $z_{t-L}=k$, the predictor is

$$\eta_t^{(k)} = b_k + \sum_{\ell=1}^L A_{k,\ell} y_{t-\ell}.$$

Gaussian emissions: $y_t \sim \mathcal{N}(\eta_t^{(k)}, R_k)$.

Poisson emissions: $y_t \sim \operatorname{Poisson}(\exp(\eta_t^{(k)}))$.

Inference and fitting condition on the first $L$ observations, so only the
remaining $T-L$ observations have associated latent states. Sampling starts
from zero history unless an explicit continuation history is supplied.
The compact posterior index $s=t-L$ means `state_probs[s]` describes the
state generating `observations[L+s]`.
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


ConditionalDistT = typing.TypeVar(
    'ConditionalDistT',
    LinearGaussian,
    LinearPoisson,
)


def _fit_ar_model(
    dist: ConditionalDistT,
    inputs: jax.Array,
    outputs: jax.Array,
    weights: jax.Array,
) -> ConditionalDistT:
    if isinstance(dist, LinearGaussian):
        return gaussian_fit.linear_from_samples_weighted(
            inputs=inputs,
            outputs=outputs,
            weights=weights,
            ridge=1e-6,
        )

    if isinstance(dist, LinearPoisson):
        return poisson_fit.linear_from_samples_weighted(
            inputs=inputs,
            outputs=outputs,
            weights=weights,
            initial_affine=dist.affine,
        )

    typing.assert_never(dist)


class AREmissions(
    typing.NamedTuple,
    typing.Generic[ConditionalDistT],
):
    r"""
    State-dependent autoregressive emissions with structured lagged inputs.

    The conditional model maps predictors of shape $(L,D_y)$ to outputs of
    shape $(D_y,)$. Predictors are ordered from most recent to oldest,
    ``(y[t-1], ..., y[t-L])``. For an affine conditional model, coefficients
    therefore have shape $(K,D_y,L,D_y)$.

    Likelihood and fitting methods receive a chronological sequence whose first
    ``L`` observations are fixed conditioning history. Only the remaining
    observations have associated latent states: posterior index $s$ corresponds
    to `observations[L+s]`.

    Autonomous sampling uses an all-zero history. Conditional continuation
    sampling accepts an explicit chronological history, ordered from oldest
    to most recent.
    """

    dist: ConditionalDistT  # K-batched, input shape (L, N)

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.dist.batch_shape[0]

    @property
    def output_dim(self) -> int:
        """Observation dimension $D_y$."""
        return self.dist.output_dim

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        if len(self.dist.input_shape) != 2:
            raise ValueError(
                'autoregressive emissions require model input shape (L, N); '
                f'got {self.dist.input_shape}'
            )

        num_lags, input_dim = self.dist.input_shape

        if input_dim != self.output_dim:
            raise ValueError(
                'autoregressive input variable dimension must match output '
                f'dimension; got input shape {self.dist.input_shape} '
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
        self: 'AREmissions[ConditionalDistT]',
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

        return self.dist.conditional(
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
            dist=self.dist,
            inputs=predictors,
            outputs=current,
            weights=posterior.state_probs,  # (T-L, K)
        )

        return self._replace(
            dist=model,
        )

    def permute(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel state-specific autoregressive emission parameters."""
        return self._replace(
            dist=self.dist.select(permutation),
        )

    def sample_continuation(
        self,
        key: jax.Array,
        states: jax.Array,
        initial_history: jax.Array,
    ) -> jax.Array:
        """
        Sample a continuation conditional on an explicit observation history.

        `initial_history` has shape $(L,D_y)$ and is chronological, from
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

            conditional = self.dist.select(state).conditional(
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
            dtype=self.dist.dtype,
        )

        return self.sample_continuation(
            key,
            states,
            initial_history,
        )
