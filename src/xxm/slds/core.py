import typing

import jax
import jax.numpy as jnp

from xxm.core.chains.discrete import DiscreteChain, DiscreteChainMarginals
from xxm.core.chains.gaussian import (
    GaussianChainMarginals,
    GaussianPairPotential,
)
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.models.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.models.gaussian import StateConditionedGaussian
from xxm.core.optim import gaussian as gaussian_fit
from xxm.lds.core import EmissionsT


class Posterior(typing.NamedTuple):
    """Structured SLDS posterior over aligned discrete and continuous states."""

    discrete: DiscreteChainMarginals  # T discrete states
    continuous: GaussianChainMarginals  # T continuous latents

    def elbo_from_chain(
        self,
        continuous_log_normalizer: jax.Array,
        discrete_prior: DiscreteChain,
    ) -> jax.Array:
        """Evidence lower bound for the structured SLDS posterior."""
        return (
            continuous_log_normalizer
            + self.discrete.expected_log_potential(discrete_prior)
            + self.discrete.entropy()
        )


class GaussianLinearSwitchingDynamics(typing.NamedTuple):
    r"""State-dependent linear Gaussian latent dynamics.

    Under the SLDS convention, state ``z[t]`` indexes the dynamics that generate
    ``x[t]`` from ``x[t-1]``. Consequently, the first transition uses ``z[1]``;
    ``z[0]`` instead indexes the initial latent distribution.
    """

    model: LinearGaussian  # K-batched, input dimension D, output dimension D

    @property
    def num_states(self) -> int:
        return self.model.covariance.shape[0]

    def compute_pair_potentials(self) -> GaussianPairPotential:
        """Return one Gaussian transition potential per discrete state."""
        return GaussianPairPotential.from_linear_conditional(
            self.model,
        )

    def fit_params(
        self,
        posterior: Posterior,
    ) -> typing.Self:
        """Fit one linear-Gaussian dynamics model per incoming state."""
        weights = posterior.discrete.state_probs[1:]  # (T-1, K)

        means = posterior.continuous.means
        second = posterior.continuous.raw_second_moments()
        cross = posterior.continuous.raw_cross_moments()

        def fit_state(weights_k):
            total = jnp.sum(weights_k)  # TODO: Define behavior for zero posterior mass.

            input_mean = jnp.einsum('t,ti->i', weights_k, means[:-1]) / total

            output_mean = jnp.einsum('t,ti->i', weights_k, means[1:]) / total

            input_second = jnp.einsum('t,tij->ij', weights_k, second[:-1]) / total

            output_second = jnp.einsum('t,tij->ij', weights_k, second[1:]) / total

            # raw_cross_moments[t] = E[x_t x_{t+1}^T],
            # while the fitter expects E[x_{t+1} x_t^T].
            output_input = (
                jnp.einsum(
                    't,tij->ij',
                    weights_k,
                    jnp.swapaxes(cross, -1, -2),
                )
                / total
            )

            return gaussian_fit.linear_from_moments(
                input_mean=input_mean,
                output_mean=output_mean,
                input_second_moment=input_second,
                output_second_moment=output_second,
                output_input_moment=output_input,
            )

        linear_gaussian = jax.vmap(
            fit_state,
            in_axes=1,
        )(weights)

        return self._replace(
            model=linear_gaussian,
        )

    def sample_next(
        self, key: jax.Array, previous: jax.Array, state: jax.Array
    ) -> jax.Array:
        """Sample the next latent using the state being entered."""
        return self.model.select(state).sample(key, previous)

    def sample(
        self, key: jax.Array, initial_latent: jax.Array, states: jax.Array
    ) -> jax.Array:
        """Sample subsequent latents from their incoming switching states."""

        def step(carry, state):
            latent, key = carry

            key, sample_key = jax.random.split(key)

            latent = self.sample_next(
                sample_key,
                latent,
                state,
            )

            return (latent, key), latent

        _, subsequent_latents = jax.lax.scan(
            step,
            (initial_latent, key),
            states,
        )

        return jnp.concatenate(
            [
                initial_latent[None],
                subsequent_latents,
            ],
            axis=0,
        )

    def permute(self, permutation: jax.Array) -> typing.Self:
        return self._replace(
            model=self.model.select(permutation),
        )


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""Switching linear dynamical system.

    Each continuous latent has an aligned discrete state. ``z[0]`` selects the
    state-dependent initial distribution of ``x[0]``. For ``t > 0``, ``z[t]``
    selects the dynamics that generate ``x[t]`` from ``x[t-1]``.

    The initial latent distributions are treated as fixed boundary parameters
    during ordinary single-sequence EM.
    """

    state_initial: CategoricalInitial
    transitions: CategoricalTransitions
    latent_initial: StateConditionedGaussian
    dynamics: GaussianLinearSwitchingDynamics
    emissions: EmissionsT

    @property
    def num_states(self) -> int:
        return self.dynamics.num_states

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self:
        """Fit learnable SLDS parameters from a structured posterior."""
        return self.__class__(
            state_initial=self.state_initial.fit_params(
                posterior.discrete,
            ),
            transitions=self.transitions.fit_params(
                posterior.discrete,
            ),
            # Keep the state-dependent boundary distributions fixed.
            latent_initial=self.latent_initial,
            dynamics=self.dynamics.fit_params(
                posterior,
            ),
            emissions=self.emissions.fit_params(
                observations,
                posterior.continuous,
            ),
        )

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Sample aligned switching states, latents, and observations."""
        if num_steps < 1:
            raise ValueError('SLDS sampling requires at least one time step.')

        (
            key_state_initial,
            key_states,
            key_latent_initial,
            key_latents,
            key_observations,
        ) = jax.random.split(
            key,
            5,
        )

        initial_state = self.state_initial.sample(key_state_initial)
        states = self.transitions.sample(key_states, initial_state, num_steps)

        initial_latent = self.latent_initial.sample(key_latent_initial, states[0])
        latents = self.dynamics.sample(key_latents, initial_latent, states[1:])

        observations = self.emissions.sample(key_observations, latents)

        return states, latents, observations

    def permute(self, permutation: jax.Array) -> typing.Self:
        return self._replace(
            state_initial=self.state_initial.permute(permutation),
            transitions=self.transitions.permute(permutation),
            latent_initial=self.latent_initial.permute(permutation),
            dynamics=self.dynamics.permute(permutation),
        )
