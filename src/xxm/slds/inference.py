import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.core.chains.discrete import DiscreteChain, DiscreteChainMarginals
from xxm.core.chains.gaussian import (
    GaussianChain,
    GaussianChainMarginals,
    GaussianPairPotential,
    GaussianPotential,
)

from .model import Model, Posterior


class ContinuousPotentials(typing.NamedTuple):
    dynamics: GaussianPairPotential
    observations: GaussianPotential
    initial: GaussianPotential

    @classmethod
    def from_model(cls, model: Model, observations: jax.Array) -> 'ContinuousPotentials':

        dynamics_potentials = model.dynamics.get_pair_potentials()
        observation_potentials = model.emissions.get_potential(observations)
        initial_potential = GaussianPotential.from_moments(model.latent_initial.model)

        return ContinuousPotentials(
            dynamics=dynamics_potentials,
            observations=observation_potentials,
            initial=initial_potential,
        )

    def to_chain(
        self,
        state_marginals: jax.Array,
    ) -> GaussianChain:

        pair_potentials = self.dynamics.weighted_sum(
            state_marginals,
        )

        chain = GaussianChain.from_pair_potentials(
            self.initial,
            pair_potentials,
        )

        chain = chain.add_local_potential(self.observations)

        return chain

    def inference(
        self,
        state_probs: jax.Array,
    ) -> tuple[GaussianChainMarginals, jax.Array]:
        return self.to_chain(state_probs).forward_backward()

    def infer_state_log_potentials(
        self,
        state_probs: jax.Array,
    ) -> jax.Array:
        posterior, _ = self.inference(state_probs)
        return self.get_expected_state_log_potentials(posterior)

    def get_expected_state_log_potentials(
        self,
        posterior: GaussianChainMarginals,
    ) -> jax.Array:
        return self.dynamics.expected_log_potentials(posterior)


class DiscretePotentials(typing.NamedTuple):
    initial: jax.Array
    transitions: jax.Array

    @classmethod
    def from_model(cls, model: Model, num_time_steps: int) -> 'DiscretePotentials':
        return DiscretePotentials(
            initial=model.state_initial.model.probs,
            transitions=model.transitions.model.broadcast((num_time_steps - 1,)).probs,
        )

    def to_chain(
        self,
        state_log_potentials,
    ) -> DiscreteChain:

        chain = DiscreteChain(
            initial_probs=self.initial,
            transition_probs=self.transitions,
            state_log_potentials=state_log_potentials,
        )

        return chain

    def inference(
        self,
        state_log_potentials: jax.Array,
    ) -> tuple[DiscreteChainMarginals, jax.Array]:
        return self.to_chain(state_log_potentials).forward_backward()

    def infer_state_probs(
        self,
        state_log_potentials: jax.Array,
    ) -> jax.Array:
        posterior, _ = self.inference(state_log_potentials)
        return posterior.state_probs

    def prior(self) -> tuple[DiscreteChainMarginals, jax.Array]:
        num_time_steps = self.transitions.shape[0] + 1

        state_log_potentials = jnp.zeros(
            (num_time_steps, self.initial.shape[0]),
            dtype=self.initial.dtype,
        )

        posterior, log_normalizer = self.inference(state_log_potentials)

        return posterior, log_normalizer


def elbo(
    model: Model,
    posterior: Posterior,
    continuous_log_normalizer: jax.Array,
) -> jax.Array:
    """Evidence lower bound for the structured SLDS posterior."""

    discrete = posterior.discrete

    expected_initial_log_prob = jnp.sum(
        jsp.special.xlogy(
            discrete.state_probs[0],
            model.state_initial.model.probs,
        )
    )

    expected_transition_log_prob = jnp.sum(
        jsp.special.xlogy(
            discrete.pair_probs,
            model.transitions.model.probs,
        )
    )

    return (
        continuous_log_normalizer
        + expected_initial_log_prob
        + expected_transition_log_prob
        + discrete.entropy()
    )


def infer_variational(
    model: Model,
    observations: jax.Array,
    num_iters: int,
) -> tuple[Posterior, jax.Array]:

    num_time_steps = observations.shape[0]

    cont_potentials = ContinuousPotentials.from_model(
        model,
        observations,
    )
    disc_potentials = DiscretePotentials.from_model(
        model,
        num_time_steps - 1,
    )

    def step(_, disc_posterior):
        cont_posterior, _ = cont_potentials.inference(disc_posterior.state_probs)

        state_log_potentials = cont_potentials.get_expected_state_log_potentials(cont_posterior)

        disc_posterior, _ = disc_potentials.inference(state_log_potentials)

        return disc_posterior

    disc_posterior, _ = disc_potentials.prior()

    disc_posterior = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        disc_posterior,
    )

    cont_posterior, cont_log_normalizer = cont_potentials.inference(disc_posterior.state_probs)

    posterior = Posterior(
        discrete=disc_posterior,
        continuous=cont_posterior,
    )

    elbo_value = elbo(
        model=model,
        posterior=posterior,
        continuous_log_normalizer=cont_log_normalizer,
    )

    return posterior, elbo_value
