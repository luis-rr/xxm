import typing

import jax
import jax.numpy as jnp

from xxm.core.discrete.chain import DiscreteChain, DiscreteChainMarginals
from xxm.core.gaussian.chain import (
    GaussianChain,
    GaussianChainMarginals,
    GaussianPairPotential,
    GaussianPotential,
)

from .core import Model, Posterior


class ContinuousPotentials(typing.NamedTuple):
    dynamics: GaussianPairPotential
    observations: GaussianPotential
    initial: GaussianPotential

    @classmethod
    def from_model(cls, model: Model, observations: jax.Array) -> 'ContinuousPotentials':

        dynamics_potentials = model.dynamics.get_pair_potentials()
        observation_potentials = model.emissions.get_potential(observations)
        initial_potential = GaussianPotential.from_moments(
            model.latent_initial.mean,
            model.latent_initial.covariance,
        )

        return ContinuousPotentials(
            dynamics=dynamics_potentials,
            observations=observation_potentials,
            initial=initial_potential,
        )

    def to_chain(
        self,
        state_marginals: jax.Array,
    ) -> GaussianChain:

        pair_potentials = self.dynamics.expected(
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
    ) -> GaussianChainMarginals:
        return self.to_chain(state_probs).forward_backward()

    def infer_state_log_potentials(
        self,
        state_probs: jax.Array,
    ) -> jax.Array:
        posterior = self.inference(state_probs)
        return self.get_expected_state_log_potentials(posterior)

    def get_expected_state_log_potentials(
        self,
        posterior: GaussianChainMarginals,
    ) -> jax.Array:
        return self.dynamics.expected_log_likelihoods(posterior)


class DiscretePotentials(typing.NamedTuple):
    initial: jax.Array
    transitions: jax.Array

    @classmethod
    def from_model(cls, model: Model, num_time_steps: int) -> 'DiscretePotentials':
        return DiscretePotentials(
            initial=model.state_initial.initial_probs,
            transitions=model.transitions.broadcast((num_time_steps - 1,)),
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
    ) -> DiscreteChainMarginals:
        return self.to_chain(state_log_potentials).forward_backward()

    def infer_state_probs(
        self,
        state_log_potentials: jax.Array,
    ) -> jax.Array:
        return self.inference(state_log_potentials).state_marginals

    def prior(self) -> DiscreteChainMarginals:
        num_time_steps = self.transitions.shape[0] + 1

        state_log_potentials = jnp.zeros(
            (num_time_steps, self.initial.shape[0]),
            dtype=self.initial.dtype,
        )

        return self.inference(state_log_potentials)


def inference_exact(
    model: Model,
    observations: jax.Array,
    num_iters: int,
) -> Posterior:

    num_time_steps = observations.shape[0]

    if num_time_steps < 2:
        raise ValueError('SLDS inference requires at least two time steps')

    if num_iters < 0:
        raise ValueError('Number of inference iterations must be non-negative')

    cont_potentials = ContinuousPotentials.from_model(model, observations)
    disc_potentials = DiscretePotentials.from_model(model, num_time_steps - 1)

    def step(_, disc_posterior):
        cont_posterior = cont_potentials.inference(disc_posterior.state_marginals)

        state_log_potentials = cont_potentials.get_expected_state_log_potentials(cont_posterior)

        return disc_potentials.inference(state_log_potentials)

    disc_posterior = disc_potentials.prior()

    disc_posterior = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        disc_posterior,
    )

    cont_posterior = cont_potentials.inference(disc_posterior.state_marginals)

    return Posterior(
        discrete=disc_posterior,
        continuous=cont_posterior,
    )
