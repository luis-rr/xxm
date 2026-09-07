r"""Single-sequence homogeneous Hidden Markov Model."""

from __future__ import annotations

import typing

import jax

from xxm.core.chains.discrete import DiscreteChainMarginals as Posterior
from xxm.core.emissions.discrete import ContinuationEmissions, Emissions
from xxm.core.latents.discrete import CategoricalInitial, CategoricalTransitions

EmissionsT = typing.TypeVar('EmissionsT', bound=Emissions)

ContinuationEmissionsT = typing.TypeVar(
    'ContinuationEmissionsT',
    bound=ContinuationEmissions,
)


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""
    Hidden Markov Model with discrete latent states.

    For memoryless emissions,

    $$p(z_{0:T-1}, y_{0:T-1}) =
    p(z_0)
    \prod_{t=0}^{T-2} p(z_{t+1}\mid z_t)
    \prod_{t=0}^{T-1} p(y_t\mid z_t).$$

    `initial` stores $p(z_0)$, `transitions` stores $p(z_{t+1}\mid z_t)$,
    and `emissions` stores the state-conditional observation model.
    Autoregressive emissions additionally condition on observation history.
    """

    initial: CategoricalInitial
    transitions: CategoricalTransitions
    emissions: EmissionsT

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.initial.num_states

    def permute(
        self,
        permutation: jax.Array,
    ) -> Model:
        """Relabel discrete states by permutation."""
        return Model(
            initial=self.initial.permute(permutation),
            transitions=self.transitions.permute(permutation),
            emissions=self.emissions.permute(permutation),
        )

    def sample_states(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """Sample state trajectory $z_{0:T-1}$ from the prior."""
        key_initial, key_transitions = jax.random.split(key)

        initial_state = self.initial.sample(
            key_initial,
        )

        return self.transitions.sample(
            key_transitions,
            initial_state,
            num_steps,
        )

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> tuple[jax.Array, jax.Array]:
        r"""Sample complete trajectory $(z_{0:T-1}, y_{0:T-1})$ autonomously."""
        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(
            key_states,
            num_steps,
        )

        observations = self.emissions.sample(key_observations, states)

        return states, observations

    def sample_continuation(
        self: Model[ContinuationEmissionsT],
        key: jax.Array,
        num_steps: int,
        initial_history: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample states and observations conditional on an explicit history."""
        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(key_states, num_steps)

        observations = self.emissions.sample_continuation(
            key_observations, states, initial_history
        )

        return states, observations

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> Model[EmissionsT]:
        """
        Perform the EM M-step from posterior state and pair marginals.
        """
        return Model(
            initial=self.initial.fit_params(posterior),
            transitions=self.transitions.fit_params(posterior),
            emissions=self.emissions.fit_params(
                observations,
                posterior,
            ),
        )
