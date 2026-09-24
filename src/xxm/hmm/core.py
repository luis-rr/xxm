r"""Homogeneous Hidden Markov models."""

from __future__ import annotations

import typing

import jax

from xxm.core import batch
from xxm.core.chains.discrete import DiscreteChainMarginals as Posterior
from xxm.core.data import Dataset
from xxm.core.emissions.discrete import Emissions
from xxm.core.latents.discrete import CategoricalInitial, CategoricalTransitions

EmissionsT = typing.TypeVar('EmissionsT', bound=Emissions)


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""
    Hidden Markov Model with discrete latent states.

    $$p(z_{0:T-1}, y_{0:T-1}) =
    p(z_0)
    \prod_{t=0}^{T-2} p(z_{t+1}\mid z_t)
    \prod_{t=0}^{T-1} p(y_t\mid z_t).$$

    `initial` stores $p(z_0)$, `transitions` stores $p(z_{t+1}\mid z_t)$,
    and `emissions` stores the state-conditional observation model.
    """

    initial: CategoricalInitial
    transitions: CategoricalTransitions
    emissions: EmissionsT

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Independent parameter batches shared by all model components."""
        batch.require_same_shape(
            self.initial.batch_shape,
            self.transitions.batch_shape,
            self.emissions.batch_shape,
        )
        if (
            self.initial.num_states != self.transitions.num_states
            or self.initial.num_states != self.emissions.num_states
        ):
            raise ValueError('HMM components must share their number of states')
        return self.initial.batch_shape

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.initial.num_states

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> Model:
        """Relabel discrete states by permutation."""
        return Model(
            initial=self.initial.permute_states(permutation),
            transitions=self.transitions.permute_states(permutation),
            emissions=self.emissions.permute_states(permutation),
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

    def fit_params(
        self,
        data: Dataset,
        posterior: Posterior,
    ) -> Model[EmissionsT]:
        """Pool independent sequences, preserving the model batch prefix."""
        if data.input_dim != 0:
            raise ValueError('stationary HMMs require data.input_dim == 0')

        model_batch_shape = self.batch_shape
        expected_batch = (*model_batch_shape, *data.batch_shape)
        if posterior.batch_shape != expected_batch:
            raise ValueError(
                f'posterior batch must be {expected_batch}; got {posterior.batch_shape}'
            )

        if posterior.num_steps != data.num_steps:
            raise ValueError('posterior and dataset must share their time dimension')

        if posterior.num_states != self.num_states:
            raise ValueError('posterior and model must share their number of states')

        working_data = data.broadcast(model_batch_shape, axis=0)

        return Model(
            initial=self.initial.fit_params(posterior),
            transitions=self.transitions.fit_params(
                posterior,
                weights=working_data.valid_transitions().materialize(
                    data.num_steps - 1
                ),
            ),
            emissions=self.emissions.fit_params(
                working_data.weighted_observations(),
                posterior,
            ),
        )
