r"""Autoregressive Hidden Markov Model."""

from __future__ import annotations

import typing

import jax

from xxm.core import batch
from xxm.core.chains.discrete import (
    DiscreteChainMarginals as Posterior,
)
from xxm.core.data import Sequences
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.lengths import LengthsSpec

from .data import ARDataset
from .emissions import AREmissions, ConditionalDistT


class Sample(typing.NamedTuple):
    states: Sequences
    observations: Sequences


class Model(
    typing.NamedTuple,
    typing.Generic[ConditionalDistT],
):
    """HMM whose state selects an autoregressive conditional distribution."""

    initial: CategoricalInitial
    transitions: CategoricalTransitions
    emissions: AREmissions[ConditionalDistT]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        batch.require_same_shape(
            self.initial.batch_shape,
            self.transitions.batch_shape,
            self.emissions.batch_shape,
        )
        return self.initial.batch_shape

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(
        self,
        permutation,
        axis: int = 0,
    ) -> typing.Self:
        return batch.permute(
            self,
            permutation,
            axis=axis,
        )

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        return batch.move_axis(
            self,
            source,
            destination,
        )

    @property
    def num_states(self) -> int:
        return self.initial.num_states

    @property
    def num_lags(self) -> int:
        return self.emissions.num_lags

    @property
    def observation_dim(self) -> int:
        return self.emissions.observation_dim

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        return self._replace(
            initial=self.initial.permute_states(permutation),
            transitions=self.transitions.permute_states(permutation),
            emissions=self.emissions.permute_states(permutation),
        )

    def sample_states(
        self,
        key: jax.Array,
        ls: LengthsSpec,
    ) -> Sequences:

        model, lengths = batch.cartesian_broadcast(self, ls)

        key_initial, key_transitions = jax.random.split(key)

        initial_state = model.initial.sample(
            key_initial,
        )

        return model.transitions.sample(
            key_transitions,
            initial_state,
            lengths,
        )

    def sample(
        self,
        key: jax.Array,
        ls: LengthsSpec,
    ) -> Sample:

        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(
            key_states,
            ls,
        )

        emissions, ls = batch.cartesian_broadcast(self.emissions, ls)

        observations = emissions.sample(key_observations, states)

        return Sample(
            states=states,
            observations=observations,
        )

    def sample_continuation(
        self,
        key: jax.Array,
        ls: LengthsSpec,
        initial_history: jax.Array,
    ) -> Sample:

        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(
            key_states,
            ls,
        )

        emissions, ls = batch.cartesian_broadcast(self.emissions, ls)

        initial_history = batch.broadcast_array(
            initial_history,
            self.batch_shape,
            axis=0,
        )

        observations = emissions.sample_continuation(
            key_observations,
            states,
            initial_history,
        )

        return Sample(
            states=states,
            observations=observations,
        )

    def fit_params(
        self,
        data: ARDataset,
        posterior: Posterior,
    ) -> typing.Self:
        """Pool independent AR sequences while preserving model batch."""
        expected_batch = (
            *self.batch_shape,
            *data.batch_shape,
        )

        if posterior.batch_shape != expected_batch:
            raise ValueError(
                f'posterior batch must be {expected_batch}; got {posterior.batch_shape}'
            )

        working_data = data

        if self.batch_shape:
            working_data = data.broadcast(
                self.batch_shape,
                axis=0,
            )

        return self._replace(
            initial=self.initial.fit_params(
                posterior,
            ),
            transitions=self.transitions.fit_params(
                posterior,
            ),
            emissions=self.emissions.fit_params(
                working_data,
                posterior,
            ),
        )
