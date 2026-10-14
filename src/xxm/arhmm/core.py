r"""Autoregressive Hidden Markov Model."""

from __future__ import annotations

import typing

import jax

from xxm.core import batch
from xxm.core.data import Sequences
from xxm.core.dists.discrete import (
    DiscreteChainMarginals as Posterior,
)
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.lengths import LengthsSpec

from .data import ARDatasetCollection
from .emissions import AREmissionsCollection, ConditionalDistT


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
    emissions: AREmissionsCollection[ConditionalDistT]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        batch.require_same_shape(
            self.initial.batch_shape,
            self.transitions.batch_shape,
            self.emissions.batch_shape,
        )
        if self.initial.num_states != self.transitions.num_states or any(
            emission.num_states != self.initial.num_states
            for emission in self.emissions.emissions
        ):
            raise ValueError('AR-HMM components must share their number of states')
        _ = self.num_lags
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
        values = [emission.num_lags for emission in self.emissions.emissions]
        assert all(value == values[0] for value in values[1:])
        return values[0]

    def observation_dim(self, session: int | None = None) -> int:
        return self.emissions.get(session).observation_dim

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
        *,
        session: int | None = None,
    ) -> Sample:

        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(
            key_states,
            ls,
        )

        emissions, ls = batch.cartesian_broadcast(self.emissions.get(session), ls)

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
        *,
        session: int | None = None,
    ) -> Sample:

        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(
            key_states,
            ls,
        )

        emissions, ls = batch.cartesian_broadcast(self.emissions.get(session), ls)

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
        posterior: Posterior,
        data: ARDatasetCollection,
    ) -> typing.Self:
        """Fit shared latent parameters and session-specific AR emissions."""

        if data.input_dim != 0:
            raise ValueError('AR-HMM does not currently support known external inputs')

        if data.num_lags != self.num_lags:
            raise ValueError('Mismatch between data.num_lags and model.num_lags')

        expected_batch = (*self.batch_shape, data.partition.total)
        if posterior.batch_shape != expected_batch:
            raise ValueError(
                f'posterior batch must be {expected_batch}; got {posterior.batch_shape}'
            )

        if posterior.max_steps != data.max_steps:
            raise ValueError('posterior and data must share their time dimension')

        if posterior.num_states != self.num_states:
            raise ValueError('posterior and model must share their number of states')

        return type(self)(
            initial=self.initial.fit_params(posterior),
            transitions=self.transitions.fit_params(posterior),
            emissions=self.emissions.fit_params(
                posterior,
                data.weighted_observations(),
            ),
        )
