r"""Homogeneous Hidden Markov models."""

from __future__ import annotations

import typing

import jax

from xxm.core import batch
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.dists.discrete import DiscreteChainMarginals as Posterior
from xxm.core.emissions.collection import DiscreteEmissionsCollection
from xxm.core.emissions.discrete import Emissions
from xxm.core.latents.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.lengths import LengthsSpec

EmissionsT = typing.TypeVar('EmissionsT', bound=Emissions)


class Sample(typing.NamedTuple):
    states: Sequences
    observations: Sequences


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
    emissions: DiscreteEmissionsCollection[EmissionsT]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Independent parameter batches shared by all model components."""
        batch.require_same_shape(
            self.initial.batch_shape,
            self.transitions.batch_shape,
            self.emissions.batch_shape,
        )
        if self.initial.num_states != self.transitions.num_states or any(
            emission.num_states != self.initial.num_states
            for emission in self.emissions.emissions
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
    ) -> typing.Self:
        """Relabel discrete states by permutation."""
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
        """Sample padded state trajectories for a batch of requested lengths."""

        model, ls = batch.cartesian_broadcast(self, ls)

        key_initial, key_transitions = jax.random.split(key)

        initial_state = model.initial.sample(
            key_initial,
        )

        return model.transitions.sample(
            key_transitions,
            initial_state,
            ls,
        )

    def sample(
        self,
        key: jax.Array,
        ls: LengthsSpec,
        *,
        session: int | None = None,
    ) -> Sample:
        r"""Sample padded state and observation trajectories."""

        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(
            key_states,
            ls,
        )

        emissions, ls = batch.cartesian_broadcast(self.emissions.get(session), ls)

        observations = emissions.sample(
            key_observations,
            states,
        )

        return Sample(
            states=states,
            observations=observations,
        )

    def fit_params(
        self,
        posterior: Posterior,
        data: DatasetCollection,
    ) -> typing.Self:
        """Fit shared latent parameters and session-specific emissions."""

        if data.input_dim != 0:
            raise ValueError('stationary HMMs require dataset.input_dim == 0')

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
