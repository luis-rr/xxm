"""
Switching Linear Dynamical System: discrete and continuous latent states.
"""

import typing

import jax

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.chains.discrete import DiscreteChainMarginals
from xxm.core.chains.gaussian import (
    GaussianChainMarginals,
)
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.emissions.collection import EmissionsCollection
from xxm.core.emissions.continuous import EmissionsT
from xxm.core.latents.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.latents.gaussian import StateConditionedGaussian
from xxm.core.latents.switching import GaussianLinearSwitchingDynamics
from xxm.core.lengths import Lengths, LengthsSpec


class Sample(typing.NamedTuple):
    states: Sequences
    latents: Sequences
    observations: Sequences


class Posterior(typing.NamedTuple):
    r"""
    Structured mean-field posterior over discrete states $z$ and continuous latents $x$.

    The approximation factorizes as $q(z,x)=q(z)q(x)$; each factor retains
    its temporal chain structure. `discrete` and `continuous` store its marginals.
    """

    discrete: DiscreteChainMarginals  # T discrete states
    continuous: GaussianChainMarginals  # T continuous latents

    @property
    def lengths(self) -> Lengths:
        return self.continuous.lengths

    @property
    def max_steps(self) -> int:
        return self.continuous.max_steps

    @property
    def batch_shape(self) -> tuple[int, ...]:
        batch.require_same_shape(self.discrete.batch_shape, self.continuous.batch_shape)
        if self.discrete.max_steps != self.continuous.max_steps:
            raise ValueError(
                'discrete and continuous posteriors must share time dimensions'
            )
        return self.discrete.batch_shape

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

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        r"""Relabel discrete states $z$ by permutation."""
        return self._replace(
            discrete=self.discrete.permute_states(permutation),
        )


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""Switching linear dynamical system.

    Each continuous latent has an aligned discrete state. ``z[0]`` selects the
    state-dependent initial distribution of ``x[0]``. For ``t > 0``, ``z[t]``
    selects the dynamics that generate ``x[t]`` from ``x[t-1]``.

    Inputs at time t affect the transition into x[t]. Parameters are shared
    across independent sequences, including the learned boundary distributions.
    """

    state_initial: CategoricalInitial
    transitions: CategoricalTransitions
    latent_initial: StateConditionedGaussian
    dynamics: GaussianLinearSwitchingDynamics
    emissions: EmissionsCollection[EmissionsT]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        batch.require_same_shape(
            self.state_initial.batch_shape,
            self.transitions.batch_shape,
            self.latent_initial.batch_shape,
            self.dynamics.batch_shape,
            self.emissions.batch_shape,
        )
        if any(
            component.num_states != self.num_states
            for component in (self.state_initial, self.transitions, self.latent_initial)
        ):
            raise ValueError('SLDS components must share their number of states')
        if self.latent_initial.dist.variable_dim != self.dynamics.latent_dim:
            raise ValueError('initial and dynamics latent dimensions must match')
        if any(
            emission.latent_dim != self.dynamics.latent_dim
            for emission in self.emissions.emissions
        ):
            raise ValueError('emission and dynamics latent dimensions must match')
        return self.state_initial.batch_shape

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
        return self.dynamics.num_states

    def fit_params(
        self,
        data: DatasetCollection,
        posterior: Posterior,
    ) -> typing.Self:
        """Pool sequence statistics while preserving the model batch prefix."""
        dataset = data.single()
        expected_batch = (*self.batch_shape, *dataset.batch_shape)
        if posterior.batch_shape != expected_batch:
            raise ValueError(f'posterior batch must be {expected_batch}')
        if posterior.continuous.max_steps != dataset.max_steps:
            raise ValueError('posterior and dataset time dimensions must match')
        if dataset.input_dim != self.dynamics.input_dim:
            raise ValueError('dataset input dimension must match dynamics')
        working_data = dataset.broadcast(self.batch_shape)
        return self.__class__(
            state_initial=self.state_initial.fit_params(
                posterior.discrete,
            ),
            transitions=self.transitions.fit_params(
                posterior.discrete,
            ),
            latent_initial=self.latent_initial.fit_params(
                posterior.discrete,
                posterior.continuous,
            ),
            dynamics=self.dynamics.fit_params(
                discrete=posterior.discrete,
                continuous=posterior.continuous,
                inputs=working_data.lengths.transitions().clear(
                    working_data.transition_inputs()
                ),
            ),
            emissions=EmissionsCollection(
                (
                    self.emissions.single().fit_params(
                        working_data.weighted_observations(),
                        posterior.continuous,
                    ),
                )
            ),
        )

    def sample(
        self,
        key: jax.Array,
        *,
        inputs: Sequences,
        session: int | None = None,
    ) -> Sample:
        """Sample aligned switching states, latents, and observations."""

        model, inputs = batch.cartesian_broadcast(self, inputs)
        ls = LengthsSpec(inputs.lengths, inputs.max_steps)

        (
            key_state_initial,
            key_states,
            key_latent_initial,
            key_latents,
            key_observations,
        ) = jax.random.split(key, 5)

        initial_state = model.state_initial.sample(key_state_initial)

        states = model.transitions.sample(key_states, initial_state, ls)

        initial_latent = model.latent_initial.sample(
            key_latent_initial, states.values[..., 0]
        )

        latents = model.dynamics.sample(
            key_latents,
            initial_latent,
            states,
            inputs=inputs,
        )

        observations = model.emissions.get(session).sample(key_observations, latents)

        return Sample(
            states=states,
            latents=latents,
            observations=observations,
        )

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """
        Relabel all discrete-state-dependent parameters by permutation.
        """
        return self._replace(
            state_initial=self.state_initial.permute_states(permutation),
            transitions=self.transitions.permute_states(permutation),
            latent_initial=self.latent_initial.permute_states(permutation),
            dynamics=self.dynamics.permute_states(permutation),
        )

    def align(
        self,
        alignment: Affine,
    ) -> typing.Self:
        """Express continuous latent parameters in aligned coordinates."""
        latent_dim = self.dynamics.dist.output_dim

        batch.require_same_shape(
            alignment.batch_shape,
            self.batch_shape,
        )

        if alignment.input_shape != (latent_dim,) or alignment.output_dim != latent_dim:
            raise ValueError(
                'SLDS latent alignment requires an affine map '
                f'({latent_dim},) -> ({latent_dim},), '
                f'got {alignment.input_shape} -> '
                f'({alignment.output_dim},)'
            )

        return self._replace(
            latent_initial=self.latent_initial.align(alignment),
            dynamics=self.dynamics.align(alignment),
            emissions=self.emissions.compose_input(alignment),
        )
