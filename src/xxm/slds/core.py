"""Switched Linear Dynamical System: discrete and continuous latent states."""

import typing

import jax

from xxm.core.affine import Affine
from xxm.core.chains.discrete import DiscreteChainMarginals
from xxm.core.chains.gaussian import (
    GaussianChainMarginals,
)
from xxm.core.emissions.continuous import EmissionsT
from xxm.core.latents.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.latents.gaussian import StateConditionedGaussian
from xxm.core.latents.switching import GaussianLinearSwitchingDynamics


class Posterior(typing.NamedTuple):
    r"""SLDS posterior over discrete states $z$ and continuous latents $x$."""

    discrete: DiscreteChainMarginals  # T discrete states
    continuous: GaussianChainMarginals  # T continuous latents

    def permute(self, permutation: jax.Array) -> typing.Self:
        r"""Relabel discrete states $z$ by permutation."""
        return self._replace(
            discrete=self.discrete.permute(permutation),
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
        """Number of discrete states $K$."""
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
                discrete=posterior.discrete,
                continuous=posterior.continuous,
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
        """Relabel the discrete state-dependent initial and transition parameters."""
        return self._replace(
            state_initial=self.state_initial.permute(permutation),
            transitions=self.transitions.permute(permutation),
            latent_initial=self.latent_initial.permute(permutation),
            dynamics=self.dynamics.permute(permutation),
        )

    def align(
        self,
        alignment: Affine,
    ) -> typing.Self:
        """Express continuous latent parameters in aligned coordinates."""
        latent_dim = self.dynamics.dist.output_dim

        if alignment.batch_shape:
            raise ValueError('SLDS latent alignment must be unbatched')

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
