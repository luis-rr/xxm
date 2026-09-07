import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.chains.discrete import DiscreteChainMarginals
from xxm.core.chains.gaussian import (
    GaussianChainMarginals,
    GaussianPairPotential,
)
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.emissions.continuous import EmissionsT
from xxm.core.models.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.models.gaussian import StateConditionedGaussian
from xxm.core.optim import gaussian as gaussian_fit


class Posterior(typing.NamedTuple):
    """Structured SLDS posterior over aligned discrete and continuous states."""

    discrete: DiscreteChainMarginals  # T discrete states
    continuous: GaussianChainMarginals  # T continuous latents

    def permute(self, permutation: jax.Array) -> typing.Self:
        """Relabel the discrete latent states."""
        return self._replace(
            discrete=self.discrete.permute(permutation),
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
        covariances = posterior.continuous.covariances
        cross_covariances = posterior.continuous.cross_covariances

        def fit_state(
            weights_k: jax.Array,
            current_model: LinearGaussian,
        ) -> LinearGaussian:
            true_total = jnp.sum(weights_k)

            # Only used to make the candidate calculation defined when the
            # state has zero posterior mass.
            total = jnp.where(
                true_total > 0,
                true_total,
                1.0,
            )

            normalized = weights_k / total

            input_mean = jnp.einsum(
                't,ti->i',
                normalized,
                means[:-1],
            )

            output_mean = jnp.einsum(
                't,ti->i',
                normalized,
                means[1:],
            )

            input_residuals = means[:-1] - input_mean

            output_residuals = means[1:] - output_mean

            input_covariance = jnp.einsum(
                't,tij->ij',
                normalized,
                covariances[:-1],
            ) + jnp.einsum(
                't,ti,tj->ij',
                normalized,
                input_residuals,
                input_residuals,
            )

            output_covariance = jnp.einsum(
                't,tij->ij',
                normalized,
                covariances[1:],
            ) + jnp.einsum(
                't,ti,tj->ij',
                normalized,
                output_residuals,
                output_residuals,
            )

            # cross_covariances[t] = Cov(x_t, x_{t+1}),
            # while the fitter expects Cov(x_{t+1}, x_t).
            output_input_covariance = jnp.einsum(
                't,tij->ij',
                normalized,
                jnp.swapaxes(
                    cross_covariances,
                    -1,
                    -2,
                ),
            ) + jnp.einsum(
                't,ti,tj->ij',
                normalized,
                output_residuals,
                input_residuals,
            )

            fitted_model = gaussian_fit.linear_from_centered_moments(
                input_mean=input_mean,
                output_mean=output_mean,
                input_covariance=input_covariance,
                output_covariance=output_covariance,
                output_input_covariance=output_input_covariance,
                ridge=1e-6,
            )

            return jax.tree.map(
                lambda fitted, current: jnp.where(
                    true_total > 0,
                    fitted,
                    current,
                ),
                fitted_model,
                current_model,
            )

        model = jax.vmap(
            fit_state,
            in_axes=(1, 0),
        )(
            weights,
            self.model,
        )

        return self._replace(
            model=model,
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
        """Express the states in a reordered coordinate system."""
        return self._replace(
            model=self.model.select(permutation),
        )

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        inverse = alignment.inverse()

        return self._replace(
            model=(self.model.compose_input(inverse).compose_output(alignment)),
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

    def align(
        self,
        alignment: Affine,
    ) -> typing.Self:
        latent_dim = self.dynamics.model.output_dim

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
