r"""Linear Dynamical Systems."""

from __future__ import annotations

import typing

import jax
from jax import numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.chains.gaussian import GaussianPairPotential, GaussianPotential
from xxm.core.data import Dataset
from xxm.core.emissions.continuous import EmissionsT
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""Linear Dynamical System model.

    $$x_0 \sim p(x_0), \qquad
    x_t\mid x_{t-1},u_t
    \sim \mathcal{N}(Ax_{t-1}+Bu_t+b,Q), \qquad
    y_t\mid x_t \sim p(y_t\mid x_t).$$

    `initial` stores $p(x_0)$, `dynamics` stores the controlled linear-Gaussian
    state evolution, and `emissions` stores $p(y_t\mid x_t)$.
    """

    initial: GaussianInitial
    dynamics: GaussianLinearDynamics
    emissions: EmissionsT

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Common structural batch of all top-level components."""
        batch_shape = self.initial.batch_shape
        batch.require_same(
            batch_shape,
            self.dynamics.batch_shape,
            self.emissions.batch_shape,
        )
        return batch_shape

    def select(self, index: batch.SelT) -> typing.Self:
        """Select independent models along their batch dimensions."""
        index = batch.selection(index, len(self.batch_shape))
        return self.__class__(
            self.initial.select(index),
            self.dynamics.select(index),
            self.emissions.select(index),
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated model batch dimensions."""
        shape, axis = batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            self.initial.broadcast(shape, axis),
            self.dynamics.broadcast(shape, axis),
            self.emissions.broadcast(shape, axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton model batch dimensions."""
        axes = batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            self.initial.squeeze(axes),
            self.dynamics.squeeze(axes),
            self.emissions.squeeze(axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder independent models along one batch axis."""
        axis = batch.axis_index(axis, len(self.batch_shape))
        permutation = batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            self.initial.permute(permutation, axis),
            self.dynamics.permute(permutation, axis),
            self.emissions.permute(permutation, axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one model batch axis to another batch position."""
        source = batch.axis_index(source, len(self.batch_shape))
        destination = batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            self.initial.move_axis(source, destination),
            self.dynamics.move_axis(source, destination),
            self.emissions.move_axis(source, destination),
        )

    def compute_initial_potential(self) -> GaussianPotential:
        """Return the canonical potential for the initial latent distribution."""
        return GaussianPotential.from_moments(self.initial.dist)

    def compute_pair_potentials(
        self,
        inputs: jax.Array,
    ) -> GaussianPairPotential:
        """Return controlled pair potentials for transition-aligned inputs."""
        conditional = self.dynamics.conditional(inputs)
        return GaussianPairPotential.from_linear_conditional(conditional)

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
        *,
        inputs: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample a complete latent and observation trajectory."""
        _ = self.batch_shape
        self.dynamics.validate_trajectory_inputs(inputs, num_steps)

        key_initial, key_latents, key_observations = jax.random.split(
            key,
            3,
        )

        initial_latent = self.initial.sample(key_initial)

        latents = self.dynamics.sample(
            key_latents,
            initial_latent,
            num_steps,
            inputs=inputs,
        )

        observations = self.emissions.sample(key_observations, latents)

        return latents, observations

    def compute_prior_means(
        self,
        num_steps: int,
        *,
        inputs: jax.Array,
    ) -> jax.Array:
        r"""Compute prior latent means $\mathbb{E}[x_t]$."""
        _ = self.batch_shape
        return self.dynamics.mean_trajectory(
            self.initial.dist.mean, num_steps, inputs=inputs
        )

    def log_joint(
        self,
        data: Dataset,
        latents: jax.Array,
    ) -> jax.Array:
        r"""
        Compute the joint log density across independent sequences.

        Observation factors are included only where `data.mask` is true.
        Latent initial and transition factors are included throughout each valid
        sequence prefix. Independent sequence log densities are summed.
        """
        model_batch_shape = self.batch_shape
        data_batch_shape = data.batch_shape
        working_model, working_data = batch.cartesian_broadcast(self, data)

        expected_latent_shape = (
            *working_data.batch_shape,
            working_data.num_steps,
            working_model.dynamics.latent_dim,
        )
        if latents.shape != expected_latent_shape:
            raise ValueError(
                f'latents must have shape {expected_latent_shape}; got {latents.shape}'
            )

        initial_log_prob = working_model.initial.dist.log_prob(latents[..., 0, :])

        valid_transitions = working_data.valid_transitions()
        transition_inputs = working_data.transition_inputs()
        safe_inputs = valid_transitions.apply(transition_inputs)
        safe_previous = valid_transitions.apply(latents[..., :-1, :])
        safe_next = valid_transitions.apply(latents[..., 1:, :])

        dynamics = working_model.dynamics.conditional(safe_inputs)
        dynamics = dynamics.conditional(safe_previous)

        transition_log_probs = dynamics.log_prob(safe_next)
        transition_log_probs = valid_transitions.apply(transition_log_probs)

        dynamics_log_prob = jnp.sum(transition_log_probs, axis=-1)

        emission_log_prob = working_model.emissions.log_likelihood(
            working_data.weighted_observations(),
            latents,
        )

        sequence_log_joint = initial_log_prob + dynamics_log_prob + emission_log_prob

        data_axes = tuple(
            range(
                len(model_batch_shape),
                len(model_batch_shape) + len(data_batch_shape),
            )
        )
        return (
            jnp.sum(sequence_log_joint, axis=data_axes)
            if data_axes
            else sequence_log_joint
        )

    def fit_params(
        self,
        data: Dataset,
        posterior: Posterior,
    ) -> typing.Self:
        """Pool dataset batch axes while preserving the model batch prefix."""
        model_batch_shape = self.batch_shape
        expected_batch = (*model_batch_shape, *data.batch_shape)

        if posterior.batch_shape != expected_batch:
            raise ValueError(
                f'posterior batch must be {expected_batch}; got {posterior.batch_shape}'
            )

        working_data = data
        if model_batch_shape:
            working_data = data.broadcast(model_batch_shape, axis=0)

        return self.__class__(
            initial=self.initial.fit_params(
                posterior,
                weights=working_data.valid().materialize(working_data.num_steps),
            ),
            dynamics=self.dynamics.fit_params(
                posterior,
                inputs=working_data.transition_inputs(),
                weights=working_data.valid_transitions().materialize(
                    working_data.num_steps - 1
                ),
            ),
            emissions=self.emissions.fit_params(
                working_data.weighted_observations(),
                posterior,
            ),
        )

    def align(self, alignment: Affine) -> typing.Self:
        """Express the LDS in aligned continuous latent coordinates."""
        return self._replace(
            initial=self.initial.align(alignment),
            dynamics=self.dynamics.align(alignment),
            emissions=self.emissions.compose_input(alignment),
        )

    def whiten(self) -> typing.Self:
        r"""Express an LDS in latent coordinates where dynamics noise is identity.

        For $Q = LL^\top$, use $\bar x = L^{-1}x$.
        """
        covariance = self.dynamics.dist.covariance
        latent_dim = self.dynamics.dist.output_dim
        batch_shape = self.dynamics.batch_shape

        cholesky = jnp.linalg.cholesky(covariance)

        identity = jnp.eye(
            latent_dim,
            dtype=covariance.dtype,
        )

        whitening = jnp.linalg.solve(
            cholesky,
            identity,
        )

        alignment = Affine(
            coefficients=whitening,
            bias=jnp.zeros(
                (*batch_shape, latent_dim),
                dtype=covariance.dtype,
            ),
        )

        return self.align(alignment)
