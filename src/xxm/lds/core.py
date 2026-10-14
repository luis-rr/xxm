r"""Linear Dynamical Systems."""

from __future__ import annotations

import typing

import jax
from jax import numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.emissions.collection import ContinuousEmissionsCollection
from xxm.core.emissions.continuous import EmissionsT
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics


class Sample(typing.NamedTuple):
    latents: Sequences
    observations: Sequences


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
    emissions: ContinuousEmissionsCollection[EmissionsT]

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Common structural batch of all top-level components."""
        batch_shape = self.initial.batch_shape
        batch.require_same_shape(
            batch_shape,
            self.dynamics.batch_shape,
            self.emissions.batch_shape,
        )
        if self.initial.dist.variable_dim != self.dynamics.latent_dim or any(
            emission.latent_dim != self.dynamics.latent_dim
            for emission in self.emissions.emissions
        ):
            raise ValueError('LDS components must share their latent dimension')
        return batch_shape

    def select(self, index: batch.SelT) -> typing.Self:
        """Select independent models along their batch dimensions."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated model batch dimensions."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton model batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder independent models along one batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one model batch axis to another batch position."""
        return batch.move_axis(self, source, destination)

    def sample(
        self,
        key: jax.Array,
        *,
        inputs: Sequences,
        session: int | None = None,
    ) -> Sample:
        """Sample latent and observation trajectories."""

        model, inputs = batch.cartesian_broadcast(self, inputs)

        key_initial, key_latents, key_observations = jax.random.split(
            key,
            3,
        )

        initial_latent = model.initial.sample(
            key_initial,
        )

        latents = model.dynamics.sample(
            key_latents,
            initial_latent,
            inputs=inputs,
        )

        observations = model.emissions.get(session).sample(
            key_observations,
            latents,
        )

        return Sample(
            latents=latents,
            observations=observations,
        )

    def compute_prior_means(
        self,
        *,
        inputs: Sequences,
    ) -> Sequences:
        r"""Compute prior latent means $\mathbb{E}[x_t]$."""

        model, inputs = batch.cartesian_broadcast(self, inputs)

        return model.dynamics.mean_trajectory(
            model.initial.dist.mean,
            inputs=inputs,
        )

    def log_joint(
        self,
        data: DatasetCollection,
        latents: jax.Array,
        *,
        session: int | None = None,
    ) -> jax.Array:
        r"""
        Compute the joint log density for one selected session.

        Select the dataset and emission with `session`; omission requires a
        singleton collection. Observation factors use the selected dataset's
        visibility mask.
        Latent initial and transition factors are included throughout each valid
        sequence prefix. Independent sequence log densities are summed.
        """
        dataset = data.get(session)
        model_batch_shape = self.batch_shape
        data_batch_shape = dataset.batch_shape
        working_model, working_data = batch.cartesian_broadcast(self, dataset)

        expected_latent_shape = (
            *working_data.batch_shape,
            working_data.max_steps,
            working_model.dynamics.latent_dim,
        )
        if latents.shape != expected_latent_shape:
            raise ValueError(
                f'latents must have shape {expected_latent_shape}; got {latents.shape}'
            )

        initial_log_prob = working_model.initial.dist.log_prob(latents[..., 0, :])

        valid_transitions = working_data.lengths.transitions()
        transition_inputs = working_data.transition_inputs()
        safe_inputs = valid_transitions.clear(transition_inputs)
        safe_previous = valid_transitions.clear(latents[..., :-1, :])
        safe_next = valid_transitions.clear(latents[..., 1:, :])

        dynamics = working_model.dynamics.conditional(safe_inputs)
        dynamics = dynamics.conditional(safe_previous)

        transition_log_probs = dynamics.log_prob(safe_next)
        transition_log_probs = valid_transitions.clear(transition_log_probs)

        dynamics_log_prob = jnp.sum(transition_log_probs, axis=-1)

        emission_log_prob = working_model.emissions.get(session).log_likelihood(
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
        posterior: Posterior,
        data: DatasetCollection,
    ) -> typing.Self:
        """Fit shared latent parameters and session-specific emissions."""

        inputs = data.pool_inputs()
        observations = data.weighted_observations()

        return self.__class__(
            initial=self.initial.fit_params(posterior),
            dynamics=self.dynamics.fit_params(posterior, inputs),
            emissions=self.emissions.fit_params(posterior, observations),
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
