r"""Linear Dynamical Systems."""

from __future__ import annotations

import typing

import jax
from jax import numpy as jnp

from xxm.core import _batch
from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.chains.gaussian import GaussianPairPotential, GaussianPotential
from xxm.core.data import Sequences, WeightedObservations
from xxm.core.emissions.continuous import EmissionsT
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics


class _AlignedData(typing.NamedTuple, typing.Generic[EmissionsT]):
    """LDS components and sequence data with explicit model × sequence batching."""

    initial: GaussianInitial
    dynamics: GaussianLinearDynamics
    emissions: EmissionsT
    observations: WeightedObservations
    transition_inputs: jax.Array
    valid: jax.Array
    valid_transitions: jax.Array


def _model_batch_shape(model: Model) -> tuple[int, ...]:
    """Return the common structural batch shape of the LDS components."""
    batch_shape = model.initial.batch_shape

    if model.dynamics.batch_shape != batch_shape:
        raise ValueError('initial and dynamics batch shapes must match')

    if model.emissions.batch_shape != batch_shape:
        raise ValueError('initial and emissions batch shapes must match')

    return batch_shape


def _align_data(
    model: Model[EmissionsT],
    data: Sequences,
) -> _AlignedData[EmissionsT]:
    """Construct the explicit Cartesian product of model batches and sequences."""
    batch_shape = _model_batch_shape(model)

    if data.input_dim() != model.dynamics.input_dim:
        raise ValueError(
            f'expected input dimension {model.dynamics.input_dim}, '
            f'got {data.input_dim()}'
        )

    num_sequences = data.num_sequences()
    sequence_axis = len(batch_shape)

    observations = data.weighted_observations()
    if batch_shape:
        observations = observations.broadcast(batch_shape, axis=0)

    return _AlignedData(
        initial=model.initial.broadcast(
            (num_sequences,),
            axis=sequence_axis,
        ),
        dynamics=model.dynamics.broadcast(
            (num_sequences,),
            axis=sequence_axis,
        ),
        emissions=model.emissions.broadcast(
            (num_sequences,),
            axis=sequence_axis,
        ),
        observations=observations,
        transition_inputs=_batch.broadcast_array(
            data.transition_inputs(), batch_shape, axis=0
        ),
        valid=_batch.broadcast_array(data.valid(), batch_shape, axis=0),
        valid_transitions=_batch.broadcast_array(
            data.valid_transitions(), batch_shape, axis=0
        ),
    )


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
        _model_batch_shape(self)
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
        _model_batch_shape(self)
        return self.dynamics.mean_trajectory(
            self.initial.dist.mean, num_steps, inputs=inputs
        )

    def log_joint(
        self,
        data: Sequences,
        latents: jax.Array,
    ) -> jax.Array:
        r"""
        Compute the joint log density across independent sequences.

        Observation factors are included only where `data.mask` is true.
        Latent initial and transition factors are included throughout each valid
        sequence prefix. Independent sequence log densities are summed.
        """
        aligned = _align_data(self, data)

        expected_latent_shape = (
            *aligned.valid.shape,
            aligned.dynamics.latent_dim,
        )
        if latents.shape != expected_latent_shape:
            raise ValueError(
                f'latents must have shape {expected_latent_shape}; got {latents.shape}'
            )

        initial_log_prob = aligned.initial.dist.log_prob(latents[..., 0, :])

        valid_transitions = aligned.valid_transitions
        safe_inputs = jnp.where(
            valid_transitions[..., None],
            aligned.transition_inputs,
            jnp.zeros((), dtype=aligned.transition_inputs.dtype),
        )
        safe_previous = jnp.where(
            valid_transitions[..., None],
            latents[..., :-1, :],
            jnp.zeros((), dtype=latents.dtype),
        )
        safe_next = jnp.where(
            valid_transitions[..., None],
            latents[..., 1:, :],
            jnp.zeros((), dtype=latents.dtype),
        )

        dynamics = aligned.dynamics.conditional(safe_inputs)
        dynamics = dynamics.conditional(safe_previous)

        transition_log_probs = dynamics.log_prob(safe_next)
        transition_log_probs = jnp.where(
            valid_transitions,
            transition_log_probs,
            jnp.zeros((), dtype=transition_log_probs.dtype),
        )
        dynamics_log_prob = jnp.sum(
            transition_log_probs,
            axis=-1,
        )

        emission_log_prob = aligned.emissions.log_likelihood(
            aligned.observations,
            latents,
        )

        sequence_log_joint = initial_log_prob + dynamics_log_prob + emission_log_prob

        return jnp.sum(sequence_log_joint, axis=-1)

    def fit_params(
        self,
        data: Sequences,
        posterior: Posterior,
    ) -> typing.Self:
        """Fit shared LDS parameters from a multi-sequence posterior."""
        aligned = _align_data(self, data)
        expected_batch = aligned.observations.batch_shape

        if posterior.batch_shape != expected_batch:
            raise ValueError(
                f'posterior batch must be {expected_batch}; got {posterior.batch_shape}'
            )

        return self.__class__(
            initial=self.initial.fit_params(
                posterior,
                weights=aligned.valid,
            ),
            dynamics=self.dynamics.fit_params(
                posterior,
                inputs=aligned.transition_inputs,
                weights=aligned.valid_transitions,
            ),
            emissions=self.emissions.fit_params(
                aligned.observations,
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
