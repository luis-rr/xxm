"""
Structured mean-field inference for Switching Linear Dynamical Systems.
"""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.chains.discrete import DiscreteChain, DiscretePotential
from xxm.core.chains.discrete import DiscreteChainMarginals as DiscretePosterior
from xxm.core.chains.gaussian import (
    GaussianChain,
    GaussianPairPotential,
    GaussianPotential,
)
from xxm.core.chains.gaussian import GaussianChainMarginals as ContinuousPosterior
from xxm.core.data import Dataset, DatasetCollection, WeightedObservations
from xxm.core.emissions.continuous import (
    EmissionsT,
    LaplaceEmissionsT,
    QuadraticEmissionsT,
)
from xxm.core.inference import InferenceState
from xxm.core.latents.discrete import homogeneous_chain
from xxm.core.lengths import Lengths
from xxm.core.optim.laplace import laplace_inference, local_gaussian_approximation
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model, Posterior


def _continuous_chain(
    initial: GaussianPotential,
    dynamics: GaussianPairPotential,
    lengths: Lengths,
) -> GaussianChain:
    """Build a chain with independent unit-precision padding, as in LDS inference.

    Dynamics already exclude invalid transitions. Padding stabilizes numerical
    factorization; validity excludes it from normalizers and entropy.
    """
    chain = GaussianChain.from_pair_potentials(initial, dynamics, lengths)
    shape = (*chain.batch_shape, chain.max_steps)
    dim = chain.variable_dim
    dtype = chain.diagonal_precision_blocks.dtype
    stabilizer = GaussianPotential(
        precision_blocks=jnp.broadcast_to(
            jnp.eye(dim, dtype=dtype), (*shape, dim, dim)
        ),
        information_vectors=jnp.zeros((*shape, dim), dtype=dtype),
        log_constant=jnp.zeros(shape, dtype=dtype),
    ).scale(~lengths.to_mask(chain.max_steps).values)
    return chain.add_local_potential(stabilizer)


class QuadraticContinuousFactors(typing.NamedTuple):
    """Quadratic observation factors and structural sequence validity."""

    observations: GaussianPotential
    lengths: Lengths

    @classmethod
    def from_model(
        cls, model: Model[QuadraticEmissionsT], data: Dataset
    ) -> typing.Self:
        return cls(
            model.emissions.single().compute_potential(data.weighted_observations()),
            data.lengths,
        )

    def infer(
        self,
        initial_potential: GaussianPotential,
        dynamics_potential: GaussianPairPotential,
    ) -> tuple[ContinuousPosterior, jax.Array]:
        chain = _continuous_chain(initial_potential, dynamics_potential, self.lengths)
        return chain.add_local_potential(self.observations).forward_backward()


class LaplaceContinuousFactors(typing.NamedTuple, typing.Generic[LaplaceEmissionsT]):
    """Nonconjugate observations with separate visibility and structural validity."""

    emissions: LaplaceEmissionsT
    observations: WeightedObservations
    search_params: OptimParams

    @classmethod
    def from_model(
        cls,
        model: Model[LaplaceEmissionsT],
        data: Dataset,
        search_params: OptimParams,
    ) -> typing.Self:
        return cls(
            model.emissions.single(), data.weighted_observations(), search_params
        )

    def approximate(
        self,
        initial_potential: GaussianPotential,
        dynamics_potential: GaussianPairPotential,
        latents: jax.Array,
    ) -> tuple[ContinuousPosterior, jax.Array]:
        chain = _continuous_chain(
            initial_potential, dynamics_potential, self.observations.lengths
        )
        return local_gaussian_approximation(
            chain=chain,
            emissions=self.emissions,
            observations=self.observations,
            latents=self.observations.lengths.clear(latents),
        )

    def infer(
        self,
        initial_potential: GaussianPotential,
        dynamics_potential: GaussianPairPotential,
        initial_latents: jax.Array,
    ) -> tuple[ContinuousPosterior, jax.Array]:
        chain = _continuous_chain(
            initial_potential, dynamics_potential, self.observations.lengths
        )
        return laplace_inference(
            chain=chain,
            emissions=self.emissions,
            observations=self.observations,
            initial_latents=self.observations.lengths.clear(initial_latents),
            search_params=self.search_params,
        )


class DiscreteFactors(typing.NamedTuple):
    """Markov-chain factors and structural sequence validity."""

    chain: DiscreteChain

    @classmethod
    def from_model(cls, model: Model[EmissionsT], data: Dataset) -> typing.Self:
        return cls(
            homogeneous_chain(
                model.state_initial,
                model.transitions,
                data.lengths_spec,
            ),
        )

    def infer(self, discrete_potential: DiscretePotential) -> DiscretePosterior:
        chain = self.chain.add_local_potential(discrete_potential)
        posterior, _ = chain.forward_backward()
        return posterior


class SwitchingFactors(typing.NamedTuple):
    """State-dependent factors with batches `(*B, K)` and `(*B, T - 1, K)`."""

    initial_potential: GaussianPotential
    dynamics_potential: GaussianPairPotential
    lengths: Lengths

    @classmethod
    def from_model(cls, model: Model[EmissionsT], data: Dataset) -> typing.Self:
        transitions = data.lengths.transitions()
        safe_inputs = transitions.clear(data.transition_inputs())
        dynamics = model.dynamics.compute_pair_potentials(safe_inputs)
        # The mask owns time, while state and Gaussian coordinates are event axes.
        dynamics = jax.tree.map(transitions.clear, dynamics)
        return cls(model.latent_initial.compute_potentials(), dynamics, data.lengths)

    def continuous_potentials(
        self,
        discrete_posterior: DiscretePosterior,
    ) -> tuple[GaussianPotential, GaussianPairPotential]:
        """Average state-specific Gaussian factors under q(z)."""
        states = self.lengths.clear(discrete_posterior.state_probs)
        initial = self.initial_potential.weighted_sum(states[..., 0, :], axis=-1)
        weights = self.lengths.transitions().clear(states[..., 1:, :])
        dynamics = self.dynamics_potential.weighted_sum(weights, axis=-1)
        return initial, dynamics

    def discrete_potential(
        self, continuous_posterior: ContinuousPosterior
    ) -> DiscretePotential:
        """Expected switching log factors, zero outside valid sequence prefixes."""
        means = self.lengths.clear(continuous_posterior.means)
        covariances = self.lengths.clear(continuous_posterior.covariances)
        num_states = self.initial_potential.batch_shape[-1]
        state_axis = len(continuous_posterior.batch_shape)
        initial_mean = batch.broadcast_array(
            means[..., 0, :], (num_states,), state_axis
        )
        initial_covariance = batch.broadcast_array(
            covariances[..., 0, :, :], (num_states,), state_axis
        )
        initial_second = (
            initial_covariance + initial_mean[..., :, None] * initial_mean[..., None, :]
        )
        initial_values = self.initial_potential.expected_log_potential(
            initial_mean, initial_second
        )

        transitions = self.lengths.transitions()
        paired = jax.tree.map(
            transitions.clear, continuous_posterior.paired_marginals()
        )
        paired = paired.broadcast((num_states,), axis=-1)
        dynamics_values = self.dynamics_potential.expected_log_potential(paired)
        dynamics_values = transitions.clear(dynamics_values)
        return DiscretePotential(
            self.lengths.clear(
                jnp.concatenate(
                    [initial_values[..., None, :], dynamics_values], axis=-2
                )
            )
        )

    def discrete_potential_from_latents(self, latents: jax.Array) -> DiscretePotential:
        """Score deterministic latents using the same factors as the moment update."""
        latents = self.lengths.clear(latents)
        shape, dim = latents.shape[:-1], latents.shape[-1]
        posterior = ContinuousPosterior(
            lengths=self.lengths,
            means=latents,
            covariances=jnp.zeros((*shape, dim, dim), dtype=latents.dtype),
            cross_covariances=jnp.zeros(
                (*shape[:-1], shape[-1] - 1, dim, dim), dtype=latents.dtype
            ),
        )
        return self.discrete_potential(posterior)


class QuadraticVI(typing.NamedTuple):
    """Structured mean-field factors for conjugate SLDS inference."""

    switching: SwitchingFactors
    continuous: QuadraticContinuousFactors
    discrete: DiscreteFactors

    @classmethod
    def from_model(
        cls, model: Model[QuadraticEmissionsT], data: Dataset
    ) -> typing.Self:
        return cls(
            SwitchingFactors.from_model(model, data),
            QuadraticContinuousFactors.from_model(model, data),
            DiscreteFactors.from_model(model, data),
        )

    def infer_continuous(
        self, discrete_posterior: DiscretePosterior
    ) -> tuple[ContinuousPosterior, jax.Array]:
        return self.continuous.infer(
            *self.switching.continuous_potentials(discrete_posterior)
        )

    def infer_discrete(
        self, continuous_posterior: ContinuousPosterior
    ) -> DiscretePosterior:
        return self.discrete.infer(
            self.switching.discrete_potential(continuous_posterior)
        )


def _quadratic_elbo(
    discrete_posterior: DiscretePosterior,
    continuous_log_normalizer: jax.Array,
    discrete_prior: DiscreteChain,
) -> jax.Array:
    """Per-sequence ELBO after the optimal conjugate continuous update."""
    return (
        continuous_log_normalizer
        + discrete_posterior.expected_log_potential(discrete_prior)
        + discrete_posterior.entropy()
    )


def _align_inference_inputs(
    model: Model[EmissionsT],
    data: Dataset,
    initial_latents: jax.Array,
) -> tuple[Model[EmissionsT], Dataset, jax.Array]:
    """Form the Cartesian model/data batch and validate the supplied trajectory."""
    if data.input_dim != model.dynamics.input_dim:
        raise ValueError('dataset input dimension must match dynamics')
    working_model, working_data = batch.cartesian_broadcast(model, data)
    expected = (*working_data.batch_shape, data.max_steps, model.dynamics.latent_dim)
    if initial_latents.shape != expected:
        raise ValueError(
            f'initial_latents must have shape {expected}; got {initial_latents.shape}'
        )
    return working_model, working_data, working_data.lengths.clear(initial_latents)


def _sum_data_objectives(values: jax.Array, model: Model, data: Dataset) -> jax.Array:
    """Sum independent-sequence objectives, preserving model batch axes."""
    axes = tuple(
        range(len(model.batch_shape), len(model.batch_shape) + len(data.batch_shape))
    )
    return jnp.sum(values, axis=axes) if axes else values


def infer_variational(
    model: Model[QuadraticEmissionsT],
    data: DatasetCollection,
    num_iters: int,
    initial_latents: jax.Array,
) -> InferenceState[Model[QuadraticEmissionsT], Posterior]:
    """Infer independent trajectories with conjugate updates for q(x).

    Initial latents have Cartesian shape `(*B_model, *B_data, T, D_x)`.
    The facade provides a visibility-respecting completion/inversion default.
    """
    dataset = data.single()
    working_model, working_data, initial_latents = _align_inference_inputs(
        model, dataset, initial_latents
    )
    inference = QuadraticVI.from_model(working_model, working_data)

    def step(_, discrete_posterior):
        continuous_posterior, _ = inference.infer_continuous(discrete_posterior)
        return inference.infer_discrete(continuous_posterior)

    discrete = inference.discrete.infer(
        inference.switching.discrete_potential_from_latents(initial_latents)
    )
    discrete = jax.lax.fori_loop(0, num_iters, step, discrete)
    continuous, log_normalizer = inference.infer_continuous(discrete)
    objective = _quadratic_elbo(discrete, log_normalizer, inference.discrete.chain)
    return InferenceState(
        model=model,
        posterior=Posterior(discrete, continuous),
        objective=_sum_data_objectives(objective, model, dataset),
    )


class LaplaceState(typing.NamedTuple):
    """Iterative state for Laplace structured mean-field inference."""

    discrete: DiscretePosterior
    continuous: ContinuousPosterior


class LaplaceVI(typing.NamedTuple, typing.Generic[LaplaceEmissionsT]):
    """Structured mean-field factors with local Laplace updates for q(x)."""

    switching: SwitchingFactors
    continuous: LaplaceContinuousFactors[LaplaceEmissionsT]
    discrete: DiscreteFactors

    @classmethod
    def from_model(
        cls,
        model: Model[LaplaceEmissionsT],
        data: Dataset,
        search_params: OptimParams,
    ) -> typing.Self:
        return cls(
            SwitchingFactors.from_model(model, data),
            LaplaceContinuousFactors.from_model(model, data, search_params),
            DiscreteFactors.from_model(model, data),
        )

    def approximate_continuous(
        self,
        discrete_posterior: DiscretePosterior,
        latents: jax.Array,
    ) -> ContinuousPosterior:
        initial, dynamics = self.switching.continuous_potentials(discrete_posterior)
        posterior, _ = self.continuous.approximate(initial, dynamics, latents)
        return posterior

    def infer_continuous(
        self,
        discrete_posterior: DiscretePosterior,
        initial_latents: jax.Array,
    ) -> ContinuousPosterior:
        initial, dynamics = self.switching.continuous_potentials(discrete_posterior)
        posterior, _ = self.continuous.infer(initial, dynamics, initial_latents)
        return posterior

    def infer_discrete(
        self, continuous_posterior: ContinuousPosterior
    ) -> DiscretePosterior:
        return self.discrete.infer(
            self.switching.discrete_potential(continuous_posterior)
        )

    def initial_state(self, initial_latents: jax.Array) -> LaplaceState:
        discrete = self.discrete.infer(
            self.switching.discrete_potential_from_latents(initial_latents)
        )
        return LaplaceState(
            discrete, self.approximate_continuous(discrete, initial_latents)
        )


def _laplace_elbo(
    model: Model[LaplaceEmissionsT],
    data: Dataset,
    continuous_posterior: ContinuousPosterior,
    discrete_posterior: DiscretePosterior,
    switching_factors: SwitchingFactors,
    discrete_prior: DiscreteChain,
) -> jax.Array:
    """Per-sequence ELBO, excluding structural padding from every contribution."""
    latent_potential = switching_factors.discrete_potential(continuous_posterior)
    states = data.lengths.clear(discrete_posterior.state_probs)
    log_values = jnp.where(states != 0, latent_potential.log_values, 0)
    expected_latent = jnp.sum(states * log_values, axis=(-2, -1))
    expected_observation = model.emissions.single().expected_log_likelihood(
        data.weighted_observations(),
        continuous_posterior,
    )
    return (
        expected_observation
        + expected_latent
        + discrete_posterior.expected_log_potential(discrete_prior)
        + continuous_posterior.entropy()
        + discrete_posterior.entropy()
    )


def infer_laplace(
    model: Model[LaplaceEmissionsT],
    data: DatasetCollection,
    num_iters: int,
    initial_latents: jax.Array,
    params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> InferenceState[Model[LaplaceEmissionsT], Posterior]:
    """Infer independent trajectories by alternating discrete and Laplace updates."""
    dataset = data.single()
    working_model, working_data, initial_latents = _align_inference_inputs(
        model, dataset, initial_latents
    )
    inference = LaplaceVI.from_model(working_model, working_data, params)

    def step(_, state):
        discrete = inference.infer_discrete(state.continuous)
        continuous = inference.infer_continuous(discrete, state.continuous.means)
        return LaplaceState(discrete, continuous)

    state = jax.lax.fori_loop(
        0, num_iters, step, inference.initial_state(initial_latents)
    )
    objective = _laplace_elbo(
        working_model,
        working_data,
        state.continuous,
        state.discrete,
        inference.switching,
        inference.discrete.chain,
    )
    return InferenceState(
        model=model,
        posterior=Posterior(state.discrete, state.continuous),
        objective=_sum_data_objectives(objective, model, dataset),
    )
