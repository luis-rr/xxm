"""
Structured mean-field inference for Switching Linear Dynamical Systems.
"""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.data import DatasetCollection, WeightedObservations
from xxm.core.dists.discrete import DiscreteChainMarginals
from xxm.core.dists.gaussian import GaussianChainMarginals
from xxm.core.emissions.collection import ContinuousEmissionsCollection
from xxm.core.emissions.continuous import (
    EmissionsT,
    LaplaceEmissionsT,
    QuadraticEmissionsT,
)
from xxm.core.inference import InferenceState
from xxm.core.inference.discrete import infer as discrete_infer
from xxm.core.inference.gaussian import infer as gaussian_infer
from xxm.core.inference.laplace import laplace_inference, local_gaussian_approximation
from xxm.core.latents.discrete import compute_chain_potentials
from xxm.core.lengths import Lengths
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams
from xxm.core.potentials.discrete import DiscreteChainPotentials, DiscretePotential
from xxm.core.potentials.gaussian import GaussianPotential
from xxm.core.potentials.switching import SwitchingGaussianPotentials

from .core import Model, Posterior


class LatentPotentials(typing.NamedTuple):
    """
    Latent potentials for structured mean-field inference.
    """

    cond: SwitchingGaussianPotentials
    prior: DiscreteChainPotentials

    @classmethod
    def from_data(
        cls,
        model: Model[EmissionsT],
        data: DatasetCollection,
    ) -> typing.Self:
        initial, inputs = batch.cartesian_broadcast(
            model.latent_initial,
            data.pool_inputs(),
        )

        conditioned = SwitchingGaussianPotentials(
            initial=initial.compute_potential(),
            pairs=model.dynamics.compute_pair_potentials(inputs),
            lengths=inputs.lengths,
        )

        return cls(
            cond=conditioned,
            prior=compute_chain_potentials(
                model.state_initial,
                model.transitions,
                data.pool_lengths_spec(),
            ),
        )

    @property
    def lengths(self) -> Lengths:
        return self.cond.lengths

    def _discrete_posterior(
        self,
        local: DiscretePotential,
    ) -> DiscreteChainMarginals:
        """Compute the discrete posterior given a local potential."""
        inferred = discrete_infer(self.prior, local)
        return inferred.posterior

    def discrete_posterior(
        self,
        continuous: GaussianChainMarginals,
    ) -> DiscreteChainMarginals:
        """Compute the discrete posterior given a continuous posterior."""
        local = self.cond.compute_discrete_potential(continuous)
        return self._discrete_posterior(local)

    def discrete_posterior_at(
        self,
        latents: jax.Array,
    ) -> DiscreteChainMarginals:
        """Compute the discrete posterior given a deterministic latent trajectory."""
        local = self.cond.compute_discrete_potential_at(latents)
        return self._discrete_posterior(local)

    def continuous_posterior_quadratic(
        self,
        observation: GaussianPotential,
        discrete: DiscreteChainMarginals,
    ) -> GaussianChainMarginals:
        """Compute the continuous posterior given a discrete posterior and quadratic observation potential."""

        potentials = self.cond.compute_continuous_potentials(discrete)
        inferred = gaussian_infer(potentials, observation)
        return inferred.posterior

    def continuous_posterior_local(
        self,
        discrete: DiscreteChainMarginals,
        emissions: ContinuousEmissionsCollection[LaplaceEmissionsT],
        observations: tuple[WeightedObservations, ...],
        latents: jax.Array,
    ) -> GaussianChainMarginals:
        """Compute the continuous posterior given a discrete posterior and local Laplace observation potentials."""

        inferred = local_gaussian_approximation(
            potentials=self.cond.compute_continuous_potentials(discrete),
            emissions=emissions,
            observations=observations,
            latents=latents,
        )

        return inferred.posterior

    def continuous_posterior_laplace(
        self,
        discrete: DiscreteChainMarginals,
        emissions: ContinuousEmissionsCollection[LaplaceEmissionsT],
        observations: tuple[WeightedObservations, ...],
        latents: jax.Array,
        params: OptimParams,
    ) -> GaussianChainMarginals:
        """Compute the continuous posterior given a discrete posterior and Laplace observation potentials."""

        inferred = laplace_inference(
            potentials=self.cond.compute_continuous_potentials(discrete),
            emissions=emissions,
            observations=observations,
            initial_latents=latents,
            search_params=params,
        )

        return inferred.posterior


def infer_variational(
    model: Model[QuadraticEmissionsT],
    data: DatasetCollection,
    num_iters: int,
    initial_latents: jax.Array,
) -> InferenceState[Model[QuadraticEmissionsT], Posterior]:

    potentials = LatentPotentials.from_data(model, data)

    observation = model.emissions.compute_potential(data.weighted_observations())

    initial_latents = potentials.lengths.clear(initial_latents)

    discrete = potentials.discrete_posterior_at(initial_latents)

    def step(_, discrete: DiscreteChainMarginals) -> DiscreteChainMarginals:
        continuous = potentials.continuous_posterior_quadratic(observation, discrete)
        discrete = potentials.discrete_posterior(continuous)
        return discrete

    discrete = jax.lax.fori_loop(0, num_iters, step, discrete)

    continuous = potentials.continuous_posterior_quadratic(observation, discrete)
    posterior = Posterior(discrete, continuous)

    expected_observations = jnp.sum(
        observation.expected_log_potential(
            continuous.means,
            continuous.raw_second_moments(),
        ),
        axis=-1,
    )

    return InferenceState(
        model=model,
        posterior=posterior,
        objective=_elbo_from_potentials(
            potentials,
            posterior,
            expected_observations,
        ),
    )


def infer_laplace(
    model: Model[LaplaceEmissionsT],
    data: DatasetCollection,
    num_iters: int,
    initial_latents: jax.Array,
    params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> InferenceState[Model[LaplaceEmissionsT], Posterior]:
    """Structured mean-field inference with Laplace continuous updates."""

    potentials = LatentPotentials.from_data(model, data)
    observations = data.weighted_observations()

    initial_latents = potentials.lengths.clear(initial_latents)

    discrete = potentials.discrete_posterior_at(initial_latents)

    continuous = potentials.continuous_posterior_local(
        discrete,
        model.emissions,
        observations,
        initial_latents,
    )

    def step(_, posterior: Posterior) -> Posterior:
        discrete = potentials.discrete_posterior(posterior.continuous)

        continuous = potentials.continuous_posterior_laplace(
            discrete,
            model.emissions,
            observations,
            posterior.continuous.means,
            params,
        )

        return Posterior(discrete, continuous)

    posterior = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        Posterior(discrete, continuous),
    )

    expected_observations = model.emissions.expected_log_likelihood(
        observations,
        posterior.continuous,
    )

    return InferenceState(
        model=model,
        posterior=posterior,
        objective=_elbo_from_potentials(
            potentials,
            posterior,
            expected_observations,
        ),
    )


def _elbo_from_potentials(
    potentials: LatentPotentials,
    posterior: Posterior,
    expected_observations: jax.Array,
) -> jax.Array:

    discrete = potentials.cond.compute_discrete_potential(posterior.continuous)

    terms = (
        potentials.prior.expected_log_potential(posterior.discrete),
        discrete.expected_log_potential(posterior.discrete),
        expected_observations,
        posterior.entropy(),
    )

    return jnp.sum(sum(terms), axis=-1)


def elbo(
    model: Model[EmissionsT],
    data: DatasetCollection,
    posterior: Posterior,
) -> jax.Array:
    """Evaluate the generative-model ELBO, preserving model batches."""

    potentials = LatentPotentials.from_data(model, data)

    expected_observations = model.emissions.expected_log_likelihood(
        data.weighted_observations(),
        posterior.continuous,
    )

    return _elbo_from_potentials(
        potentials,
        posterior,
        expected_observations,
    )
