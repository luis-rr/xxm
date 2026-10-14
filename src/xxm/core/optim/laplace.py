"""Laplace approximation and local quadratic approximation utilities."""

import typing

import jax
from jax import numpy as jnp

from xxm.core.chains.gaussian import (
    GaussianChain,
    GaussianChainMarginals,
    GaussianPotential,
    InferredPosterior,
)
from xxm.core.optim.newton import NewtonSearch, OptimParams

ObservationsT = typing.TypeVar('ObservationsT')
ObservationsT_contra = typing.TypeVar(
    'ObservationsT_contra',
    contravariant=True,
)


class LaplaceLikelihood(typing.Protocol[ObservationsT_contra]):
    def log_likelihood(
        self,
        observations: ObservationsT_contra,
        latents: jax.Array,
    ) -> jax.Array: ...

    def compute_local_potential(
        self,
        observations: ObservationsT_contra,
        latents: jax.Array,
    ) -> GaussianPotential: ...


class _NewtonSearchParams(typing.NamedTuple):
    """Latent trajectory optimized during Laplace inference."""

    latents: jax.Array

    def take_step(self, direction: typing.Self, step_size: jax.Array) -> typing.Self:
        return self.__class__(
            latents=(self.latents + step_size[..., None, None] * direction.latents),
        )

    def relative_change_from(self, other: typing.Self) -> jax.Array:
        distance = jnp.linalg.norm(
            self.latents - other.latents,
            axis=(-2, -1),
        )

        scale = 1.0 + jnp.linalg.norm(
            other.latents,
            axis=(-2, -1),
        )
        return distance / scale

    def where(self, mask: jax.Array, other: typing.Self) -> typing.Self:

        return self.__class__(
            latents=jnp.where(
                mask[..., None, None],
                self.latents,
                other.latents,
            ),
        )


class _NewtonSearchModel(
    typing.NamedTuple,
    typing.Generic[ObservationsT],
):
    """Log-potential objective for finding a latent mode by Newton search."""

    latent_chain: GaussianChain
    emissions: LaplaceLikelihood[ObservationsT]
    observations: ObservationsT

    def _safe_latents(
        self,
        latents: jax.Array,
    ) -> jax.Array:

        return self.latent_chain.lengths.clear(latents)

    def objective(self, params: _NewtonSearchParams) -> jax.Array:
        """Evaluate the chain log potential plus the observation log likelihood."""

        latents = self._safe_latents(params.latents)

        latents_potential = self.latent_chain.log_potential(latents)
        emissions_potential = self.emissions.log_likelihood(self.observations, latents)

        return latents_potential + emissions_potential

    def newton_direction(self, params: _NewtonSearchParams) -> _NewtonSearchParams:
        """Compute the Newton direction for the latent trajectory."""

        latents = self._safe_latents(params.latents)

        observation_potential = self.emissions.compute_local_potential(
            self.observations,
            latents,
        )

        local_posterior_chain = self.latent_chain.add_local_potential(
            observation_potential,
        )

        # The mean is also the mode of this local Gaussian approximation.
        # TODO: Use a solver for the mode that avoids computing the full posterior.
        posterior, _ = local_posterior_chain.forward_backward()
        newton_latents = posterior.means

        return _NewtonSearchParams(
            latents=newton_latents - latents,
        )


def local_gaussian_approximation(
    chain: GaussianChain,
    emissions: LaplaceLikelihood[ObservationsT],
    observations: ObservationsT,
    latents: jax.Array,
) -> InferredPosterior[GaussianChainMarginals]:
    """Construct a local Gaussian posterior approximation around `latents`."""

    observation_potential = emissions.compute_local_potential(
        observations,
        latents,
    )

    posterior_chain = chain.add_local_potential(observation_potential)

    return posterior_chain.forward_backward()


def laplace_inference(
    chain: GaussianChain,
    emissions: LaplaceLikelihood[ObservationsT],
    observations: ObservationsT,
    initial_latents: jax.Array,
    search_params: OptimParams,
) -> InferredPosterior[GaussianChainMarginals]:
    """
    Approximate normalized latent marginals and the log normalizer by Laplace inference.

    Damped Newton iterations seek the latent mode. At each
    iteration, the emission log likelihood is replaced by its local quadratic
    approximation, producing a Gaussian chain whose mean gives the Newton
    candidate.

    The final local Gaussian approximation defines the returned marginals
    and approximate log normalizer. The latter approximates the marginal
    log likelihood when `chain` represents a normalized latent prior;
    expected switching factors in SLDS inference need not satisfy this condition.
    """

    search_params.validate()

    newton_model = _NewtonSearchModel(
        latent_chain=chain,
        emissions=emissions,
        observations=observations,
    )

    initial_params = _NewtonSearchParams(
        latents=initial_latents,
    )

    search = NewtonSearch[_NewtonSearchParams](
        model=newton_model,
        optim_params=search_params,
    )

    final = search.optimize(params=initial_params)

    return local_gaussian_approximation(
        chain=chain,
        emissions=emissions,
        observations=observations,
        latents=final.params.latents,
    )
