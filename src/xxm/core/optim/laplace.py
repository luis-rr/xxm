import typing

import jax
from jax import numpy as jnp

from xxm.core.chains.gaussian import GaussianChain, GaussianChainMarginals
from xxm.core.emissions.continuous import (
    LaplaceEmissions,
    LaplaceEmissionsT,
)
from xxm.core.optim.newton import NewtonSearch, OptimParams


class _NewtonSearchParams(typing.NamedTuple):
    """Latent trajectory optimized during Laplace inference."""

    latents: jax.Array

    def take_step(self, direction: typing.Self, step_size: jax.Array) -> typing.Self:

        return self.__class__(
            latents=self.latents + step_size * direction.latents,
        )

    def relative_change_from(self, other: typing.Self) -> jax.Array:

        distance = jnp.linalg.norm(self.latents - other.latents)

        return distance / (1.0 + jnp.linalg.norm(other.latents))

    def where(self, mask: jax.Array, other: typing.Self) -> typing.Self:

        return self.__class__(
            latents=jnp.where(
                mask,
                self.latents,
                other.latents,
            ),
        )


class _NewtonSearchModel(
    typing.NamedTuple,
    typing.Generic[LaplaceEmissionsT],
):
    """Find the Laplace posterior mode."""

    latent_chain: GaussianChain
    emissions: LaplaceEmissionsT
    observations: jax.Array

    def objective(self, params: _NewtonSearchParams) -> jax.Array:
        """Evaluate the log joint at a latent trajectory."""

        return self.latent_chain.log_potential(
            params.latents
        ) + self.emissions.log_likelihood(
            self.observations,
            params.latents,
        )

    def newton_direction(self, params: _NewtonSearchParams) -> _NewtonSearchParams:
        """Compute the Newton direction for the latent trajectory."""

        observation_potential = self.emissions.compute_local_potential(
            self.observations,
            params.latents,
        )

        local_posterior_chain = self.latent_chain.add_local_potential(
            observation_potential,
        )

        # The mean is also the mode of this local Gaussian approximation.
        # TODO: Use a solver for the mode that avoids computing the full posterior.
        posterior, _ = local_posterior_chain.forward_backward()
        newton_latents = posterior.means

        return _NewtonSearchParams(
            latents=newton_latents - params.latents,
        )


def laplace_inference(
    chain: GaussianChain,
    emissions: LaplaceEmissions,
    observations: jax.Array,
    initial_latents: jax.Array,
    search_params: OptimParams,
) -> tuple[GaussianChainMarginals, jax.Array]:
    """
    Approximate the posterior over latents using Laplace inference.

    The posterior mode is found with damped Newton iterations. At each
    iteration, the emission likelihood is replaced by its local quadratic
    approximation, producing a Gaussian chain whose mean gives the Newton
    candidate.

    The final Gaussian chain, evaluated at the converged mode, is returned
    as the Laplace approximation to p(x | y).
    """

    latents = initial_latents

    search_params.validate()

    newton_model = _NewtonSearchModel(
        latent_chain=chain,
        emissions=emissions,
        observations=observations,
    )

    initial_params = _NewtonSearchParams(
        latents=latents,
    )

    search = NewtonSearch[_NewtonSearchParams](
        model=newton_model,
        optim_params=search_params,
    )

    final = search.optimize(params=initial_params)

    latents = final.params.latents

    # Rebuild the approximation at the final mode: the covariance of the
    # Laplace approximation must use the Hessian evaluated there.
    observation_potential = emissions.compute_local_potential(
        observations,
        latents,
    )

    posterior_chain = newton_model.latent_chain.add_local_potential(
        observation_potential,
    )

    return posterior_chain.forward_backward()
