import typing

import jax
from jax import numpy as jnp

from xxm.core.chains.gaussian import GaussianChain as Chain
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.emissions.continuous import LaplaceEmissionsT, QuadraticEmissionsT
from xxm.core.optim.newton import NewtonSearch

from .model import Model


def to_chain(
    model: Model,
    num_steps: int,
) -> Chain:
    """Construct the Gaussian chain defined by the latent LDS prior."""
    return Chain.from_pair_potentials(
        model.compute_initial_potential(),
        model.compute_pair_potentials(num_steps),
    )


def infer_exact(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
) -> tuple[Posterior, jax.Array]:
    """Compute the exact posterior over latents for quadratic emissions."""

    latent_chain = to_chain(
        model,
        num_steps=observations.shape[0],
    )

    observation_potential = model.emissions.compute_potential(observations)

    posterior_chain = latent_chain.add_local_potential(
        observation_potential,
    )

    return posterior_chain.forward_backward()


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

    latent_chain: Chain
    emissions: LaplaceEmissionsT
    observations: jax.Array

    def objective(self, params: _NewtonSearchParams) -> jax.Array:
        """Evaluate the log joint at a latent trajectory."""

        return self.latent_chain.log_potential(params.latents) + self.emissions.log_likelihood(
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


def infer_laplace(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    initial_latents: jax.Array | None = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
) -> tuple[Posterior, jax.Array]:
    """
    Approximate the posterior over latents using Laplace inference.

    The posterior mode is found with damped Newton iterations. At each
    iteration, the emission likelihood is replaced by its local quadratic
    approximation, producing a Gaussian chain whose mean gives the Newton
    candidate.

    The final Gaussian chain, evaluated at the converged mode, is returned
    as the Laplace approximation to p(x | y).
    """

    if max_iter < 1:
        raise ValueError('max_iter must be at least 1')

    if tol <= 0.0:
        raise ValueError('tol must be positive')

    if max_line_search_iters < 0:
        raise ValueError('max_line_search_iters must be non-negative')

    num_steps = observations.shape[0]

    if initial_latents is None:
        latents = model.compute_prior_means(num_steps)
    else:
        latents = initial_latents

    newton_model = _NewtonSearchModel(
        latent_chain=to_chain(
            model=model,
            num_steps=num_steps,
        ),
        emissions=model.emissions,
        observations=observations,
    )

    initial_params = _NewtonSearchParams(
        latents=latents,
    )

    search = NewtonSearch[_NewtonSearchParams](
        model=newton_model,
        max_line_search_iters=max_line_search_iters,
        tol=tol,
    )

    final = search.optimize(params=initial_params, max_iter=max_iter)

    latents = final.params.latents

    # Rebuild the approximation at the final mode: the covariance of the
    # Laplace approximation must use the Hessian evaluated there.
    observation_potential = model.emissions.compute_local_potential(
        observations,
        latents,
    )

    posterior_chain = newton_model.latent_chain.add_local_potential(
        observation_potential,
    )

    return posterior_chain.forward_backward()
