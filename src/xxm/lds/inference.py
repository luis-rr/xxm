"""LDS inference via Gaussian message passing and Laplace approximation."""

import jax

from xxm.core.chains.gaussian import GaussianChain as Chain
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.emissions.continuous import (
    LaplaceEmissionsT,
    QuadraticEmissionsT,
)
from xxm.core.inference import Inferred
from xxm.core.optim.laplace import laplace_inference
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model


def to_chain(
    model: Model[QuadraticEmissionsT] | Model[LaplaceEmissionsT],
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
) -> Inferred[
    Model[QuadraticEmissionsT],
    Posterior,
]:
    """Run exact Gaussian forward-backward inference."""

    latent_chain = to_chain(
        model,
        num_steps=observations.shape[0],
    )

    observation_potential = model.emissions.compute_potential(observations)

    posterior_chain = latent_chain.add_local_potential(
        observation_potential,
    )

    posterior, log_normalizer = posterior_chain.forward_backward()

    return Inferred(
        model=model,
        posterior=posterior,
        objective=log_normalizer,
    )


def infer_laplace(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    initial_latents: jax.Array | None = None,
    params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> Inferred[
    Model[LaplaceEmissionsT],
    Posterior,
]:
    """
    Approximate the latent posterior and marginal log likelihood using Laplace
    inference.

    Damped Newton iterations seek the posterior mode. The final local Gaussian
    approximation defines the returned posterior and Laplace approximation to
    the marginal log likelihood.
    """

    num_steps = observations.shape[0]

    chain = to_chain(
        model,
        num_steps=num_steps,
    )

    if initial_latents is None:
        latents = model.compute_prior_means(num_steps)
    else:
        latents = initial_latents

    posterior, log_normalizer = laplace_inference(
        chain=chain,
        emissions=model.emissions,
        observations=observations,
        initial_latents=latents,
        search_params=params,
    )

    return Inferred(
        model=model,
        posterior=posterior,
        objective=log_normalizer,
    )
