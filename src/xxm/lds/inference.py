import jax

from xxm.core.chains.gaussian import GaussianChain as Chain
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.emissions.continuous import (
    LaplaceEmissionsT,
    QuadraticEmissionsT,
)
from xxm.core.optim.laplace import laplace_inference

from .core import Model


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

    num_steps = observations.shape[0]

    chain = to_chain(
        model,
        num_steps=observations.shape[0],
    )

    if initial_latents is None:
        latents = model.compute_prior_means(num_steps)
    else:
        latents = initial_latents

    return laplace_inference(
        chain=chain,
        emissions=model.emissions,
        observations=observations,
        initial_latents=latents,
        max_iter=max_iter,
        tol=tol,
        max_line_search_iters=max_line_search_iters,
    )
