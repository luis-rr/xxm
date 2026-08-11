from __future__ import annotations

import jax
from jax import numpy as jnp

from ..gaussian_chain import fit_linear_gaussian_from_moments
from .core import GaussianEmissions, Model, Posterior


def _m_step_initial_probs(
    posterior: Posterior,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Maximum-likelihood update of the initial-state probabilities.
    """
    initial_mean = posterior.means[0]
    initial_covariance = posterior.covariances[0]

    return initial_mean, initial_covariance


def _m_step_dynamics(
    posterior: Posterior,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    means = posterior.means
    second = posterior.raw_second_moments()
    cross = posterior.raw_cross_moments()

    return fit_linear_gaussian_from_moments(
        input_mean=jnp.mean(means[:-1], axis=0),
        output_mean=jnp.mean(means[1:], axis=0),
        input_second_moment=jnp.mean(second[:-1], axis=0),
        output_second_moment=jnp.mean(second[1:], axis=0),
        output_input_moment=jnp.mean(cross, axis=0).T,
    )


def _m_step_emissions(
    observations: jax.Array,
    posterior: Posterior,
) -> GaussianEmissions:
    means = posterior.means
    second = posterior.raw_second_moments()

    num_samples = observations.shape[0]

    readout, bias, noise_covariance = fit_linear_gaussian_from_moments(
        input_mean=jnp.mean(means, axis=0),
        output_mean=jnp.mean(observations, axis=0),
        input_second_moment=jnp.mean(second, axis=0),
        output_second_moment=observations.T @ observations / num_samples,
        output_input_moment=observations.T @ means / num_samples,
    )

    return GaussianEmissions(
        readout=readout,
        bias=bias,
        noise_covariance=noise_covariance,
    )


def em_step(
    model: Model,
    observations: jax.Array,
) -> tuple[Model, Posterior]:

    # e step
    posterior = model.inference(observations)

    # m step
    initial_mean, initial_covariance = _m_step_initial_probs(posterior)
    dynamics_matrix, dynamics_bias, dynamics_covariance = _m_step_dynamics(posterior)
    emissions = _m_step_emissions(observations, posterior)

    new_params = Model(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_matrix=dynamics_matrix,
        dynamics_bias=dynamics_bias,
        dynamics_noise_covariance=dynamics_covariance,
        emissions=emissions,
    )

    return new_params, posterior


def fit_em(
    model: Model,
    observations: jax.Array,
    num_iters: int,
) -> tuple[Model, jax.Array]:

    def step(model, _):
        new_model, posterior = em_step(model, observations)
        return new_model, posterior.log_normalizer

    model, log_likelihoods = jax.lax.scan(
        step,
        model,
        xs=None,
        length=num_iters,
    )

    return model, log_likelihoods


fit_em_jit = jax.jit(fit_em, static_argnames='num_iters')
