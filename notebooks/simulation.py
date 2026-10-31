"""Synthetic ground-truth setup shared by the example notebooks."""

import jax
import jax.numpy as jnp


def random_emission_embedding(
    key: jax.Array,
    *,
    latent_dim: int,
    observation_dim: int,
    observation_std: float,
    emission_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Random well-conditioned linear-Gaussian emission embedding."""

    if observation_dim < latent_dim:
        raise ValueError('observation_dim must be at least latent_dim')

    raw = jax.random.normal(
        key,
        (observation_dim, latent_dim),
    )

    # Random orthonormal basis for a latent_dim-dimensional
    # subspace of observation space.
    coefficients, _ = jnp.linalg.qr(
        raw,
        mode='reduced',
    )

    coefficients = emission_scale * coefficients

    bias = jnp.zeros(observation_dim)

    covariance = observation_std**2 * jnp.eye(observation_dim)

    return coefficients, bias, covariance
