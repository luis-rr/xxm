"""JAX-native K-means assignments."""

import jax
import jax.numpy as jnp


def kmeans_assignments(
    key: jax.Array,
    observations: jax.Array,
    num_groups: int,
    num_iters: int = 20,
    *,
    sample_weights: jax.Array | None = None,
) -> jax.Array:
    """Compute K-means assignments with optional sample weights.

    Zero-weight observations are excluded from center initialization
    and center updates. Requires at least num_groups positive-weight
    observations when sample_weights is provided.
    """
    observations = jnp.asarray(
        observations,
        dtype=jnp.result_type(observations, jnp.float32),
    )

    if sample_weights is None:
        initial_indices = jax.random.choice(
            key,
            observations.shape[0],
            shape=(num_groups,),
            replace=False,
        )
    else:
        sample_weights = jnp.asarray(
            sample_weights,
            dtype=observations.dtype,
        )

        initial_indices = jax.random.choice(
            key,
            observations.shape[0],
            shape=(num_groups,),
            replace=False,
            p=sample_weights / sample_weights.sum(),
        )

    initial_centers = observations[initial_indices]

    def assign(centers):
        distances = jnp.sum(
            (observations[:, None, :] - centers[None, :, :]) ** 2,
            axis=-1,
        )
        return jnp.argmin(distances, axis=1)

    def step(_, centers):
        assignments = assign(centers)

        weights = jax.nn.one_hot(
            assignments,
            num_groups,
            dtype=observations.dtype,
        )

        if sample_weights is not None:
            weights = weights * sample_weights[:, None]

        counts = weights.sum(axis=0)

        new_centers = (
            weights.T
            @ observations
            / jnp.where(counts[:, None] > 0, counts[:, None], 1)
        )

        return jnp.where(
            (counts > 0)[:, None],
            new_centers,
            centers,
        )

    centers = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        initial_centers,
    )

    return assign(centers)
