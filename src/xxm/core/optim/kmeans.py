"""JAX-native K-means assignments."""

import jax
import jax.numpy as jnp


def kmeans_assignments(
    key: jax.Array,
    observations: jax.Array,
    num_groups: int,
    num_iters: int = 20,
) -> jax.Array:
    """Compute K-means cluster assignments."""
    observations = jnp.asarray(
        observations,
        dtype=jnp.result_type(observations, jnp.float32),
    )

    initial_indices = jax.random.choice(
        key,
        observations.shape[0],
        shape=(num_groups,),
        replace=False,
    )
    initial_centers = observations[initial_indices]

    def step(_, centers):
        distances = jnp.sum(
            (observations[:, None, :] - centers[None, :, :]) ** 2,
            axis=-1,
        )
        assignments = jnp.argmin(distances, axis=1)

        weights = jax.nn.one_hot(assignments, num_groups, dtype=observations.dtype)
        counts = weights.sum(axis=0)

        new_centers = weights.T @ observations / jnp.maximum(counts[:, None], 1)

        # Keep the old center if a cluster is empty.
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

    distances = jnp.sum(
        (observations[:, None, :] - centers[None, :, :]) ** 2,
        axis=-1,
    )
    return jnp.argmin(distances, axis=1)
