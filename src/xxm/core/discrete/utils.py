import itertools

import jax
import jax.numpy as jnp


def match_states(costs: jax.Array) -> jax.Array:
    """Find the permutation minimizing total pairwise state cost."""
    permutation = min(
        itertools.permutations(range(costs.shape[0])),
        key=lambda p: sum(costs[k, p[k]] for k in range(len(p))),
    )
    return jnp.asarray(permutation)


def match_states_by_mean(
    means0: jax.Array,
    means1: jax.Array,
) -> jax.Array:
    """Use pairwise costs to match latent states."""

    costs = jnp.linalg.norm(
        means0[:, None, :] - means1[None, :, :],
        axis=-1,
    )

    return match_states(costs)


def match_states_by_conditional_mean(
    means0: jax.Array,  # (T, K, N)
    means1: jax.Array,  # (T, K, N)
) -> jax.Array:
    """Match states by their conditional means across time."""
    differences = means0[:, :, None, :] - means1[:, None, :, :]  # (T, K, K, N)

    costs = jnp.mean(
        jnp.sum(differences**2, axis=-1),
        axis=0,
    )  # (K, K)

    return match_states(costs)
