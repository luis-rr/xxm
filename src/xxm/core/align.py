import itertools

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine


def match_states(costs: jax.Array) -> jax.Array:
    """Return the source-state permutation that best matches target states.

    ``costs[target, source]`` is the cost of matching each source state to
    each target state. The returned permutation can be applied to the source
    state axis to express it in the target-state ordering.
    """
    permutation = min(
        itertools.permutations(range(costs.shape[0])),
        key=lambda p: sum(costs[k, p[k]] for k in range(len(p))),
    )
    return jnp.asarray(permutation)


def match_states_by_mean(
    source: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """Return the permutation aligning source states to target states.

    States are matched by Euclidean distance between their associated means.
    The returned permutation should be applied to ``source``.
    """
    costs = jnp.linalg.norm(
        target[:, None, :] - source[None, :, :],
        axis=-1,
    )

    return match_states(costs)


def match_states_by_conditional_mean(
    source: jax.Array,  # (T, K, N)
    target: jax.Array,  # (T, K, N)
) -> jax.Array:
    """Return the permutation aligning source states to target states.

    States are matched by their mean squared difference in conditional means
    across time. The returned permutation should be applied to ``source``.
    """
    differences = (
        target[:, :, None, :] - source[:, None, :, :]
    )  # (T, K_target, K_source, N)

    costs = jnp.mean(
        jnp.sum(differences**2, axis=-1),
        axis=0,
    )  # (K_target, K_source)

    return match_states(costs)


def match_states_to_true(
    state_probs: jax.Array,
    true_states: jax.Array,
) -> jax.Array:
    """Return the permutation aligning inferred states to true state labels.

    ``state_probs`` defines the source-state ordering and ``true_states`` the
    target ordering. The returned permutation should be applied to the
    inferred states or model.
    """
    num_states = state_probs.shape[-1]

    true_state_probs = jax.nn.one_hot(
        true_states,
        num_states,
    )

    costs = -(true_state_probs.T @ state_probs)

    return match_states(costs)


def align_procrustes(
    source: jax.Array,
    target: jax.Array,
) -> Affine:
    """Return a similarity transform aligning source points to target points.

    The fitted transform contains a translation, one global scale, and an
    orthogonal rotation or reflection. Applying it to ``source`` gives its
    least-squares Procrustes alignment with ``target``.
    """
    source_mean = jnp.mean(source, axis=0)
    target_mean = jnp.mean(target, axis=0)

    x = source - source_mean
    y = target - target_mean

    u, singular_values, vt = jnp.linalg.svd(
        x.T @ y,
        full_matrices=False,
    )

    orthogonal = u @ vt
    scale = jnp.sum(singular_values) / jnp.sum(x**2)

    # Affine convention: output = coefficients @ input + bias.
    coefficients = scale * orthogonal.T
    bias = target_mean - coefficients @ source_mean

    return Affine(
        coefficients=coefficients,
        bias=bias,
    )


def align_affine(
    source: jax.Array,
    target: jax.Array,
) -> Affine:
    """Return the least-squares affine transform from source to target.

    Unlike Procrustes alignment, the fitted transform may independently scale,
    shear, rotate, or reflect dimensions. Applying it to ``source`` gives its
    least-squares affine alignment with ``target``.
    """
    source_mean = jnp.mean(source, axis=0)
    target_mean = jnp.mean(target, axis=0)

    x = source - source_mean
    y = target - target_mean

    coefficients = jnp.linalg.lstsq(
        x,
        y,
        rcond=None,
    )[0].T

    bias = target_mean - coefficients @ source_mean

    return Affine(
        coefficients=coefficients,
        bias=bias,
    )
