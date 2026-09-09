"""State alignment and coordinate alignment utilities."""

import itertools

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian
from xxm.core.optim import gaussian as _gaussian_fit


def match_states(costs: jax.Array) -> jax.Array:
    r"""Permutation minimizing total cost matrix.

    `costs[target, source]` is cost of matching source state to target state.
    Returns permutation to apply to source to match target ordering.
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
    """Permutation aligning source states to target by Euclidean distance of means."""
    costs = jnp.linalg.norm(
        target[:, None, :] - source[None, :, :],
        axis=-1,
    )

    return match_states(costs)


def match_states_by_conditional_mean(
    source: jax.Array,  # (T, K, N)
    target: jax.Array,  # (T, K, N)
) -> jax.Array:
    """Permutation aligning source to target by time-averaged conditional means."""
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
    """Permutation aligning inferred discrete states to ground-truth labels."""
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
    """Fit a source-to-target orthogonal Procrustes map with scaling and translation.

    The returned map minimizes squared errors between `alignment.apply(source)`
    and `target`.
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
    """Fit an unconstrained affine map from source to target by least squares.

    The returned map minimizes squared errors between `alignment.apply(source)`
    and `target`.
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


def zscore_gaussian(gaussian: Gaussian) -> Affine:
    """Return an affine map that z-scores a Gaussian distribution.

    For a batched Gaussian, first moment-match the batch into a single
    distribution. The returned map centers each variable at zero and scales
    it to unit marginal variance.
    """
    if len(gaussian.batch_shape) > 1:
        raise ValueError(
            'gaussian must be unbatched or have exactly one batch dimension'
        )

    if gaussian.batch_shape:
        gaussian = _gaussian_fit.from_moment_match(gaussian)

    scale = jnp.sqrt(gaussian.variance)

    coefficients = jnp.diag(1.0 / scale)
    bias = -gaussian.mean / scale

    return Affine(
        coefficients=coefficients,
        bias=bias,
    )


def zscore_samples(data: jax.Array, covariance_floor: float = 0.0) -> Affine:
    """Return an affine map that z-scores samples along the first axis."""
    return zscore_gaussian(
        _gaussian_fit.from_samples(
            data,
            covariance_floor=covariance_floor,
        )
    )
