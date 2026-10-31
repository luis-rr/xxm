"""State alignment and coordinate alignment utilities."""

import itertools
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.dists.gaussian import FullGaussian
from xxm.core.optim import gaussian_full as _gaussian_full_fit

PermutationEval = typing.Callable[[jax.Array], jax.Array]


def maximize_permutation_exhaustive(
    objective: PermutationEval,
    size: int,
) -> jax.Array:
    """Maximize an objective over all permutations."""
    permutations = jnp.asarray(
        list(itertools.permutations(range(size))),
        dtype=jnp.int32,
    )

    scores = jax.vmap(objective)(permutations)
    return permutations[jnp.argmax(scores)]


def _improve_permutation(
    objective: PermutationEval,
    start: jax.Array,
) -> jax.Array:
    """Improve a permutation through best pair swaps."""
    size = start.shape[0]

    pairs = jnp.asarray(
        list(itertools.combinations(range(size), 2)),
        dtype=jnp.int32,
    ).reshape(-1, 2)

    def condition(carry):
        _, _, improved = carry
        return improved

    def step(carry):
        current, current_score, _ = carry

        def swap(pair):
            i, j = pair
            return current.at[i].set(current[j]).at[j].set(current[i])

        neighbors = jax.vmap(swap)(pairs)

        candidates = jnp.concatenate(
            (current[None], neighbors),
            axis=0,
        )

        scores = jax.vmap(objective)(candidates)
        best = jnp.argmax(scores)

        return (
            candidates[best],
            scores[best],
            scores[best] > current_score,
        )

    best, _, _ = jax.lax.while_loop(
        condition,
        step,
        (
            start,
            objective(start),
            jnp.asarray(True),
        ),
    )

    return best


def maximize_permutation_random(
    objective: PermutationEval,
    start: jax.Array,
    key: jax.Array,
    *,
    num_restarts: int = 16,
) -> jax.Array:
    """Maximize using multistart pair-swap local search."""
    size = start.shape[0]

    random_starts = jax.vmap(lambda k: jax.random.permutation(k, size))(
        jax.random.split(key, num_restarts)
    )

    starts = jnp.concatenate(
        (
            start[None],
            jnp.arange(size, dtype=start.dtype)[None],
            random_starts,
        ),
        axis=0,
    )

    candidates = jax.vmap(
        lambda permutation: _improve_permutation(
            objective,
            permutation,
        )
    )(starts)

    scores = jax.vmap(objective)(candidates)
    return candidates[jnp.argmax(scores)]


# TODO: switch everything to use maximization objectives instead of minimzation of costs.
def match_permutation(costs: jax.Array) -> jax.Array:
    """Permutation minimizing total pairwise matching cost.

    `costs[target, source]` is the cost of matching one source item to one
    target item. Returns the permutation to apply to source so that its
    leading dimension follows target ordering.
    """
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError(f'costs must be a square matrix, got shape {costs.shape}')

    permutation = min(
        itertools.permutations(range(costs.shape[0])),
        key=lambda p: sum(costs[k, p[k]] for k in range(len(p))),
    )

    return jnp.asarray(permutation)


def match_states(costs: jax.Array) -> jax.Array:
    """Permutation minimizing total state-matching cost."""
    return match_permutation(costs)


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


def align_orthogonal(
    source: jax.Array,
    target: jax.Array,
) -> Affine:
    """Fit a source-to-target orthogonal map with no scaling or translation."""
    u, _, vt = jnp.linalg.svd(
        source.T @ target,
        full_matrices=False,
    )

    orthogonal = u @ vt

    return Affine(
        coefficients=orthogonal.T,
        bias=jnp.zeros(
            target.shape[-1],
            dtype=target.dtype,
        ),
    )


def sign_match(
    source: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """Return signs that best align batched source arrays to target arrays."""
    if source.shape != target.shape:
        raise ValueError(
            'source and target must have the same shape, '
            f'got {source.shape} and {target.shape}'
        )

    if source.ndim < 2:
        raise ValueError(
            'expected a leading batch dimension and at least one value dimension'
        )

    value_axes = tuple(range(1, source.ndim))
    inner_product = jnp.sum(
        source * target,
        axis=value_axes,
    )

    return jnp.where(
        inner_product >= 0,
        jnp.ones_like(inner_product),
        -jnp.ones_like(inner_product),
    )


def match_signed_permutation(
    source: jax.Array,
    target: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Match batched arrays up to permutation and independent signs.

    The leading dimension indexes items to match. The remaining dimensions
    define each item.

    Returns the permutation to apply to `source`, followed by the signs to
    apply after permutation.
    """
    if source.shape != target.shape:
        raise ValueError(
            'source and target must have the same shape, '
            f'got {source.shape} and {target.shape}'
        )

    if source.ndim < 2:
        raise ValueError(
            'expected a leading batch dimension and at least one value dimension'
        )

    value_axes = tuple(range(2, source.ndim + 1))

    inner_products = jnp.sum(
        target[:, None] * source[None, :],
        axis=value_axes,
    )

    permutation = match_permutation(
        -jnp.abs(inner_products),
    )

    signs = sign_match(
        source[permutation],
        target,
    )

    return permutation, signs


def zscore_gaussian(gaussian: FullGaussian) -> Affine:
    """Return an affine map that z-scores a Gaussian distribution.

    For a batched Gaussian, first moment-match the batch into a single
    distribution. Structural batch entries are flattened and equally
    moment-matched before computing the z-score transform. The returned map
    centers each variable at zero and scales it to unit marginal variance.
    """
    if gaussian.batch_shape:
        gaussian = batch.flatten_batch(
            gaussian,
            gaussian.batch_shape,
        )
        gaussian = _gaussian_full_fit.from_moment_match(gaussian)

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
        _gaussian_full_fit.from_samples(
            data,
            covariance_floor=covariance_floor,
        )
    )


def align_lda(
    latent: jax.Array,  # (T, D)
    labels: jax.Array,  # (T,), integer labels
    ridge: float = 1e-6,
) -> Affine:
    num_classes = int(jnp.max(labels)) + 1
    dim = latent.shape[-1]

    mean = jnp.mean(latent, axis=0)

    counts = jnp.bincount(labels, length=num_classes)
    sums = jax.ops.segment_sum(
        latent,
        labels,
        num_segments=num_classes,
    )
    class_means = sums / counts[:, None]

    centered_within = latent - class_means[labels]

    within = centered_within.T @ centered_within
    within = within + ridge * jnp.eye(dim)

    centered_classes = class_means - mean
    between = centered_classes.T @ (counts[:, None] * centered_classes)

    chol = jnp.linalg.cholesky(within)

    tmp = jax.scipy.linalg.solve_triangular(
        chol,
        between,
        lower=True,
    )
    whitened_between = jax.scipy.linalg.solve_triangular(
        chol,
        tmp.T,
        lower=True,
    ).T

    eigenvalues, eigenvectors_white = jnp.linalg.eigh(whitened_between)

    order = jnp.argsort(eigenvalues)[::-1]
    eigenvectors_white = eigenvectors_white[:, order]

    directions = jax.scipy.linalg.solve_triangular(
        chol.T,
        eigenvectors_white,
        lower=False,
    )

    coefficients = directions.T
    bias = -coefficients @ mean

    return Affine(
        coefficients=coefficients,
        bias=bias,
    )
