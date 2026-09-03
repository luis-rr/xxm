import jax
import jax.numpy as jnp

from xxm.core.affine import Affine


def align_procrustes(
    source: jax.Array,
    target: jax.Array,
) -> Affine:
    """Fit a translation, global scale, and orthogonal transform."""

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
