r"""Linear dynamics restricted to 2D planes, for viewing high-dimensional flows.

A plane is an affine embedding $x = o + B z$ of plane coordinates $z$, with an
orthonormal basis $B$ `(D, 2)` and origin $o$, stored as an `Affine`
`(2,) -> (D,)`. Its pseudoinverse $z = B^\top (x - o)$ gives plane
coordinates. Restricted dynamics are `Affine` maps `(2,) -> (2,)` of plane
coordinates, which `plot_dyn_linear` draws as the displacement $f(z) - z$.

All functions accept arbitrary, explicitly aligned batch shapes `*B`.
"""

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.dists.gaussian import FullGaussian

ORTHONORMAL_ATOL = 1e-5


def plane_embedding(basis: jax.Array, origin: jax.Array | None = None) -> Affine:
    r"""Embedding $z \mapsto o + B z$ of plane coordinates into the full space.

    `basis` has shape `(*B, D, 2)` with orthonormal columns; `origin` has
    shape `(*B, D)` and defaults to zero.
    """
    basis = jnp.asarray(basis)
    if basis.ndim < 2 or basis.shape[-1] != 2:
        raise ValueError(f'basis must have shape (*B, D, 2); got {basis.shape}')

    gram = jnp.einsum('...di,...dj->...ij', basis, basis)
    if not bool(
        jnp.allclose(gram, jnp.eye(2, dtype=gram.dtype), atol=ORTHONORMAL_ATOL)
    ):
        raise ValueError('basis columns must be orthonormal')

    if origin is None:
        origin = jnp.zeros(basis.shape[:-1], dtype=basis.dtype)
    origin = jnp.asarray(origin)
    if origin.shape != basis.shape[:-1]:
        raise ValueError(
            f'origin must have shape {basis.shape[:-1]}; got {origin.shape}'
        )

    return Affine(coefficients=basis, bias=origin)


def plane_coordinates(embedding: Affine, values: jax.Array) -> jax.Array:
    r"""Plane coordinates $z = B^\top (x - o)$ of `values` shaped `(*B, *Q, D)`."""
    return embedding.pseudoinverse().apply(jnp.asarray(values))


def restrict_dynamics(
    dynamics: Affine,
    embedding: Affine,
    *,
    lift: Affine | None = None,
) -> Affine:
    r"""Dynamics in plane coordinates, $z \mapsto P(f(L(z)))$.

    $f$ is `dynamics` `(D,) -> (D,)`, $P$ the projection
    `embedding.pseudoinverse()` and $L$ the `lift` `(2,) -> (D,)`. The default
    lift is the embedding itself: a slice through states on the plane. All
    maps share one batch shape.
    """
    lift = embedding if lift is None else lift
    return embedding.pseudoinverse().compose(dynamics.compose(lift))


def conditional_lift(embedding: Affine, gaussian: FullGaussian) -> Affine:
    r"""Conditional mean $z \mapsto \mathbb{E}[x \mid B^\top (x - o) = z]$.

    For $x \sim \mathcal{N}(\mu, \Sigma)$ and gain
    $G = \Sigma B (B^\top \Sigma B)^{-1}$, this is the affine map
    $z \mapsto \mu + G (z - B^\top (\mu - o))$. Used as the `lift` of
    `restrict_dynamics`, it gives the expected displacement given the position
    on the plane, accounting for the coordinates off the plane. Batch shapes
    must match.
    """
    batch.require_same_shape(embedding.batch_shape, gaussian.batch_shape)

    basis = embedding.coefficients  # (*B, D, 2)
    origin = embedding.bias  # (*B, D)
    mean = gaussian.mean  # (*B, D)
    covariance = gaussian.covariance  # (*B, D, D)

    cross = covariance @ basis  # (*B, D, 2): Σ B
    plane_covariance = jnp.swapaxes(basis, -1, -2) @ cross  # (*B, 2, 2): Bᵀ Σ B
    # G = Σ B (Bᵀ Σ B)⁻¹, via the symmetric solve Gᵀ = (Bᵀ Σ B)⁻¹ (Σ B)ᵀ
    gain = jnp.swapaxes(
        jnp.linalg.solve(plane_covariance, jnp.swapaxes(cross, -1, -2)), -1, -2
    )  # (*B, D, 2)

    mean_coordinates = jnp.einsum('...di,...d->...i', basis, mean - origin)
    bias = mean - jnp.einsum('...di,...i->...d', gain, mean_coordinates)

    return Affine(coefficients=gain, bias=bias)


def fixed_points(dynamics: Affine) -> jax.Array:
    r"""Fixed points $x^* = (I - A)^{-1} b$ of square affine maps, shape `(*B, D)`."""
    if dynamics.input_shape != (dynamics.output_dim,):
        raise ValueError(
            'fixed points require a square vector-to-vector map, '
            f'got {dynamics.input_shape} -> ({dynamics.output_dim},)'
        )

    identity = jnp.eye(dynamics.output_dim, dtype=dynamics.dtype)
    return jnp.linalg.solve(identity - dynamics.coefficients, dynamics.bias[..., None])[
        ..., 0
    ]
