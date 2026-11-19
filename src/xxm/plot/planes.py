r"""Linear dynamics restricted to 2D planes, for viewing high-dimensional flows.

A plane is an affine embedding $x = o + B z$ of plane coordinates $z$, with an
orthonormal basis $B$ `(D, 2)` and origin $o$, stored as an `Affine`
`(2,) -> (D,)`. Its pseudoinverse $z = B^\top (x - o)$ gives plane
coordinates. Restricted dynamics are `Affine` maps `(2,) -> (2,)` of plane
coordinates, which `plot_dyn_linear` draws as the displacement $f(z) - z$.

Functions accept arbitrary, explicitly aligned batch shapes `*B`, except
`frame_planes` and `plane_intersection`, which build unbatched geometry.
"""

import itertools

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


def orthonormal_embedding(embedding: Affine) -> Affine:
    r"""The same planes as `embedding`, with orthonormal bases, e.g. for drawing.

    For planes $z \mapsto o + Gz$ with independent columns $G$ `(*B, D, 2)`,
    e.g. a `conditional_lift`, the basis is the polar factor
    $\hat G = G (G^\top G)^{-1/2} = U V^\top$ of $G = U S V^\top$: the
    orthonormal basis of the plane closest to $G$. The origin $o$ is kept. A
    square of plane coordinates then maps to a square in space.
    """
    if embedding.input_shape != (2,):
        raise ValueError(
            f'expected plane embeddings (2,) -> (D,); got {embedding.input_shape}'
        )
    left, _, right_t = jnp.linalg.svd(embedding.coefficients, full_matrices=False)
    return plane_embedding(left @ right_t, embedding.bias)


def recentred_embedding(embedding: Affine, point: jax.Array) -> Affine:
    r"""The same planes as `embedding`, with origin at `point` projected onto them.

    For orthonormal planes $z \mapsto o + Bz$ (e.g. from
    `orthonormal_embedding`) and `point` $p$ `(*B, D)`, the new origin is
    $o + B B^\top (p - o)$, the point of the plane closest to $p$.
    """
    basis = embedding.coefficients  # (*B, D, 2)
    origin = embedding.bias  # (*B, D)
    point = jnp.asarray(point)
    if point.shape != origin.shape:
        raise ValueError(f'point must have shape {origin.shape}; got {point.shape}')

    offset = jnp.einsum('...di,...d->...i', basis, point - origin)
    return plane_embedding(
        basis, origin + jnp.einsum('...di,...i->...d', basis, offset)
    )


def frame_planes(frame: jax.Array) -> Affine:
    r"""Planes spanned by pairs of columns of a frame, batch `(K choose 2,)`.

    `frame` `(D, K)` has orthonormal columns $q_1, \dots, q_K$. Plane
    $(i, j)$, $i < j$, has basis $[q_i, q_j]$; planes come in the order of
    `itertools.combinations`, e.g. $(q_1, q_2), (q_1, q_3), (q_2, q_3)$ for
    $K = 3$. `frame_planes(jnp.eye(3))` gives the coordinate planes.
    """
    frame = jnp.asarray(frame)
    if frame.ndim != 2 or frame.shape[-1] < 2:
        raise ValueError(f'frame must have shape (D, K) with K >= 2; got {frame.shape}')

    pairs = itertools.combinations(range(frame.shape[-1]), 2)
    basis = jnp.stack([frame[:, [i, j]] for i, j in pairs])  # (K choose 2, D, 2)
    return plane_embedding(basis)


def plane_intersection(embedding0: Affine, embedding1: Affine) -> Affine:
    r"""Line where two planes of 3D space meet, $t \mapsto p + t\,d$, `(1,) -> (3,)`.

    For planes $x = o_k + B_k z$ with normals $n_k = b_{k,1} \times b_{k,2}$,
    the direction is the unit $d \propto n_0 \times n_1$ and $p$ the point of
    the line closest to the origin. Bases need not be orthonormal, so
    conditional lifts work too. Planes must be unbatched and not parallel.
    """
    for embedding in (embedding0, embedding1):
        if embedding.batch_shape or embedding.input_shape != (2,):
            raise ValueError('expected unbatched plane embeddings (2,) -> (D,)')
        if embedding.output_dim != 3:
            raise ValueError(
                f'plane intersections need D = 3; got D = {embedding.output_dim}'
            )

    normals = [
        jnp.cross(embedding.coefficients[:, 0], embedding.coefficients[:, 1])
        for embedding in (embedding0, embedding1)
    ]
    normals = [normal / jnp.linalg.norm(normal) for normal in normals]
    direction = jnp.cross(normals[0], normals[1])
    length = jnp.linalg.norm(direction)
    if not bool(length > ORTHONORMAL_ATOL):
        raise ValueError('parallel planes have no intersection line')
    direction = direction / length

    # p lies on both planes, n_k · p = n_k · o_k, and is closest to the origin, d · p = 0
    system = jnp.stack([*normals, direction])  # (3, 3)
    targets = jnp.stack(
        [
            normals[0] @ embedding0.bias,
            normals[1] @ embedding1.bias,
            jnp.zeros((), dtype=direction.dtype),
        ]
    )
    point = jnp.linalg.solve(system, targets)

    return Affine(coefficients=direction[:, None], bias=point)


def plane_coordinates(embedding: Affine, values: jax.Array) -> jax.Array:
    r"""Plane coordinates $z = B^\top (x - o)$ of `values` shaped `(*B, *Q, D)`."""
    return embedding.pseudoinverse().apply(jnp.asarray(values))


def slice_dynamics(dynamics: Affine, embedding: Affine) -> Affine:
    r"""Dynamics in plane coordinates for states on the plane, $z \mapsto P(f(o + Bz))$.

    `lift_dynamics` with the embedding as the lift: a slice through the plane.
    """
    return lift_dynamics(dynamics, embedding, embedding)


def lift_dynamics(dynamics: Affine, embedding: Affine, lift: Affine) -> Affine:
    r"""Dynamics in plane coordinates, $z \mapsto P(f(L(z)))$.

    $f$ is `dynamics` `(D,) -> (D,)`, $P$ the projection
    `embedding.pseudoinverse()` and $L$ the `lift` `(2,) -> (D,)`, e.g.
    `conditional_lift`. All maps share one batch shape.
    """
    return embedding.pseudoinverse().compose(dynamics.compose(lift))


def conditional_lift(embedding: Affine, gaussian: FullGaussian) -> Affine:
    r"""Conditional mean $z \mapsto \mathbb{E}[x \mid B^\top (x - o) = z]$.

    For $x \sim \mathcal{N}(\mu, \Sigma)$ and gain
    $G = \Sigma B (B^\top \Sigma B)^{-1}$, this is the affine map
    $z \mapsto \mu + G (z - B^\top (\mu - o))$. Used as the `lift` of
    `lift_dynamics`, it gives the expected displacement given the position
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


def displacement_gain(dynamics: Affine) -> jax.Array:
    r"""Spectral norm $\|A - I\|_2$ of square affine maps, shape `(*B,)`.

    The displacement $f(x) - x = (A - I)\,x + b$ changes by at most
    $\|A - I\|_2$ per unit step in $x$: the strength of the flow, independent
    of where it is centered. Non-finite maps give NaN.
    """
    if dynamics.input_shape != (dynamics.output_dim,):
        raise ValueError(
            'displacement gain requires a square vector-to-vector map, '
            f'got {dynamics.input_shape} -> ({dynamics.output_dim},)'
        )

    identity = jnp.eye(dynamics.output_dim, dtype=dynamics.dtype)
    gain = jnp.linalg.norm(dynamics.coefficients - identity, ord=2, axis=(-2, -1))
    finite = jnp.isfinite(dynamics.coefficients).all(axis=(-2, -1))
    return jnp.where(finite, gain, jnp.nan)
