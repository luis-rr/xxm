"""Centered Hermite basis mathematics and Gaussian coefficient drift."""

from __future__ import annotations

import dataclasses
import functools
import typing

import jax
import jax.numpy as jnp

from xxm.core.dists.gaussian import Gaussian


def _hermite_segment_data(
    positions: jax.Array,
    num_knots: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return knot-value and slope maps, segment indices, and local coordinates."""
    values_basis = jnp.eye(
        num_knots,
        dtype=positions.dtype,
    )

    # Slopes multiplied by the knot spacing. Interior centered derivatives are
    # 0.5 * (beta[j + 1] - beta[j - 1]); endpoint slopes are fixed to zero.
    slope_basis = jnp.zeros_like(values_basis)
    interior = jnp.arange(1, num_knots - 1)
    slope_basis = slope_basis.at[interior, interior - 1].set(-0.5)
    slope_basis = slope_basis.at[interior, interior + 1].set(0.5)

    scaled = positions * (num_knots - 1)
    segment = jnp.clip(
        jnp.floor(scaled).astype(jnp.int32),
        0,
        num_knots - 2,
    )
    u = scaled - segment
    return values_basis, slope_basis, segment, u


def _raw_hermite_basis(
    positions: jax.Array,
    num_knots: int,
) -> jax.Array:
    """Evaluate the unconstrained cubic Hermite knot-value basis."""
    values_basis, slope_basis, segment, u = _hermite_segment_data(positions, num_knots)

    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2

    return (
        h00[:, None] * values_basis[segment]
        + h10[:, None] * slope_basis[segment]
        + h01[:, None] * values_basis[segment + 1]
        + h11[:, None] * slope_basis[segment + 1]
    )


def _raw_hermite_second_derivative(
    positions: jax.Array,
    num_knots: int,
) -> jax.Array:
    """Evaluate second time derivatives of the unconstrained Hermite basis."""
    values_basis, slope_basis, segment, u = _hermite_segment_data(positions, num_knots)

    h00 = 12.0 * u - 6.0
    h10 = 6.0 * u - 4.0
    h01 = -12.0 * u + 6.0
    h11 = 6.0 * u - 2.0

    scale = (num_knots - 1) ** 2

    return scale * (
        h00[:, None] * values_basis[segment]
        + h10[:, None] * slope_basis[segment]
        + h01[:, None] * values_basis[segment + 1]
        + h11[:, None] * slope_basis[segment + 1]
    )


def _hermite_quadrature(
    num_knots: int,
    *,
    dtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array]:
    """Return four-point Gauss-Legendre quadrature over all knot intervals."""
    nodes = jnp.asarray(
        [
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ],
        dtype=dtype,
    )
    weights = jnp.asarray(
        [
            0.3478548451374538,
            0.6521451548625461,
            0.6521451548625461,
            0.3478548451374538,
        ],
        dtype=dtype,
    )

    num_segments = num_knots - 1

    segment = jnp.arange(
        num_segments,
        dtype=dtype,
    )[:, None]

    local_positions = 0.5 * (nodes[None, :] + 1.0)

    positions = ((segment + local_positions) / num_segments).reshape(-1)

    integration_weights = jnp.broadcast_to(
        weights[None, :] / (2.0 * num_segments),
        (num_segments, weights.shape[0]),
    ).reshape(-1)

    return positions, integration_weights


def _centered_hermite_map(
    num_knots: int,
    *,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    r"""Map $J-1$ free coefficients to zero-integral Hermite curves."""
    positions, weights = _hermite_quadrature(
        num_knots,
        dtype=dtype,
    )

    raw_basis = _raw_hermite_basis(
        positions,
        num_knots,
    )

    # mean_functional @ knot_values is the exact integral of the cubic
    # Hermite curve on [0, 1].
    mean_functional = jnp.einsum(
        't,tj->j',
        weights,
        raw_basis,
    )

    # The remaining columns span exactly the null space of the temporal mean.
    orthogonal, _ = jnp.linalg.qr(
        mean_functional[:, None],
        mode='complete',
    )

    return orthogonal[:, 1:]


def hermite_basis(
    num_steps: int,
    num_knots: int,
) -> jax.Array:
    r"""Construct the centered cubic Hermite basis for dynamics steps.

    Knot locations are equally spaced on $[0,1]$. Interior slopes use centered
    finite differences and endpoint slopes are fixed to zero. The returned
    basis has shape `(num_steps - 1, num_knots - 1)`.

    It evaluates a continuous temporal family satisfying

    $$\int_0^1 \lambda(s)\,ds = 0.$$

    Thus the temporal family contains no constant component while using
    exactly $J-1$ free coefficients for $J$ knots.

    Row `t - 1` corresponds to the incoming dynamics that generate `x[t]`
    from `x[t - 1]`, for `t = 1, ..., num_steps - 1`.
    """
    if num_steps < 1:
        raise ValueError('num_steps must be at least 1')

    if num_knots < 2:
        raise ValueError('num_knots must be at least 2')

    if num_steps == 1:
        return jnp.zeros((0, num_knots - 1))

    positions = jnp.linspace(
        0.0,
        1.0,
        num_steps - 1,
    )

    raw_basis = _raw_hermite_basis(
        positions,
        num_knots,
    )

    coefficient_map = _centered_hermite_map(
        num_knots,
        dtype=raw_basis.dtype,
    )

    return raw_basis @ coefficient_map


def hermite_penalty_matrices(
    num_knots: int,
    *,
    dtype: jax.typing.DTypeLike,
) -> tuple[jax.Array, jax.Array]:
    r"""Return amplitude and curvature penalties for the centered Hermite basis.

    For $\lambda(t)=H(t)\beta$,

    $$\beta^\top R_0\beta = \int_0^1 \lambda(t)^2\,dt,$$

    and

    $$\beta^\top R_2\beta = \int_0^1 \lambda''(t)^2\,dt.$$
    """
    if num_knots < 2:
        raise ValueError('num_knots must be at least 2')

    coefficient_map = _centered_hermite_map(
        num_knots,
        dtype=dtype,
    )

    positions, integration_weights = _hermite_quadrature(
        num_knots,
        dtype=dtype,
    )

    basis = (
        _raw_hermite_basis(
            positions,
            num_knots,
        )
        @ coefficient_map
    )

    second_derivative = (
        _raw_hermite_second_derivative(
            positions,
            num_knots,
        )
        @ coefficient_map
    )

    amplitude = jnp.einsum(
        't,ti,tj->ij',
        integration_weights,
        basis,
        basis,
    )

    curvature = jnp.einsum(
        't,ti,tj->ij',
        integration_weights,
        second_derivative,
        second_derivative,
    )

    amplitude = 0.5 * (amplitude + amplitude.T)
    curvature = 0.5 * (curvature + curvature.T)

    return amplitude, curvature


def _coefficient_prior(
    *,
    num_motifs: int,
    num_knots: int,
    alpha_amp: float,
    alpha_smooth: float,
    dtype: jax.typing.DTypeLike,
    batch_shape: tuple[int, ...] = (),
) -> Gaussian:
    r"""Construct the Gaussian prior over centered Hermite coefficients.

    For one motif,

    $$\Lambda_\beta
    =
    \alpha_{\rm amp}R_0
    +
    \alpha_{\rm smooth}R_2,$$

    where $R_0$ measures integrated squared temporal amplitude and $R_2$
    measures integrated squared temporal curvature.
    """
    amplitude, curvature = hermite_penalty_matrices(
        num_knots=num_knots,
        dtype=dtype,
    )

    coefficient_precision = alpha_amp * amplitude + alpha_smooth * curvature

    precision = jnp.kron(
        jnp.eye(
            num_motifs,
            dtype=dtype,
        ),
        coefficient_precision,
    )

    num_coefficients = num_knots - 1
    variable_dim = num_motifs * num_coefficients

    covariance = jnp.linalg.solve(
        precision,
        jnp.eye(
            variable_dim,
            dtype=dtype,
        ),
    )

    covariance = 0.5 * (covariance + covariance.T)

    return Gaussian(
        mean=jnp.zeros(
            batch_shape + (variable_dim,),
            dtype=dtype,
        ),
        covariance=jnp.broadcast_to(
            covariance,
            batch_shape + (variable_dim, variable_dim),
        ),
    )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['coefficient_prior'],
    meta_fields=['num_motifs'],
)
@dataclasses.dataclass(frozen=True)
class HermiteDrift:
    r"""Centered Hermite drift independently replicated over batch dimensions `*B`.

    The coefficient prior has batch shape `*B` and event dimension `P * C`,
    where `P` is the number of motifs and `C = J - 1` is the number of free
    centered Hermite coefficients per motif. Coefficients therefore have shape
    `(*B, P * C)` and realized temporal weights have shape `(T, *B, P)`.

    The same object can be used as a state-indexed temporal latent by placing
    the discrete-state axis first in the batch shape, `(K, *B)`. In that case
    `num_states` and `permute_states` operate on that first batch axis. Hermite
    trajectories are autonomous, so optional switching `gates` supplied to
    `sample` are validated for shape but do not affect the sampled trajectories.
    """

    coefficient_prior: Gaussian
    num_motifs: int

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Independent structural batch dimensions `*B`."""
        return self.coefficient_prior.batch_shape

    @property
    def variable_dim(self) -> int:
        """Number `P` of Hermite drift coordinates."""
        return self.num_motifs

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        """Coefficient-prior dtype."""
        return self.coefficient_prior.dtype

    @property
    def num_states(self) -> int:
        """Size `K` of the first batch axis when used as a switching latent."""
        if len(self.batch_shape) < 1:
            raise ValueError(
                'state-indexed Hermite drift must have batch shape (K, *B)'
            )

        if self.batch_shape[0] < 1:
            raise ValueError('Hermite drift must contain at least one state')

        return self.batch_shape[0]

    @property
    def num_coefficients(self) -> int:
        """Number of free centered Hermite coefficients per motif."""
        if self.num_motifs < 1:
            raise ValueError('drift must contain at least one motif')

        if self.coefficient_prior.variable_dim % self.num_motifs != 0:
            raise ValueError(
                'coefficient prior dimension must be divisible by the number of motifs'
            )

        return self.coefficient_prior.variable_dim // self.num_motifs

    @property
    def num_knots(self) -> int:
        """Number of Hermite knots defining the temporal basis."""
        return self.num_coefficients + 1

    def temporal_weights(
        self,
        coefficients: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        r"""Compute motif weights from batched centered coefficients.

        `coefficients` has shape `(*B, P * C)` and the returned weights have
        shape `(num_steps - 1, *B, P)`.
        """
        expected_shape = self.batch_shape + (self.num_motifs * self.num_coefficients,)

        if coefficients.shape != expected_shape:
            raise ValueError(
                f'coefficients must have shape {expected_shape}; '
                f'got {coefficients.shape}'
            )

        basis = hermite_basis(
            num_steps=num_steps,
            num_knots=self.num_knots,
        ).astype(coefficients.dtype)

        coefficients = coefficients.reshape(
            self.batch_shape
            + (
                self.num_motifs,
                self.num_coefficients,
            )
        )

        return jnp.einsum(
            'tc,...pc->t...p',
            basis,
            coefficients,
        )

    @classmethod
    def from_function_prior(
        cls,
        *,
        num_motifs: int,
        num_knots: int,
        alpha_amp: float,
        alpha_smooth: float,
        dtype: jax.typing.DTypeLike = jnp.float32,
        batch_shape: tuple[int, ...] = (),
    ) -> typing.Self:
        """Construct identical amplitude/curvature priors over batch `*B`."""
        if num_motifs < 1:
            raise ValueError('num_motifs must be at least 1')
        if num_knots < 2:
            raise ValueError('num_knots must be at least 2')
        if alpha_amp <= 0.0:
            raise ValueError('alpha_amp must be positive')
        if alpha_smooth < 0.0:
            raise ValueError('alpha_smooth must be non-negative')
        if any(size < 1 for size in batch_shape):
            raise ValueError('batch dimensions must all be positive')

        return cls(
            coefficient_prior=_coefficient_prior(
                num_motifs=num_motifs,
                num_knots=num_knots,
                alpha_amp=alpha_amp,
                alpha_smooth=alpha_smooth,
                dtype=dtype,
                batch_shape=batch_shape,
            ),
            num_motifs=num_motifs,
        )

    def sample_coefficients(self, key: jax.Array) -> jax.Array:
        """Sample coefficients with shape `(*B, P * C)`."""
        return self.coefficient_prior.sample(key)

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
        gates: jax.Array | None = None,
    ) -> jax.Array:
        r"""Sample `num_steps` temporal values with shape `(T, *B, P)`.

        `gates` is accepted so a state-batched Hermite drift satisfies the
        switching temporal-latent interface. Hermite trajectories are not
        state-gated, so gate values do not affect the sample.
        """
        if num_steps < 0:
            raise ValueError('num_steps must be non-negative')

        if gates is not None:
            _ = self.num_states

            expected_gate_shape = (max(num_steps - 1, 0),) + self.batch_shape[1:]

            if gates.shape != expected_gate_shape:
                raise ValueError(
                    f'gates must have shape {expected_gate_shape}, got {gates.shape}'
                )

        return self.temporal_weights(
            self.sample_coefficients(key),
            num_steps=num_steps + 1,
        )

    def log_prob(self, coefficients: jax.Array) -> jax.Array:
        """Evaluate coefficient prior densities over batch `*B`."""
        return self.coefficient_prior.log_prob(coefficients)

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel the distinguished first batch axis `K`."""
        permutation = jnp.asarray(permutation)

        if permutation.shape != (self.num_states,):
            raise ValueError(f'permutation must have shape {(self.num_states,)}')

        return dataclasses.replace(
            self,
            coefficient_prior=self.coefficient_prior.select(permutation),
        )

    def reorient_variables(
        self,
        signs: jax.Array,
    ) -> typing.Self:
        """Reorient motif coordinates globally or independently by batch."""
        signs = jnp.asarray(signs)

        global_shape = (self.num_motifs,)
        batched_shape = self.batch_shape + global_shape

        if signs.shape not in (global_shape, batched_shape):
            raise ValueError(
                f'signs must have shape {global_shape} or {batched_shape}, '
                f'got {signs.shape}'
            )

        coefficient_signs = jnp.repeat(
            signs,
            self.num_coefficients,
            axis=-1,
        )

        return dataclasses.replace(
            self,
            coefficient_prior=self.coefficient_prior.reorient_variables(
                coefficient_signs
            ),
        )

    def permute_variables(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel motif coordinates globally or independently by batch."""
        permutation = jnp.asarray(permutation)

        global_shape = (self.num_motifs,)
        batched_shape = self.batch_shape + global_shape

        if permutation.shape not in (global_shape, batched_shape):
            raise ValueError(
                'permutation must have shape '
                f'{global_shape} or {batched_shape}, '
                f'got {permutation.shape}'
            )

        coefficient_permutation = (
            permutation[..., :, None] * self.num_coefficients
            + jnp.arange(
                self.num_coefficients,
                dtype=permutation.dtype,
            )
        ).reshape(permutation.shape[:-1] + (self.num_motifs * self.num_coefficients,))

        return dataclasses.replace(
            self,
            coefficient_prior=self.coefficient_prior.permute_variables(
                coefficient_permutation
            ),
        )
