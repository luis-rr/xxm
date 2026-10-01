"""Centered cubic Hermite basis mathematics."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp


class HermiteSegments(typing.NamedTuple):
    values_basis: jax.Array
    slope_basis: jax.Array
    segment: jax.Array
    local_position: jax.Array


class HermiteQuadrature(typing.NamedTuple):
    positions: jax.Array
    weights: jax.Array


class HermitePenalties(typing.NamedTuple):
    amplitude: jax.Array
    curvature: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class HermiteBasis:
    """Uniform knots defining a zero-integral cubic Hermite basis.

    `num_knots` is static PyTree metadata because it determines array shapes.
    """

    num_knots: int = jax.tree.static()

    @classmethod
    def from_num_knots(cls, num_knots: int) -> typing.Self:
        if num_knots < 2:
            raise ValueError('num_knots must be at least 2')
        return cls(num_knots)

    @property
    def num_coefficients(self) -> int:
        return self.num_knots - 1

    def at_positions(self, positions: jax.Array) -> HermiteEvaluation:
        return HermiteEvaluation.from_positions(self, positions)

    def regularly_spaced(self, num_steps: int) -> HermiteEvaluation:
        """Construct exactly `num_steps` positions on `[0, 1]`."""
        if num_steps < 0:
            raise ValueError('num_steps must be nonnegative')
        return self.at_positions(jnp.linspace(0.0, 1.0, num_steps))

    def quadrature(self, *, dtype: jax.typing.DTypeLike) -> HermiteQuadrature:
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

        num_segments = self.num_knots - 1

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

        return HermiteQuadrature(positions, integration_weights)

    def coefficient_map(self, *, dtype: jax.typing.DTypeLike) -> jax.Array:
        r"""Map $J-1$ free coefficients to zero-integral Hermite curves."""
        positions, weights = self.quadrature(
            dtype=dtype,
        )

        raw_basis = self.at_positions(positions).raw_basis()

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

    def penalty_matrices(self, *, dtype: jax.typing.DTypeLike) -> HermitePenalties:
        r"""Return amplitude and curvature penalties for the centered Hermite basis.

        For $\lambda(t)=H(t)\beta$,

        $$\beta^\top R_0\beta = \int_0^1 \lambda(t)^2\,dt,$$

        and

        $$\beta^\top R_2\beta = \int_0^1 \lambda''(t)^2\,dt.$$
        """
        coefficient_map = self.coefficient_map(
            dtype=dtype,
        )

        positions, integration_weights = self.quadrature(
            dtype=dtype,
        )

        evaluation = self.at_positions(positions)
        basis = evaluation.raw_basis() @ coefficient_map

        second_derivative = evaluation.raw_second_derivative() @ coefficient_map

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

        return HermitePenalties(amplitude, curvature)


class HermiteEvaluation(typing.NamedTuple):
    """A Hermite basis evaluated at positions with arbitrary leading dimensions."""

    basis: HermiteBasis
    positions: jax.Array

    @classmethod
    def from_positions(cls, basis: HermiteBasis, positions: jax.Array) -> typing.Self:
        positions = jnp.asarray(positions)
        positions = positions.astype(jnp.result_type(positions, jnp.float32))
        return cls(basis, positions)

    def segment_data(self) -> HermiteSegments:
        """Return knot-value and slope maps, segment indices, and local coordinates."""
        values_basis = jnp.eye(
            self.basis.num_knots,
            dtype=self.positions.dtype,
        )

        # Slopes multiplied by the knot spacing. Interior centered derivatives are
        # 0.5 * (beta[j + 1] - beta[j - 1]); endpoint slopes are fixed to zero.
        slope_basis = jnp.zeros_like(values_basis)
        interior = jnp.arange(1, self.basis.num_knots - 1)
        slope_basis = slope_basis.at[interior, interior - 1].set(-0.5)
        slope_basis = slope_basis.at[interior, interior + 1].set(0.5)

        scaled = self.positions * (self.basis.num_knots - 1)
        segment = jnp.clip(
            jnp.floor(scaled).astype(jnp.int32),
            0,
            self.basis.num_knots - 2,
        )
        local_position = scaled - segment
        return HermiteSegments(values_basis, slope_basis, segment, local_position)

    def raw_basis(self) -> jax.Array:
        """Evaluate the unconstrained cubic Hermite knot-value basis."""
        values_basis, slope_basis, segment, local_position = self.segment_data()

        h00 = 2.0 * local_position**3 - 3.0 * local_position**2 + 1.0
        h10 = local_position**3 - 2.0 * local_position**2 + local_position
        h01 = -2.0 * local_position**3 + 3.0 * local_position**2
        h11 = local_position**3 - local_position**2

        return (
            h00[..., None] * values_basis[segment]
            + h10[..., None] * slope_basis[segment]
            + h01[..., None] * values_basis[segment + 1]
            + h11[..., None] * slope_basis[segment + 1]
        )

    def raw_second_derivative(self) -> jax.Array:
        """Evaluate second time derivatives of the unconstrained Hermite basis."""
        values_basis, slope_basis, segment, local_position = self.segment_data()

        h00 = 12.0 * local_position - 6.0
        h10 = 6.0 * local_position - 4.0
        h01 = -12.0 * local_position + 6.0
        h11 = 6.0 * local_position - 2.0

        scale = (self.basis.num_knots - 1) ** 2

        return scale * (
            h00[..., None] * values_basis[segment]
            + h10[..., None] * slope_basis[segment]
            + h01[..., None] * values_basis[segment + 1]
            + h11[..., None] * slope_basis[segment + 1]
        )

    def centered_basis(self) -> jax.Array:
        """Return shape `(*positions.shape, num_knots - 1)`."""
        return self.raw_basis() @ self.basis.coefficient_map(dtype=self.positions.dtype)
