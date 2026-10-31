"""Gaussian distributions over centered temporal Hermite functions."""

from __future__ import annotations

import dataclasses
import functools
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.data import Sequences
from xxm.core.dists.gaussian import FullGaussian
from xxm.core.latents.hermite_core import HermiteBasis
from xxm.core.lengths import LengthsSpec
from xxm.core.potentials.gaussian import GaussianPotential


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['coefficients'],
    meta_fields=['basis'],
)
@dataclasses.dataclass(frozen=True)
class HermiteGaussian:
    r"""Gaussian distribution over centered temporal Hermite functions.

    `coefficients` has batch shape `*B` and event dimension `P * C`,
    where `C = basis.num_coefficients` and `P` is the number of temporal
    variables.

    Temporal realizations therefore have shape `(*B, T, P)`.
    """

    coefficients: FullGaussian
    basis: HermiteBasis

    def __post_init__(self) -> None:
        if self.coefficients.variable_dim < self.basis.num_coefficients:
            raise ValueError(
                'coefficient dimension must contain at least one Hermite variable'
            )

        if self.coefficients.variable_dim % self.basis.num_coefficients:
            raise ValueError(
                'coefficient dimension must be divisible by the Hermite basis size'
            )

    @classmethod
    def from_coefficient_gaussian(
        cls,
        coefficients: FullGaussian,
        *,
        variable_dim: int,
    ) -> typing.Self:
        """Infer the Hermite basis from a flattened coefficient Gaussian."""
        if variable_dim < 1:
            raise ValueError('variable_dim must be at least 1')

        if coefficients.variable_dim % variable_dim:
            raise ValueError('coefficient dimension must be divisible by variable_dim')

        num_coefficients = coefficients.variable_dim // variable_dim

        return cls(
            coefficients=coefficients,
            basis=HermiteBasis.from_num_knots(num_coefficients + 1),
        )

    @classmethod
    def from_function_prior(
        cls,
        *,
        variable_dim: int,
        num_knots: int,
        alpha_amp: float,
        alpha_smooth: float,
        dtype: jax.typing.DTypeLike = jnp.float32,
        batch_shape: tuple[int, ...] = (),
    ) -> typing.Self:
        """Construct identical amplitude/curvature priors over batch `*B`."""
        if variable_dim < 1:
            raise ValueError('variable_dim must be at least 1')

        if alpha_amp <= 0.0:
            raise ValueError('alpha_amp must be positive')

        if alpha_smooth < 0.0:
            raise ValueError('alpha_smooth must be non-negative')

        if any(size < 1 for size in batch_shape):
            raise ValueError('batch dimensions must all be positive')

        basis = HermiteBasis.from_num_knots(num_knots)

        penalties = basis.penalty_matrices(dtype=dtype)

        coefficient_precision = (
            alpha_amp * penalties.amplitude + alpha_smooth * penalties.curvature
        )

        precision = jnp.kron(
            jnp.eye(
                variable_dim,
                dtype=dtype,
            ),
            coefficient_precision,
        )

        coefficient_dim = variable_dim * basis.num_coefficients

        precision = jnp.broadcast_to(
            precision,
            batch_shape
            + (
                coefficient_dim,
                coefficient_dim,
            ),
        )

        coefficients = FullGaussian.from_canonical(
            precision=precision,
            information=jnp.zeros(
                batch_shape + (coefficient_dim,),
                dtype=dtype,
            ),
        )

        return cls(
            coefficients=coefficients,
            basis=basis,
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Independent structural batch dimensions `*B`."""
        return self.coefficients.batch_shape

    @property
    def variable_dim(self) -> int:
        """Number `P` of temporal Hermite variables."""
        return self.coefficients.variable_dim // self.num_coefficients

    @property
    def dtype(
        self,
    ) -> jax.typing.DTypeLike:
        return self.coefficients.dtype

    @property
    def num_coefficients(self) -> int:
        """Number `C` of free centered coefficients per temporal variable."""
        return self.basis.num_coefficients

    @property
    def num_knots(self) -> int:
        return self.basis.num_knots

    def reorient_variables(
        self,
        signs: jax.Array,
    ) -> typing.Self:
        """Flip temporal-variable orientations."""
        signs = batch.broadcast_trailing(
            signs,
            self.batch_shape,
            (self.variable_dim,),
        )

        coefficient_signs = jnp.repeat(
            signs,
            self.num_coefficients,
            axis=-1,
        )

        return dataclasses.replace(
            self,
            coefficients=self.coefficients.reorient_variables(coefficient_signs),
        )

    def permute_variables(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel temporal variables while preserving coefficient groups."""
        permutation = batch.broadcast_trailing(
            permutation,
            self.batch_shape,
            (self.variable_dim,),
        )

        coefficient_permutation = (
            permutation[..., :, None] * self.num_coefficients
            + jnp.arange(
                self.num_coefficients,
                dtype=permutation.dtype,
            )
        ).reshape(self.batch_shape + (self.coefficients.variable_dim,))

        return dataclasses.replace(
            self,
            coefficients=self.coefficients.permute_variables(coefficient_permutation),
        )

    def with_coefficients(
        self,
        coefficients: FullGaussian,
    ) -> typing.Self:
        """Replace coefficient moments while preserving Hermite structure."""
        result = type(self)(
            coefficients=coefficients,
            basis=self.basis,
        )

        return result

    def temporal_basis(
        self,
        ls: LengthsSpec,
    ) -> jax.Array:
        """Evaluate the centered basis on normalized sequence-local time grids.

        Returns shape `(*S, T, C)`, where `*S = ls.batch_shape`.

        The grid batch intentionally need not equal `self.batch_shape`.
        """
        denominator = ls.lengths.shift(-1).clip_min(1).values

        positions = (
            jnp.arange(
                ls.max_steps,
                dtype=self.dtype,
            )
            / denominator[..., None]
        )

        values = self.basis.at_positions(positions).centered_basis().astype(self.dtype)

        return ls.lengths.clear(values)

    def sample_coefficients(
        self,
        key: jax.Array,
    ) -> jax.Array:
        """Sample flattened coefficient vectors `(*B, P * C)`."""
        return self.coefficients.sample(key)

    def sample(
        self,
        key: jax.Array,
        ls: LengthsSpec,
    ) -> Sequences:
        """Sample temporal values on exactly the timeline described by `ls`."""
        return self.sample_from_coefficients(
            self.sample_coefficients(key),
            ls,
        )

    def sample_from_coefficients(
        self,
        coefficients: jax.Array,
        ls: LengthsSpec,
    ) -> Sequences:
        """Evaluate fixed flattened coefficients on exactly `ls`."""
        batch.require_same_shape(
            self.batch_shape,
            ls.batch_shape,
        )

        expected = self.batch_shape + (self.coefficients.variable_dim,)

        if coefficients.shape != expected:
            raise ValueError(
                f'coefficients must have shape {expected}; got {coefficients.shape}'
            )

        coefficients = coefficients.reshape(
            self.batch_shape
            + (
                self.variable_dim,
                self.num_coefficients,
            )
        )

        values = jnp.einsum(
            '...tc,...pc->...tp',
            self.temporal_basis(ls).astype(coefficients.dtype),
            coefficients,
        )

        return Sequences.from_padded(
            values,
            ls.lengths,
        )

    def marginals(
        self,
        ls: LengthsSpec,
    ) -> FullGaussian:
        """Project coefficient uncertainty to masked temporal marginals."""
        batch.require_same_shape(
            self.batch_shape,
            ls.batch_shape,
        )

        return self.marginals_from_basis(self.temporal_basis(ls))

    def marginals_from_basis(
        self,
        basis_values: jax.Array,
    ) -> FullGaussian:
        """Project coefficient moments through an evaluated temporal basis.

        The basis has shape (*B, T, C). Invalid times should already
        be zeroed in the supplied basis.
        """
        if basis_values.ndim != len(self.batch_shape) + 2:
            raise ValueError('basis values must have shape (*B, T, C)')

        if basis_values.shape[:-2] != self.batch_shape:
            raise ValueError(
                f'basis batch must be {self.batch_shape}; got {basis_values.shape[:-2]}'
            )

        if basis_values.shape[-1] != self.num_coefficients:
            raise ValueError(
                f'basis coefficient dimension must be '
                f'{self.num_coefficients}; got {basis_values.shape[-1]}'
            )

        mean = self.coefficients.mean.reshape(
            (
                *self.batch_shape,
                self.variable_dim,
                self.num_coefficients,
            )
        )

        covariance = self.coefficients.covariance.reshape(
            (
                *self.batch_shape,
                self.variable_dim,
                self.num_coefficients,
                self.variable_dim,
                self.num_coefficients,
            )
        )

        temporal_mean = jnp.einsum(
            '...tc,...pc->...tp',
            basis_values,
            mean,
        )

        temporal_covariance = jnp.einsum(
            '...tc,...pcqd,...td->...tpq',
            basis_values,
            covariance,
            basis_values,
        )

        temporal_covariance = 0.5 * (
            temporal_covariance + jnp.swapaxes(temporal_covariance, -1, -2)
        )

        return FullGaussian.from_moments(temporal_mean, temporal_covariance)

    def log_prob(
        self,
        coefficients: jax.Array,
    ) -> jax.Array:
        """Evaluate coefficient-space log density."""
        return self.coefficients.log_prob(coefficients)

    def entropy(
        self,
    ) -> jax.Array:
        """Coefficient-space differential entropy."""
        return self.coefficients.entropy()

    def expected_log_prob(
        self,
        other: typing.Self,
    ) -> jax.Array:
        """Return `E_other[log self]` in coefficient space."""
        if self.basis != other.basis:
            raise ValueError('Hermite Gaussians must use the same basis')

        return self.coefficients.expected_log_prob(other.coefficients)

    def select(
        self,
        index: batch.SelT,
    ) -> typing.Self:
        return batch.select(
            self,
            index,
        )

    def broadcast(
        self,
        shape,
        axis: int = 0,
    ) -> typing.Self:
        return batch.broadcast(
            self,
            shape,
            axis=axis,
        )

    def squeeze(
        self,
        axis=None,
    ) -> typing.Self:
        return batch.squeeze(
            self,
            axis=axis,
        )

    def permute(
        self,
        permutation,
        axis: int = 0,
    ) -> typing.Self:
        return batch.permute(
            self,
            permutation,
            axis=axis,
        )

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        return batch.move_axis(
            self,
            source,
            destination,
        )

    def compute_potential(self) -> GaussianPotential:
        return GaussianPotential.from_moments(self.coefficients)
