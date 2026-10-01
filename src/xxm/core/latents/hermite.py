"""Gaussian coefficient priors and temporal Hermite spline realizations."""

from __future__ import annotations

import dataclasses
import functools
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.data import Sequences
from xxm.core.dists.gaussian import Gaussian
from xxm.core.latents.hermite_core import HermiteBasis
from xxm.core.lengths import LengthsSpec


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=['coefficient_prior'],
    meta_fields=['num_motifs'],
)
@dataclasses.dataclass(frozen=True)
class HermiteSpline:
    r"""Centered Hermite spline independently replicated over batch dimensions `*B`.

    The coefficient prior has batch shape `*B` and event dimension `P * C`,
    where `P` is the number of motifs and `C = J - 1` is the number of free
    centered Hermite coefficients per motif. Coefficients therefore have shape
    `(*B, P * C)` and realized temporal weights have shape `(*B, T, P)`.
    """

    coefficient_prior: Gaussian
    num_motifs: int

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Independent structural batch dimensions `*B`."""
        return self.coefficient_prior.batch_shape

    @property
    def variable_dim(self) -> int:
        """Number `P` of Hermite coordinates."""
        return self.num_motifs

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        """Coefficient-prior dtype."""
        return self.coefficient_prior.dtype

    @property
    def num_coefficients(self) -> int:
        """Number of free centered Hermite coefficients per motif."""
        if self.num_motifs < 1:
            raise ValueError('Hermite spline must contain at least one motif')

        if self.coefficient_prior.variable_dim % self.num_motifs != 0:
            raise ValueError(
                'coefficient prior dimension must be divisible by the number of motifs'
            )

        return self.coefficient_prior.variable_dim // self.num_motifs

    @property
    def num_knots(self) -> int:
        """Number of Hermite knots defining the temporal basis."""
        return self.num_coefficients + 1

    @property
    def basis(self) -> HermiteBasis:
        """Temporal basis implied by the coefficient prior."""
        return HermiteBasis.from_num_knots(self.num_knots)

    @staticmethod
    def _coefficient_prior(
        *,
        num_motifs: int,
        basis: HermiteBasis,
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
        penalties = basis.penalty_matrices(dtype=dtype)

        coefficient_precision = (
            alpha_amp * penalties.amplitude + alpha_smooth * penalties.curvature
        )

        precision = jnp.kron(
            jnp.eye(num_motifs, dtype=dtype),
            coefficient_precision,
        )

        num_coefficients = basis.num_coefficients
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
        basis = HermiteBasis.from_num_knots(num_knots)
        if alpha_amp <= 0.0:
            raise ValueError('alpha_amp must be positive')
        if alpha_smooth < 0.0:
            raise ValueError('alpha_smooth must be non-negative')
        if any(size < 1 for size in batch_shape):
            raise ValueError('batch dimensions must all be positive')

        return cls(
            coefficient_prior=cls._coefficient_prior(
                num_motifs=num_motifs,
                basis=basis,
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
        ls: LengthsSpec,
    ) -> Sequences:

        return self.sample_from_coefficients(
            self.sample_coefficients(key),
            ls,
        )

    def sample_from_coefficients(
        self,
        coefficients: jax.Array,
        ls: LengthsSpec,
    ) -> Sequences:
        """Evaluate coefficients on exactly the timeline described by `ls`."""
        batch.require_same_shape(self.batch_shape, ls.batch_shape)
        expected_shape = self.batch_shape + (self.num_motifs * self.num_coefficients,)
        if coefficients.shape != expected_shape:
            raise ValueError(f'coefficients must have shape {expected_shape}')

        denominator = ls.lengths.shift(-1).clip_min(1).values

        positions = jnp.arange(ls.max_steps, dtype=self.dtype) / denominator[..., None]

        positions = ls.lengths.clear(positions)

        basis = self.basis.at_positions(positions).centered_basis()
        basis = basis.astype(coefficients.dtype)

        coefficients = coefficients.reshape(
            self.batch_shape + (self.num_motifs, self.num_coefficients)
        )

        values = jnp.einsum('...tc,...pc->...tp', basis, coefficients)

        return Sequences.from_padded(values, ls.lengths)

    def log_prob(self, coefficients: jax.Array) -> jax.Array:
        """Evaluate coefficient prior densities over batch `*B`."""
        return self.coefficient_prior.log_prob(coefficients)

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

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at axis."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)
