r"""Gaussian chain inference using canonical form and sequential block elimination.

The canonical potential is

$$\log f(x) = -\frac12 x^\top J x + h^\top x + c,$$

where $J$ is symmetric positive definite. The corresponding normalized Gaussian has mean
$J^{-1}h$ and covariance $J^{-1}$. The log normalizer is $\log \int f(x)\, dx$.
"""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.dists.gaussian import (
    GaussianChainMarginals,
)
from xxm.core.inference import InferredPosterior
from xxm.core.lengths import Lengths
from xxm.core.potentials.gaussian import (
    GaussianChainPotentials,
    GaussianPairPotential,
    GaussianPotential,
)


class GaussianChain(typing.NamedTuple):
    r"""Represents a Gaussian chain with block-tridiagonal structure in canonical form.

    $$\log f(x)=-\frac12x^\top Jx+h^\top x+c.$$

    Here $J$ is symmetric positive definite and block tridiagonal,
    $h$ is the information vector, and $c$ is `log_constant`.

    The lower block convention is $B_t=J_{t+1,t}$, so the upper block is
    $B_t^\top$.

    `information_vectors[t]` contains the block of $h$ associated with $x_t$.

    Leading batch dimensions index independent chains; time and variable
    dimensions are intrinsic.
    """

    diagonal_precision_blocks: jax.Array  # (*B, T, N, N)
    lower_precision_blocks: jax.Array  # (*B, T - 1, N, N)
    information_vectors: jax.Array  # (*B, T, N)
    log_constant: jax.Array  # (*B,)
    lengths: Lengths

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading independent-chain dimensions."""
        return self.diagonal_precision_blocks.shape[:-3]

    def validate(self) -> None:
        """Validate structural lengths at an explicit host boundary."""
        batch.require_same_shape(
            self.batch_shape,
            self.lower_precision_blocks.shape[:-3],
            self.information_vectors.shape[:-2],
            self.log_constant.shape,
            self.lengths.batch_shape,
        )

    def forward_backward(
        self,
    ) -> InferredPosterior[GaussianChainMarginals]:
        """Infer each chain, excluding padded suffixes from its log normalizer."""
        if self.max_steps < 1:
            raise ValueError('Gaussian chain inference requires at least one time step')

        batch_shape = self.batch_shape
        t = self.max_steps
        d = self.variable_dim

        batch.require_same_shape(
            batch_shape,
            self.log_constant.shape,
            self.lengths.batch_shape,
        )

        if (
            self.diagonal_precision_blocks.shape != batch_shape + (t, d, d)
            or self.lower_precision_blocks.shape != batch_shape + (t - 1, d, d)
            or self.information_vectors.shape != batch_shape + (t, d)
        ):
            raise ValueError(
                'Gaussian chain fields must have aligned batch, time, '
                'and variable dimensions'
            )

        if not batch_shape:
            return self._forward_backward_single()

        return batch.vmap_batch(
            GaussianChain._forward_backward_single,
            self,
            batch_shape=self.batch_shape,
        )

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
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

    @classmethod
    def from_potentials(
        cls,
        potentials: GaussianChainPotentials,
    ) -> GaussianChain:
        """Construct chains from potential batches `*B` and `(*B, T - 1)`."""
        return cls.from_pair_potentials(
            initial_potential=potentials.initial,
            pair_potentials=potentials.pairs,
            lengths=potentials.lengths,
        )

    @classmethod
    def from_pair_potentials(
        cls,
        initial_potential: GaussianPotential,  # batch *B
        pair_potentials: GaussianPairPotential,  # batch (*B, T - 1)
        lengths: Lengths,
    ) -> GaussianChain:
        """Construct chains from potential batches `*B` and `(*B, T - 1)`."""

        if pair_potentials.variable_dim != initial_potential.variable_dim:
            raise ValueError(
                'initial_potential and pair_potentials must have the same variable dimension. '
                f'Got {initial_potential.variable_dim} and {pair_potentials.variable_dim}'
            )

        batch_shape = initial_potential.batch_shape

        if (
            len(pair_potentials.batch_shape) != len(batch_shape) + 1
            or pair_potentials.batch_shape[:-1] != batch_shape
        ):
            raise ValueError(
                'pair potentials must have batch shape (*B, T - 1) matching the initial potential'
            )
        max_steps = pair_potentials.batch_shape[-1] + 1
        latent_dim = initial_potential.variable_dim
        dtype = initial_potential.precision_blocks.dtype

        diagonal = jnp.zeros(
            batch_shape + (max_steps, latent_dim, latent_dim),
            dtype=dtype,
        )
        diagonal = diagonal.at[..., 0, :, :].add(initial_potential.precision_blocks)
        diagonal = diagonal.at[..., :-1, :, :].add(pair_potentials.left_precision)
        diagonal = diagonal.at[..., 1:, :, :].add(pair_potentials.right_precision)

        # Make padded variables independent, with unit precision.
        padding = ~lengths.to_mask(max_steps).values

        diagonal = diagonal + (
            padding[..., None, None] * jnp.eye(latent_dim, dtype=dtype)
        )

        information_vectors = jnp.zeros(
            batch_shape + (max_steps, latent_dim),
            dtype=dtype,
        )
        information_vectors = information_vectors.at[..., 0, :].add(
            initial_potential.information_vectors
        )
        information_vectors = information_vectors.at[..., :-1, :].add(
            pair_potentials.left_information
        )
        information_vectors = information_vectors.at[..., 1:, :].add(
            pair_potentials.right_information
        )

        log_constant = initial_potential.log_constant + jnp.sum(
            pair_potentials.log_constant, axis=-1
        )

        return cls(
            diagonal_precision_blocks=diagonal,
            lower_precision_blocks=pair_potentials.lower_precision,
            information_vectors=information_vectors,
            log_constant=log_constant,
            lengths=lengths,
        )

    @property
    def max_steps(self) -> int:
        """Number of time steps $T$."""
        return self.diagonal_precision_blocks.shape[-3]

    @property
    def variable_dim(self) -> int:
        """Dimension of each continuous state."""
        return self.diagonal_precision_blocks.shape[-1]

    def add_local_potential(
        self,
        potential: GaussianPotential,
    ) -> GaussianChain:
        """Add one unary Gaussian potential at each time step."""

        expected_batch_shape = self.batch_shape + (self.max_steps,)

        if potential.batch_shape != expected_batch_shape:
            raise ValueError(
                'potential must contain exactly one potential per time step; '
                f'expected leading shape {expected_batch_shape}, '
                f'got {potential.batch_shape}'
            )

        if potential.variable_dim != self.variable_dim:
            raise ValueError(
                'potential variable dimension must match chain variable dimension; '
                f'expected {self.variable_dim}, got {potential.variable_dim}'
            )

        return GaussianChain(
            lengths=self.lengths,
            diagonal_precision_blocks=(
                self.diagonal_precision_blocks + potential.precision_blocks
            ),
            lower_precision_blocks=self.lower_precision_blocks,
            information_vectors=(
                self.information_vectors + potential.information_vectors
            ),
            log_constant=(self.log_constant + jnp.sum(potential.log_constant, axis=-1)),
        )

    def log_potential(
        self,
        latent: jax.Array,
    ) -> jax.Array:
        r"""Compute $\log f(x)$ for a latent trajectory."""

        expected_shape = self.batch_shape + (self.max_steps, self.variable_dim)

        if latent.shape != expected_shape:
            raise ValueError(
                f'latent must have shape {expected_shape}. Got shape {latent.shape}'
            )

        diagonal_terms = jnp.einsum(
            '...ti,...tij,...tj->...',
            latent,
            self.diagonal_precision_blocks,
            latent,
        )

        cross_terms = jnp.einsum(
            '...ti,...tij,...tj->...',
            latent[..., 1:, :],
            self.lower_precision_blocks,
            latent[..., :-1, :],
        )

        linear_terms = jnp.sum(latent * self.information_vectors, axis=(-2, -1))

        return -0.5 * diagonal_terms - cross_terms + linear_terms + self.log_constant

    def dense_precision(self) -> jax.Array:
        """Construct the dense precision matrix for reference and testing.

        This helper is intentionally dense and should not be used by the main
        Gaussian-chain inference path.
        """

        diagonal_blocks = self.diagonal_precision_blocks
        lower_blocks = self.lower_precision_blocks

        time_steps, variable_dim, _ = diagonal_blocks.shape[-3:]
        dense_size = time_steps * variable_dim

        dense = jnp.zeros(
            self.batch_shape + (dense_size, dense_size),
            dtype=diagonal_blocks.dtype,
        )

        for t in range(time_steps):
            start = t * variable_dim
            stop = (t + 1) * variable_dim

            dense = dense.at[..., start:stop, start:stop].set(
                diagonal_blocks[..., t, :, :]
            )

            if t < time_steps - 1:
                next_start = (t + 1) * variable_dim
                next_stop = (t + 2) * variable_dim
                lower_block = lower_blocks[..., t, :, :]

                dense = dense.at[
                    ...,
                    start:stop,
                    next_start:next_stop,
                ].set(jnp.swapaxes(lower_block, -1, -2))

                dense = dense.at[
                    ...,
                    next_start:next_stop,
                    start:stop,
                ].set(lower_block)

        return dense

    def _forward_elimination(
        self,
    ) -> _GaussianChainFactorization:
        """Perform forward block elimination for a Gaussian chain."""

        initial_carry = _ForwardEliminationCarry.from_canonical(
            self.diagonal_precision_blocks[0],
            self.information_vectors[0],
        )

        def step(
            carry: _ForwardEliminationCarry,
            inputs: tuple[jax.Array, jax.Array, jax.Array],
        ):
            lower_precision, next_precision, next_information = inputs

            result = carry.eliminate_next(
                lower_precision,
                next_precision,
                next_information,
            )

            return result.next_carry, result

        final_carry, results = jax.lax.scan(
            step,
            initial_carry,
            (
                self.lower_precision_blocks,
                self.diagonal_precision_blocks[1:],
                self.information_vectors[1:],
            ),
        )

        precision_cholesky_factors = jnp.concatenate(
            [
                initial_carry.cholesky[None],
                results.next_carry.cholesky,
            ],
        )

        return _GaussianChainFactorization(
            precision_cholesky_factors=precision_cholesky_factors,
            conditional_mean=results.conditional_mean,
            terminal_mean=final_carry.mean_offset,
        )

    def _forward_backward_single(
        self,
    ) -> InferredPosterior[GaussianChainMarginals]:
        """Compute moments and a validity-aware log normalizer."""

        factorization = self._forward_elimination()

        means = factorization.backward_means()

        covariances, cross_covariances = factorization.backward_covariances()

        log_det_terms = jax.vmap(_log_det_from_cholesky)(
            factorization.precision_cholesky_factors
        )

        quadratic_terms = jnp.sum(self.information_vectors * means, axis=-1)

        quadratic_terms = self.lengths.clear(quadratic_terms)
        quadratic_term = jnp.sum(quadratic_terms)

        log_det_terms = self.lengths.clear(log_det_terms)
        log_det_precision = jnp.sum(log_det_terms)

        num_valid = self.lengths.values
        num_valid = num_valid.astype(self.diagonal_precision_blocks.dtype)

        total_dimension = num_valid * self.variable_dim

        log_normalizer = (
            self.log_constant
            + 0.5 * quadratic_term
            - 0.5 * log_det_precision
            + 0.5
            * total_dimension
            * jnp.log(
                jnp.array(
                    2.0 * jnp.pi,
                    dtype=self.diagonal_precision_blocks.dtype,
                )
            )
        )

        posterior = GaussianChainMarginals(
            lengths=self.lengths,
            means=means,
            covariances=covariances,
            cross_covariances=cross_covariances,
        )

        return InferredPosterior(
            posterior=posterior,
            log_normalizer=log_normalizer,
        )


class _ForwardEliminationCarry(typing.NamedTuple):
    r"""Effective precision Cholesky factor and offset $J^{-1}h$ at one step."""

    cholesky: jax.Array
    mean_offset: jax.Array

    @classmethod
    def from_canonical(
        cls,
        precision: jax.Array,
        information: jax.Array,
    ) -> typing.Self:
        """Construct the elimination carry from effective canonical parameters."""
        cholesky = jnp.linalg.cholesky(precision)

        return cls(
            cholesky=cholesky,
            mean_offset=_solve_from_cholesky(
                cholesky,
                information,
            ),
        )

    def eliminate_next(
        self,
        lower_precision: jax.Array,
        next_precision: jax.Array,
        next_information: jax.Array,
    ) -> _ForwardEliminationResult:
        r"""Eliminate $x_t$ and retain its conditional mean as a function of $x_{t+1}$."""
        mean_coefficient = -_solve_from_cholesky(
            self.cholesky,
            lower_precision.T,
        )

        conditional_mean = Affine(
            coefficients=mean_coefficient,
            bias=self.mean_offset,
        )

        effective_precision = next_precision + lower_precision @ mean_coefficient

        effective_information = next_information - lower_precision @ self.mean_offset

        next_carry = self.__class__.from_canonical(
            effective_precision,
            effective_information,
        )

        return _ForwardEliminationResult(
            next_carry=next_carry,
            conditional_mean=conditional_mean,
        )


class _ForwardEliminationResult(typing.NamedTuple):
    """Next elimination carry and the eliminated state's conditional mean map."""

    next_carry: _ForwardEliminationCarry
    conditional_mean: Affine


class _GaussianChainFactorization(typing.NamedTuple):
    """State produced by sequential Gaussian-chain block elimination."""

    precision_cholesky_factors: jax.Array
    conditional_mean: Affine
    terminal_mean: jax.Array  # (D,)

    def backward_means(self) -> jax.Array:
        """Compute marginal means by backward recursion."""

        def step(next_mean, conditional_mean):
            mean = conditional_mean.apply(next_mean)
            return mean, mean

        _, means_rest = jax.lax.scan(
            step,
            self.terminal_mean,
            self.conditional_mean,
            reverse=True,
        )

        return jnp.concatenate(
            [
                means_rest,
                self.terminal_mean[None],
            ],
        )

    def backward_covariances(
        self,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute marginal and adjacent cross-covariances by backward recursion."""
        cholesky_factors = self.precision_cholesky_factors
        coefficients = self.conditional_mean.coefficients

        last_covariance = _inverse_from_cholesky(cholesky_factors[-1])

        def step(next_covariance, inputs):
            cholesky, coefficient = inputs

            conditional_covariance = _inverse_from_cholesky(cholesky)

            covariance = (
                conditional_covariance + coefficient @ next_covariance @ coefficient.T
            )

            cross_covariance = coefficient @ next_covariance

            return covariance, (
                covariance,
                cross_covariance,
            )

        _, (covariances_rest, cross_covariances) = jax.lax.scan(
            step,
            last_covariance,
            (
                cholesky_factors[:-1],
                coefficients,
            ),
            reverse=True,
        )

        covariances = jnp.concatenate(
            [
                covariances_rest,
                last_covariance[None],
            ],
        )

        return covariances, cross_covariances


def _solve_from_cholesky(
    cholesky_factor: jax.Array,
    rhs: jax.Array,
) -> jax.Array:
    """Solve an SPD system from its lower-triangular Cholesky factor."""
    intermediate = jsp_linalg.solve_triangular(
        cholesky_factor,
        rhs,
        lower=True,
    )

    return jsp_linalg.solve_triangular(
        cholesky_factor.T,
        intermediate,
        lower=False,
    )


def _inverse_from_cholesky(
    cholesky_factor: jax.Array,
) -> jax.Array:
    """Compute an SPD inverse from its Cholesky factor."""
    identity = jnp.eye(
        cholesky_factor.shape[0],
        dtype=cholesky_factor.dtype,
    )

    return _solve_from_cholesky(
        cholesky_factor,
        identity,
    )


def _log_det_from_cholesky(
    cholesky_factor: jax.Array,
) -> jax.Array:
    """Compute an SPD log determinant from its Cholesky factor."""
    return 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky_factor)))
