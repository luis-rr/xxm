r"""Inference for a block-tridiagonal Gaussian chain.

Uses sequential block elimination to compute local marginal moments and
the log normalizer without forming the full dense precision matrix.

The canonical potential is

    log f(x) = -1/2 x.T @ J @ x + h.T @ x + constant

where J is symmetric positive definite. The corresponding normalized Gaussian
has

    mean = J^{-1} h
    covariance = J^{-1}.

The reported log normalizer is

    log \int f(x) dx.

"""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg


def _precision_and_log_det(
    covariance: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    cholesky = jnp.linalg.cholesky(covariance)

    precision = jsp_linalg.cho_solve(
        (cholesky, True),
        jnp.eye(covariance.shape[0], dtype=covariance.dtype),
    )

    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))

    return precision, log_det


class GaussianPotential(typing.NamedTuple):
    r"""Gaussian potential in canonical form.

    Represents

        log f(x) = -1/2 x.T @ J @ x + h.T @ x + c,

    where ``precision_blocks`` contains J, ``information_vectors``
    contains h, and ``log_constant`` is c.
    """

    precision_blocks: jax.Array  # (..., D, D)
    information_vectors: jax.Array  # (..., D)
    log_constant: jax.Array  # scalar

    @classmethod
    def from_moments(
        cls,
        mean: jax.Array,
        covariance: jax.Array,
    ) -> GaussianPotential:
        precision, log_det_covariance = _precision_and_log_det(covariance)

        information = precision @ mean

        d = mean.shape[-1]

        log_constant = (
            -0.5 * mean @ information - 0.5 * log_det_covariance - 0.5 * d * jnp.log(2.0 * jnp.pi)
        )

        return GaussianPotential(
            precision_blocks=precision,
            information_vectors=information,
            log_constant=log_constant,
        )


class GaussianPairPotential(typing.NamedTuple):
    r"""Pairwise Gaussian potential in canonical form.

    Represents

        log f(x_0, x_1)
        = -1/2 x_0.T @ J_00 @ x_0
          - x_1.T @ J_10 @ x_0
          - 1/2 x_1.T @ J_11 @ x_1
          + h_0.T @ x_0
          + h_1.T @ x_1
          + c.

    The fields contain the corresponding precision, information,
    and constant terms.
    """

    left_precision: jax.Array  # (..., D, D)
    right_precision: jax.Array  # (..., D, D)
    lower_precision: jax.Array  # (..., D, D)
    left_information: jax.Array  # (..., D)
    right_information: jax.Array  # (..., D)
    log_constant: jax.Array  # scalar

    @classmethod
    def from_linear_conditional(
        cls,
        matrix: jax.Array,
        bias: jax.Array,
        covariance: jax.Array,
    ) -> GaussianPairPotential:
        precision, log_det_covariance = _precision_and_log_det(covariance)

        precision_matrix = precision @ matrix
        precision_bias = precision @ bias

        d = bias.shape[-1]

        return GaussianPairPotential(
            left_precision=matrix.T @ precision_matrix,
            right_precision=precision,
            lower_precision=-precision_matrix,
            left_information=-matrix.T @ precision_bias,
            right_information=precision_bias,
            log_constant=(
                -0.5 * bias @ precision_bias
                - 0.5 * log_det_covariance
                - 0.5 * d * jnp.log(2.0 * jnp.pi)
            ),
        )


class GaussianChain(typing.NamedTuple):
    r"""Represents a Gaussian chain with block-tridiagonal structure in canonical form.

    The canonical potential is

        log f(x) = -1/2 x.T @ J @ x + h.T @ x + c

    where J is symmetric positive definite and block tridiagonal,
    h is the information vector, and c is ``log_constant``.


    The block convention is

        diagonal_precision_blocks[t] = J[t, t]
        lower_precision_blocks[t] = J[t + 1, t]

    so that the dense precision has the form

        D0      B0.T
        B0      D1      B1.T
                B1      D2
                        ...

    with ``B_t = lower_precision_blocks[t]``.

    ``information_vectors[t]`` contains the block of ``h`` associated with the variable ``x_t``.
    """

    diagonal_precision_blocks: jax.Array
    lower_precision_blocks: jax.Array
    information_vectors: jax.Array
    log_constant: jax.Array  # scalar

    @property
    def num_time_steps(self) -> int:
        return self.diagonal_precision_blocks.shape[0]

    @property
    def variable_dim(self) -> int:
        return self.diagonal_precision_blocks.shape[1]

    def validate(self) -> None:
        if self.diagonal_precision_blocks.ndim != 3:
            raise ValueError('diagonal_precision_blocks must have shape (T, D, D)')

        if self.lower_precision_blocks.ndim != 3:
            raise ValueError('lower_precision_blocks must have shape (T - 1, D, D)')

        if self.information_vectors.ndim != 2:
            raise ValueError('information_vectors must have shape (T, D)')

        t = self.num_time_steps
        d = self.variable_dim
        d_ = self.diagonal_precision_blocks.shape[2]

        if d != d_:
            raise ValueError('diagonal_precision_blocks must have shape (T, D, D)')

        if t < 1:
            raise ValueError('Chain must contain at least one time step')

        if d < 1:
            raise ValueError('Chain must contain at least one variable dimension')

        if self.information_vectors.shape != (t, d):
            raise ValueError('information_vectors must have shape (T, D)')

        if self.lower_precision_blocks.shape != (max(t - 1, 0), d, d):
            raise ValueError('lower_precision_blocks must have shape (T - 1, D, D)')

    def add_local_potential(
        self,
        potential: GaussianPotential,
    ) -> GaussianChain:
        return GaussianChain(
            diagonal_precision_blocks=(self.diagonal_precision_blocks + potential.precision_blocks),
            lower_precision_blocks=self.lower_precision_blocks,
            information_vectors=(self.information_vectors + potential.information_vectors),
            log_constant=(self.log_constant + potential.log_constant),
        )

    def dense_precision(self) -> jax.Array:
        """Construct the dense precision matrix for reference and testing.

        This helper is intentionally dense and should not be used by the main
        Gaussian-chain inference path.
        """
        self.validate()

        diagonal_blocks = self.diagonal_precision_blocks
        lower_blocks = self.lower_precision_blocks

        time_steps, variable_dim, _ = diagonal_blocks.shape
        dense_size = time_steps * variable_dim

        dense = jnp.zeros(
            (dense_size, dense_size),
            dtype=diagonal_blocks.dtype,
        )

        for t in range(time_steps):
            start = t * variable_dim
            stop = (t + 1) * variable_dim

            dense = dense.at[start:stop, start:stop].set(diagonal_blocks[t])

            if t < time_steps - 1:
                next_start = (t + 1) * variable_dim
                next_stop = (t + 2) * variable_dim
                lower_block = lower_blocks[t]

                dense = dense.at[
                    start:stop,
                    next_start:next_stop,
                ].set(lower_block.T)

                dense = dense.at[
                    next_start:next_stop,
                    start:stop,
                ].set(lower_block)

        return dense

    def _factorize(
        self,
    ) -> _GaussianChainFactorization:
        """Perform sequential block elimination for a Gaussian chain.

        Each effective diagonal precision block is factorized once with Cholesky
        and reused for subsequent solves.
        """
        diagonal_blocks = self.diagonal_precision_blocks
        lower_blocks = self.lower_precision_blocks
        information_vectors = self.information_vectors

        time_steps, _, _ = diagonal_blocks.shape

        precision_cholesky_factors = jnp.zeros_like(diagonal_blocks)
        effective_information_vectors = jnp.zeros_like(information_vectors)

        first_cholesky = jnp.linalg.cholesky(diagonal_blocks[0])

        precision_cholesky_factors = precision_cholesky_factors.at[0].set(first_cholesky)
        effective_information_vectors = effective_information_vectors.at[0].set(
            information_vectors[0]
        )

        for t in range(1, time_steps):
            previous_cholesky = precision_cholesky_factors[t - 1]
            lower_block = lower_blocks[t - 1]

            solved_lower_transpose = _solve_from_cholesky(
                previous_cholesky,
                lower_block.T,
            )

            effective_precision = diagonal_blocks[t] - lower_block @ solved_lower_transpose

            current_cholesky = jnp.linalg.cholesky(effective_precision)
            precision_cholesky_factors = precision_cholesky_factors.at[t].set(current_cholesky)

            previous_information_vector = effective_information_vectors[t - 1]
            solved_previous_information_vector = _solve_from_cholesky(
                previous_cholesky,
                previous_information_vector,
            )

            current_information_vector = (
                information_vectors[t] - lower_block @ solved_previous_information_vector
            )

            effective_information_vectors = effective_information_vectors.at[t].set(
                current_information_vector
            )

        return _GaussianChainFactorization(
            precision_cholesky_factors=precision_cholesky_factors,
            lower_precision_blocks=lower_blocks,
            effective_information_vectors=effective_information_vectors,
        )

    def forward_backward(
        self,
    ) -> GaussianChainMarginals:
        """Compute moments and log normalizer for a Gaussian chain."""

        self.validate()

        factorization = self._factorize()

        precision_cholesky_factors = factorization.precision_cholesky_factors
        lower_blocks = factorization.lower_precision_blocks
        effective_information_vectors = factorization.effective_information_vectors

        time_steps, variable_dim, _ = self.diagonal_precision_blocks.shape

        dtype = self.diagonal_precision_blocks.dtype

        means = jnp.zeros(
            (time_steps, variable_dim),
            dtype=dtype,
        )
        covariances = jnp.zeros(
            (time_steps, variable_dim, variable_dim),
            dtype=dtype,
        )
        cross_covariances = jnp.zeros(
            (max(time_steps - 1, 0), variable_dim, variable_dim),
            dtype=dtype,
        )

        if time_steps == 0:
            return GaussianChainMarginals(
                means=means,
                covariances=covariances,
                cross_covariances=cross_covariances,
                log_normalizer=jnp.array(0.0, dtype=dtype),
            )

        # Backward mean recursion.
        means = means.at[-1].set(
            _solve_from_cholesky(
                precision_cholesky_factors[-1],
                effective_information_vectors[-1],
            )
        )

        for t in range(time_steps - 2, -1, -1):
            lower_block = lower_blocks[t]

            rhs = effective_information_vectors[t] - lower_block.T @ means[t + 1]

            means = means.at[t].set(
                _solve_from_cholesky(
                    precision_cholesky_factors[t],
                    rhs,
                )
            )

        # Backward covariance recursion.
        covariances = covariances.at[-1].set(_inverse_from_cholesky(precision_cholesky_factors[-1]))

        for t in range(time_steps - 2, -1, -1):
            lower_block = lower_blocks[t]
            next_covariance = covariances[t + 1]

            conditional_covariance = _inverse_from_cholesky(precision_cholesky_factors[t])

            mean_coefficient = -_solve_from_cholesky(
                precision_cholesky_factors[t],
                lower_block.T,
            )

            covariance = (
                conditional_covariance + mean_coefficient @ next_covariance @ mean_coefficient.T
            )

            cross_covariance = mean_coefficient @ next_covariance

            covariances = covariances.at[t].set(covariance)
            cross_covariances = cross_covariances.at[t].set(cross_covariance)

        # det(J) is the product of the determinants of the effective
        # precision blocks produced during sequential elimination.
        log_det_precision = jnp.sum(jax.vmap(_log_det_from_cholesky)(precision_cholesky_factors))

        quadratic_term = jnp.sum(self.information_vectors * means)

        total_dimension = time_steps * variable_dim

        log_normalizer = (
            self.log_constant
            + 0.5 * quadratic_term
            - 0.5 * log_det_precision
            + 0.5 * total_dimension * jnp.log(jnp.array(2.0 * jnp.pi, dtype=dtype))
        )

        return GaussianChainMarginals(
            means=means,
            covariances=covariances,
            cross_covariances=cross_covariances,
            log_normalizer=log_normalizer,
        )


class _GaussianChainFactorization(typing.NamedTuple):
    """State produced by sequential Gaussian-chain block elimination."""

    precision_cholesky_factors: jax.Array
    lower_precision_blocks: jax.Array
    effective_information_vectors: jax.Array


class GaussianChainMarginals(typing.NamedTuple):
    r"""Marginal moments and log normalizer of a Gaussian chain.

    * ``means[t] = E[x_t]``.
    * ``covariances[t] = Cov(x_t, x_t)``.
    * ``cross_covariances[t] = Cov(x_t, x_{t+1})``.
    * ``log_normalizer = log ∫ f(x) dx``.
    """

    means: jax.Array
    covariances: jax.Array
    cross_covariances: jax.Array
    log_normalizer: jax.Array


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


def _inverse_from_cholesky(cholesky_factor: jax.Array) -> jax.Array:
    """Compute an SPD inverse from its Cholesky factor."""
    identity = jnp.eye(
        cholesky_factor.shape[0],
        dtype=cholesky_factor.dtype,
    )
    return _solve_from_cholesky(cholesky_factor, identity)


def _log_det_from_cholesky(cholesky_factor: jax.Array) -> jax.Array:
    """Compute an SPD log determinant from its Cholesky factor."""
    return 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky_factor)))
