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
        """Adding a collection of unary Gaussian factors to a Gaussian chain"""

        return GaussianChain(
            diagonal_precision_blocks=(self.diagonal_precision_blocks + potential.precision_blocks),
            lower_precision_blocks=self.lower_precision_blocks,
            information_vectors=(self.information_vectors + potential.information_vectors),
            log_constant=(self.log_constant + potential.log_constant),
        )

    def log_potential(
        self,
        latent: jax.Array,
    ) -> jax.Array:
        """Compute log f(x) for a latent trajectory."""

        diagonal_terms = jnp.einsum(
            'ti,tij,tj->',
            latent,
            self.diagonal_precision_blocks,
            latent,
        )

        cross_terms = jnp.einsum(
            'ti,tij,tj->',
            latent[1:],
            self.lower_precision_blocks,
            latent[:-1],
        )

        linear_terms = jnp.sum(latent * self.information_vectors)

        return -0.5 * diagonal_terms - cross_terms + linear_terms + self.log_constant

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

        first_cholesky = jnp.linalg.cholesky(diagonal_blocks[0])

        def step(carry, x):
            prev_cholesky, prev_eff_info = carry
            lower_block, diag_block, info_vec = x

            solved_lower_transpose = _solve_from_cholesky(prev_cholesky, lower_block.T)
            effective_precision = diag_block - lower_block @ solved_lower_transpose
            current_cholesky = jnp.linalg.cholesky(effective_precision)

            solved_prev_info = _solve_from_cholesky(prev_cholesky, prev_eff_info)
            current_eff_info = info_vec - lower_block @ solved_prev_info

            return (current_cholesky, current_eff_info), (current_cholesky, current_eff_info)

        _, (rest_cholesky, rest_eff_info) = jax.lax.scan(
            step,
            (first_cholesky, information_vectors[0]),
            (lower_blocks, diagonal_blocks[1:], information_vectors[1:]),
        )

        precision_cholesky_factors = jnp.concatenate([first_cholesky[None], rest_cholesky])
        effective_information_vectors = jnp.concatenate(
            [information_vectors[0][None], rest_eff_info]
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

        if time_steps == 0:
            return GaussianChainMarginals(
                means=jnp.zeros((time_steps, variable_dim), dtype=dtype),
                covariances=jnp.zeros((time_steps, variable_dim, variable_dim), dtype=dtype),
                cross_covariances=jnp.zeros((0, variable_dim, variable_dim), dtype=dtype),
                log_normalizer=jnp.array(0.0, dtype=dtype),
            )

        # Backward mean recursion.
        last_mean = _solve_from_cholesky(
            precision_cholesky_factors[-1],
            effective_information_vectors[-1],
        )

        def backward_mean_step(carry, x):
            next_mean = carry
            lower_block, cholesky, eff_info = x
            mean = _solve_from_cholesky(cholesky, eff_info - lower_block.T @ next_mean)
            return mean, mean

        _, means_rest = jax.lax.scan(
            backward_mean_step,
            last_mean,
            (lower_blocks, precision_cholesky_factors[:-1], effective_information_vectors[:-1]),
            reverse=True,
        )

        means = jnp.concatenate([means_rest, last_mean[None]])

        # Backward covariance recursion.
        last_covariance = _inverse_from_cholesky(precision_cholesky_factors[-1])

        def backward_cov_step(carry, x):
            next_covariance = carry
            lower_block, cholesky = x

            conditional_covariance = _inverse_from_cholesky(cholesky)
            mean_coefficient = -_solve_from_cholesky(cholesky, lower_block.T)

            covariance = (
                conditional_covariance + mean_coefficient @ next_covariance @ mean_coefficient.T
            )
            cross_covariance = mean_coefficient @ next_covariance

            return covariance, (covariance, cross_covariance)

        _, (covs_rest, cross_covariances) = jax.lax.scan(
            backward_cov_step,
            last_covariance,
            (lower_blocks, precision_cholesky_factors[:-1]),
            reverse=True,
        )

        covariances = jnp.concatenate([covs_rest, last_covariance[None]])

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


class GaussianChainMoments(typing.NamedTuple):
    means: jax.Array
    second_moments: jax.Array
    cross_second_moments: jax.Array


class GaussianChainMarginals(typing.NamedTuple):
    r"""Marginal central moments and log normalizer of a Gaussian chain.

    * ``means[t] = E[x_t]``.
    * ``covariances[t] = Cov(x_t, x_t)``.
    * ``cross_covariances[t] = Cov(x_t, x_{t+1})``.
    * ``log_normalizer = log ∫ f(x) dx``.
    """

    means: jax.Array
    covariances: jax.Array
    cross_covariances: jax.Array
    log_normalizer: jax.Array

    def raw_second_moments(self) -> jax.Array:
        """Return E[x_t x_t.T], shape (T, D, D)."""
        extra = jnp.einsum(
            'ti,tj->tij',
            self.means,
            self.means,
        )

        return self.covariances + extra

    def raw_cross_moments(self) -> jax.Array:
        """Return E[x_t x_{t+1}.T], shape (T - 1, D, D)."""
        extra = jnp.einsum(
            'ti,tj->tij',
            self.means[:-1],
            self.means[1:],
        )

        return self.cross_covariances + extra


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
