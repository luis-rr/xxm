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

from xxm.stats.gaussian import Gaussian, LinearGaussian


def _precision_and_log_det(
    covariance: jax.Array,  # (..., N, N)
) -> tuple[jax.Array, jax.Array]:
    """Compute precision matrices and covariance log determinants."""

    if covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError('covariance must have shape (..., N, N)')

    variable_dim = covariance.shape[-1]

    cholesky = jnp.linalg.cholesky(covariance)

    identity = jnp.broadcast_to(
        jnp.eye(variable_dim, dtype=covariance.dtype),
        covariance.shape,
    )

    precision = jsp_linalg.cho_solve(
        (cholesky, True),
        identity,
    )

    log_det = 2.0 * jnp.sum(
        jnp.log(jnp.diagonal(cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )

    return precision, log_det


class GaussianPotential(typing.NamedTuple):
    r"""Gaussian potential in canonical form.

    Represents

        log f(x) = -1/2 x.T @ J @ x + h.T @ x + c,

    where ``precision_blocks`` contains J, ``information_vectors``
    contains h, and ``log_constant`` is c.

    Leading dimensions are treated as batch dimensions for independent potentials.
    """

    precision_blocks: jax.Array  # (..., N, N)
    information_vectors: jax.Array  # (..., N)
    log_constant: jax.Array  # (...)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        precision_shape = self.precision_blocks.shape[:-2]
        information_shape = self.information_vectors.shape[:-1]
        log_constant_shape = self.log_constant.shape
        assert precision_shape == information_shape == log_constant_shape
        return precision_shape

    @property
    def variable_dim(self) -> int:
        return self.precision_blocks.shape[-1]

    def validate(self) -> None:
        if self.precision_blocks.ndim < 2:
            raise ValueError('precision_blocks must have shape (..., N, N)')

        if self.precision_blocks.shape[-2] != self.precision_blocks.shape[-1]:
            raise ValueError('precision_blocks must have shape (..., N, N)')

        if self.variable_dim < 1:
            raise ValueError('Potential must contain at least one variable dimension')

        if self.information_vectors.shape != self.batch_shape + (self.variable_dim,):
            raise ValueError(
                'information_vectors must have shape (..., N) matching precision_blocks'
            )

        if self.log_constant.shape != self.batch_shape:
            raise ValueError('log_constant must have the same leading shape as precision_blocks')

    @classmethod
    def from_moments(
        cls,
        gaussian: Gaussian,
    ) -> GaussianPotential:
        if gaussian.mean.ndim < 1:
            raise ValueError('mean must have shape (..., N)')

        if gaussian.covariance.ndim < 2:
            raise ValueError('covariance must have shape (..., N, N)')

        d = gaussian.mean.shape[-1]
        batch_shape = gaussian.mean.shape[:-1]

        if d < 1:
            raise ValueError('mean must contain at least one variable dimension')

        if gaussian.covariance.shape != batch_shape + (d, d):
            raise ValueError(
                'mean and covariance must have shapes (..., N) and (..., N, N) '
                'with matching leading dimensions'
            )

        precision, log_det_covariance = _precision_and_log_det(gaussian.covariance)

        information = jnp.einsum(
            '...ij,...j->...i',
            precision,
            gaussian.mean,
        )

        quadratic = jnp.sum(gaussian.mean * information, axis=-1)

        return cls(
            precision_blocks=precision,
            information_vectors=information,
            log_constant=(
                -0.5 * quadratic - 0.5 * log_det_covariance - 0.5 * d * jnp.log(2.0 * jnp.pi)
            ),
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

    Leading dimensions are treated as batch dimensions for independent potentials.
    """

    left_precision: jax.Array  # (..., N, N)
    right_precision: jax.Array  # (..., N, N)
    lower_precision: jax.Array  # (..., N, N)
    left_information: jax.Array  # (..., N)
    right_information: jax.Array  # (..., N)
    log_constant: jax.Array  # (...)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        left_precision_shape = self.left_precision.shape[:-2]
        right_precision_shape = self.right_precision.shape[:-2]
        lower_precision_shape = self.lower_precision.shape[:-2]
        left_information_shape = self.left_information.shape[:-1]
        right_information_shape = self.right_information.shape[:-1]
        log_constant_shape = self.log_constant.shape
        assert (
            left_precision_shape
            == right_precision_shape
            == lower_precision_shape
            == left_information_shape
            == right_information_shape
            == log_constant_shape
        )
        return left_precision_shape

    @property
    def variable_dim(self) -> int:
        return self.left_precision.shape[-1]

    def validate(self) -> None:
        if self.left_precision.ndim < 2:
            raise ValueError('left_precision must have shape (..., N, N)')

        if self.left_precision.shape[-2] != self.left_precision.shape[-1]:
            raise ValueError('left_precision must have shape (..., N, N)')

        if self.variable_dim < 1:
            raise ValueError('Potential must contain at least one variable dimension')

        matrix_shape = self.batch_shape + (
            self.variable_dim,
            self.variable_dim,
        )
        vector_shape = self.batch_shape + (self.variable_dim,)

        if self.right_precision.shape != matrix_shape:
            raise ValueError('right_precision must have shape (..., N, N) matching left_precision')

        if self.lower_precision.shape != matrix_shape:
            raise ValueError('lower_precision must have shape (..., N, N) matching left_precision')

        if self.left_information.shape != vector_shape:
            raise ValueError('left_information must have shape (..., N) matching left_precision')

        if self.right_information.shape != vector_shape:
            raise ValueError('right_information must have shape (..., N) matching left_precision')

        if self.log_constant.shape != self.batch_shape:
            raise ValueError('log_constant must have the same leading shape as left_precision')

    @classmethod
    def from_linear_conditional(
        cls,
        lin_gaussian: LinearGaussian,
    ) -> GaussianPairPotential:
        if lin_gaussian.affine.coefficients.ndim < 2:
            raise ValueError('matrix must have shape (..., N, N)')

        if lin_gaussian.affine.coefficients.shape[-2] != lin_gaussian.affine.coefficients.shape[-1]:
            raise ValueError('matrix must have shape (..., N, N)')

        n = lin_gaussian.affine.coefficients.shape[-1]
        batch_shape = lin_gaussian.affine.coefficients.shape[:-2]

        if n < 1:
            raise ValueError('matrix must contain at least one variable dimension')

        if lin_gaussian.affine.bias.shape != batch_shape + (n,):
            raise ValueError(
                'matrix and bias must have shapes (..., N, N) and (..., N) '
                'with matching leading dimensions'
            )

        if lin_gaussian.covariance.shape != batch_shape + (n, n):
            raise ValueError(
                'matrix and covariance must both have shape (..., N, N) '
                'with matching leading dimensions'
            )

        precision, log_det_covariance = _precision_and_log_det(lin_gaussian.covariance)

        matrix_t = jnp.swapaxes(lin_gaussian.affine.coefficients, -1, -2)

        precision_matrix = precision @ lin_gaussian.affine.coefficients

        precision_bias = jnp.einsum(
            '...ij,...j->...i',
            precision,
            lin_gaussian.affine.bias,
        )

        return cls(
            left_precision=matrix_t @ precision_matrix,
            right_precision=precision,
            lower_precision=-precision_matrix,
            left_information=-jnp.einsum(
                '...ij,...j->...i',
                matrix_t,
                precision_bias,
            ),
            right_information=precision_bias,
            log_constant=(
                -0.5 * jnp.sum(lin_gaussian.affine.bias * precision_bias, axis=-1)
                - 0.5 * log_det_covariance
                - 0.5 * n * jnp.log(2.0 * jnp.pi)
            ),
        )

    def broadcast(self, batch_shape) -> GaussianPairPotential:
        return GaussianPairPotential(
            left_precision=jnp.broadcast_to(
                self.left_precision,
                batch_shape + self.left_precision.shape,
            ),
            right_precision=jnp.broadcast_to(
                self.right_precision,
                batch_shape + self.right_precision.shape,
            ),
            lower_precision=jnp.broadcast_to(
                self.lower_precision,
                batch_shape + self.lower_precision.shape,
            ),
            left_information=jnp.broadcast_to(
                self.left_information,
                batch_shape + self.left_information.shape,
            ),
            right_information=jnp.broadcast_to(
                self.right_information,
                batch_shape + self.right_information.shape,
            ),
            log_constant=jnp.broadcast_to(
                self.log_constant,
                batch_shape + self.log_constant.shape,
            ),
        )

    def expected(
        self,
        weights: jax.Array,  # (..., K)
    ) -> GaussianPairPotential:
        """Average a batch of potentials using weights over the last axis."""

        if self.batch_shape != (weights.shape[-1],):
            raise ValueError(
                'last dimension of weights must match the potential batch dimension; '
                f'expected {self.batch_shape}, got {weights.shape}'
            )

        return GaussianPairPotential(
            left_precision=jnp.einsum(
                '...k,kij->...ij',
                weights,
                self.left_precision,
            ),
            right_precision=jnp.einsum(
                '...k,kij->...ij',
                weights,
                self.right_precision,
            ),
            lower_precision=jnp.einsum(
                '...k,kij->...ij',
                weights,
                self.lower_precision,
            ),
            left_information=jnp.einsum(
                '...k,ki->...i',
                weights,
                self.left_information,
            ),
            right_information=jnp.einsum(
                '...k,ki->...i',
                weights,
                self.right_information,
            ),
            log_constant=jnp.einsum(
                '...k,k->...',
                weights,
                self.log_constant,
            ),
        )

    def expected_log_potentials(
        self,
        posterior: GaussianChainMarginals,
    ) -> jax.Array:
        """Return E_q(x)[log p(x[t+1] | x[t], z[t]=k)], shape (T-1, K)."""

        means = posterior.means
        second = posterior.raw_second_moments()
        cross = posterior.raw_cross_moments()

        return (
            -0.5
            * jnp.einsum(
                'kij,tij->tk',
                self.left_precision,
                second[:-1],
            )
            - jnp.einsum(
                'kij,tji->tk',
                self.lower_precision,
                cross,
            )
            - 0.5
            * jnp.einsum(
                'kij,tij->tk',
                self.right_precision,
                second[1:],
            )
            + jnp.einsum(
                'ki,ti->tk',
                self.left_information,
                means[:-1],
            )
            + jnp.einsum(
                'ki,ti->tk',
                self.right_information,
                means[1:],
            )
            + self.log_constant[None, :]
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

    ``information_vectors[t]`` contains the block of ``h`` associated
    with the variable ``x_t``.

    A ``GaussianChain`` represents a single chain. Batch dimensions are
    intentionally not supported; use ``jax.vmap`` over chains instead.
    """

    diagonal_precision_blocks: jax.Array  # (T, N, N)
    lower_precision_blocks: jax.Array  # (T - 1, N, N)
    information_vectors: jax.Array  # (T, N)
    log_constant: jax.Array  # scalar

    @classmethod
    def from_pair_potentials(
        cls,
        initial_potential: GaussianPotential,  # (N, N)
        pair_potentials: GaussianPairPotential,  # (T-1, N, N)
    ) -> GaussianChain:
        """Construct a Gaussian chain from initial and time-indexed pair potentials."""

        num_time_steps = pair_potentials.left_precision.shape[0] + 1
        state_dim = initial_potential.precision_blocks.shape[0]
        dtype = initial_potential.precision_blocks.dtype

        diagonal = jnp.zeros(
            (num_time_steps, state_dim, state_dim),
            dtype=dtype,
        )
        diagonal = diagonal.at[0].add(initial_potential.precision_blocks)
        diagonal = diagonal.at[:-1].add(pair_potentials.left_precision)
        diagonal = diagonal.at[1:].add(pair_potentials.right_precision)

        information_vectors = jnp.zeros(
            (num_time_steps, state_dim),
            dtype=dtype,
        )
        information_vectors = information_vectors.at[0].add(initial_potential.information_vectors)
        information_vectors = information_vectors.at[:-1].add(pair_potentials.left_information)
        information_vectors = information_vectors.at[1:].add(pair_potentials.right_information)

        log_constant = initial_potential.log_constant + jnp.sum(pair_potentials.log_constant)

        return cls(
            diagonal_precision_blocks=diagonal,
            lower_precision_blocks=pair_potentials.lower_precision,
            information_vectors=information_vectors,
            log_constant=log_constant,
        )

    @property
    def num_time_steps(self) -> int:
        return self.diagonal_precision_blocks.shape[0]

    @property
    def variable_dim(self) -> int:
        return self.diagonal_precision_blocks.shape[1]

    def validate(self) -> None:
        if self.diagonal_precision_blocks.ndim != 3:
            raise ValueError('diagonal_precision_blocks must have shape (T, N, N)')

        if self.lower_precision_blocks.ndim != 3:
            raise ValueError('lower_precision_blocks must have shape (T - 1, N, N)')

        if self.information_vectors.ndim != 2:
            raise ValueError('information_vectors must have shape (T, N)')

        if self.log_constant.ndim != 0:
            raise ValueError('log_constant must be scalar')

        t = self.num_time_steps
        d = self.variable_dim

        if self.diagonal_precision_blocks.shape[2] != d:
            raise ValueError('diagonal_precision_blocks must have shape (T, N, N)')

        if t < 1:
            raise ValueError('Chain must contain at least one time step')

        if d < 1:
            raise ValueError('Chain must contain at least one variable dimension')

        if self.information_vectors.shape != (t, d):
            raise ValueError('information_vectors must have shape (T, N)')

        if self.lower_precision_blocks.shape != (t - 1, d, d):
            raise ValueError('lower_precision_blocks must have shape (T - 1, N, N)')

    def add_local_potential(
        self,
        potential: GaussianPotential,
    ) -> GaussianChain:
        """Add one unary Gaussian potential at each time step."""
        self.validate()
        potential.validate()

        expected_batch_shape = (self.num_time_steps,)

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
            diagonal_precision_blocks=(self.diagonal_precision_blocks + potential.precision_blocks),
            lower_precision_blocks=self.lower_precision_blocks,
            information_vectors=(self.information_vectors + potential.information_vectors),
            log_constant=(self.log_constant + jnp.sum(potential.log_constant)),
        )

    def log_potential(
        self,
        latent: jax.Array,
    ) -> jax.Array:
        """Compute log f(x) for a latent trajectory."""
        self.validate()

        expected_shape = (self.num_time_steps, self.variable_dim)

        if latent.shape != expected_shape:
            raise ValueError(f'latent must have shape {expected_shape}. Got shape {latent.shape}')

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

            solved_lower_transpose = _solve_from_cholesky(
                prev_cholesky,
                lower_block.T,
            )
            effective_precision = diag_block - lower_block @ solved_lower_transpose
            current_cholesky = jnp.linalg.cholesky(effective_precision)

            solved_prev_info = _solve_from_cholesky(
                prev_cholesky,
                prev_eff_info,
            )
            current_eff_info = info_vec - lower_block @ solved_prev_info

            return (
                current_cholesky,
                current_eff_info,
            ), (
                current_cholesky,
                current_eff_info,
            )

        _, (rest_cholesky, rest_eff_info) = jax.lax.scan(
            step,
            (first_cholesky, information_vectors[0]),
            (
                lower_blocks,
                diagonal_blocks[1:],
                information_vectors[1:],
            ),
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

        # Backward mean recursion.
        last_mean = _solve_from_cholesky(
            precision_cholesky_factors[-1],
            effective_information_vectors[-1],
        )

        def backward_mean_step(carry, x):
            next_mean = carry
            lower_block, cholesky, eff_info = x

            mean = _solve_from_cholesky(
                cholesky,
                eff_info - lower_block.T @ next_mean,
            )

            return mean, mean

        _, means_rest = jax.lax.scan(
            backward_mean_step,
            last_mean,
            (
                lower_blocks,
                precision_cholesky_factors[:-1],
                effective_information_vectors[:-1],
            ),
            reverse=True,
        )

        means = jnp.concatenate([means_rest, last_mean[None]])

        # Backward covariance recursion.
        last_covariance = _inverse_from_cholesky(precision_cholesky_factors[-1])

        def backward_cov_step(carry, x):
            next_covariance = carry
            lower_block, cholesky = x

            conditional_covariance = _inverse_from_cholesky(cholesky)
            mean_coefficient = -_solve_from_cholesky(
                cholesky,
                lower_block.T,
            )

            covariance = (
                conditional_covariance + mean_coefficient @ next_covariance @ mean_coefficient.T
            )
            cross_covariance = mean_coefficient @ next_covariance

            return covariance, (
                covariance,
                cross_covariance,
            )

        _, (covs_rest, cross_covariances) = jax.lax.scan(
            backward_cov_step,
            last_covariance,
            (
                lower_blocks,
                precision_cholesky_factors[:-1],
            ),
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
        """Return E[x_t x_t.T], shape (T, N, N)."""
        extra = jnp.einsum(
            'ti,tj->tij',
            self.means,
            self.means,
        )

        return self.covariances + extra

    def raw_cross_moments(self) -> jax.Array:
        """Return E[x_t x_{t+1}.T], shape (T - 1, N, N)."""
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
