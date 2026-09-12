"""Thin time-major batched wrappers for Gaussian chain inference."""

from __future__ import annotations

import math
import typing

import jax
import jax.numpy as jnp

from xxm.core.chains.gaussian import (
    GaussianChain,
    GaussianChainMarginals,
    GaussianPairPotential,
    GaussianPotential,
)
from xxm.core.dists.gaussian import Gaussian, PairedGaussian


def _flatten_batch(
    values: jax.Array,
    batch_shape: tuple[int, ...],
) -> jax.Array:
    """Flatten leading structural batch dimensions into one axis."""
    batch_ndim = len(batch_shape)
    batch_size = math.prod(batch_shape)

    return values.reshape((batch_size,) + values.shape[batch_ndim:])


def _unflatten_batch(
    values: jax.Array,
    batch_shape: tuple[int, ...],
) -> jax.Array:
    """Restore flattened leading structural batch dimensions."""
    return values.reshape(batch_shape + values.shape[1:])


def _flatten_time_batch(
    values: jax.Array,
    batch_shape: tuple[int, ...],
) -> jax.Array:
    """Convert `(T, *B, ...)` to `(prod(B), T, ...)`."""
    batch_ndim = len(batch_shape)
    batch_size = math.prod(batch_shape)

    values = jnp.moveaxis(
        values,
        0,
        batch_ndim,
    )

    return values.reshape((batch_size,) + values.shape[batch_ndim:])


def _unflatten_time_batch(
    values: jax.Array,
    batch_shape: tuple[int, ...],
) -> jax.Array:
    """Convert `(prod(B), T, ...)` to `(T, *B, ...)`."""
    batch_ndim = len(batch_shape)

    values = values.reshape(batch_shape + values.shape[1:])

    return jnp.moveaxis(
        values,
        batch_ndim,
        0,
    )


class GaussianChainBatch(typing.NamedTuple):
    r"""Batch of independent Gaussian chains.

    External arrays are time-major:

    - `diagonal_precision_blocks`: `(T, *B, D, D)`
    - `lower_precision_blocks`: `(T - 1, *B, D, D)`
    - `information_vectors`: `(T, *B, D)`
    - `log_constant`: `(*B,)`

    Structural batch dimensions `*B` index independent Gaussian chains.
    Internally these dimensions are flattened and inference is delegated to
    the single-chain `GaussianChain` implementation with `jax.vmap`.
    """

    diagonal_precision_blocks: jax.Array
    lower_precision_blocks: jax.Array
    information_vectors: jax.Array
    log_constant: jax.Array

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Structural batch shape."""
        diagonal_shape = self.diagonal_precision_blocks.shape[1:-2]
        lower_shape = self.lower_precision_blocks.shape[1:-2]
        information_shape = self.information_vectors.shape[1:-1]
        log_constant_shape = self.log_constant.shape

        assert diagonal_shape == lower_shape == information_shape == log_constant_shape

        return diagonal_shape

    @property
    def num_steps(self) -> int:
        """Number of time steps $T$."""
        return self.diagonal_precision_blocks.shape[0]

    @property
    def variable_dim(self) -> int:
        """Dimension of each Gaussian variable."""
        return self.diagonal_precision_blocks.shape[-1]

    @classmethod
    def from_pair_potentials(
        cls,
        initial_potential: GaussianPotential,
        pair_potentials: GaussianPairPotential,
    ) -> typing.Self:
        """Construct batched chains from time-major Gaussian potentials.

        `initial_potential` has batch shape `(*B,)`.

        `pair_potentials` has batch shape `(T - 1, *B)`.
        """
        batch_shape = initial_potential.batch_shape

        if not batch_shape:
            raise ValueError('initial_potential must have at least one batch dimension')

        expected_pair_ndim = len(batch_shape) + 1

        if (
            len(pair_potentials.batch_shape) != expected_pair_ndim
            or pair_potentials.batch_shape[1:] != batch_shape
        ):
            raise ValueError(
                'pair_potentials must have batch shape '
                f'(T - 1, *{batch_shape}); '
                f'got {pair_potentials.batch_shape}'
            )

        if pair_potentials.variable_dim != initial_potential.variable_dim:
            raise ValueError(
                'initial and pair potentials must have the same variable dimension'
            )

        flat_initial = GaussianPotential(
            precision_blocks=_flatten_batch(
                initial_potential.precision_blocks,
                batch_shape,
            ),
            information_vectors=_flatten_batch(
                initial_potential.information_vectors,
                batch_shape,
            ),
            log_constant=_flatten_batch(
                initial_potential.log_constant,
                batch_shape,
            ),
        )

        flat_pairs = GaussianPairPotential(
            left_precision=_flatten_time_batch(
                pair_potentials.left_precision,
                batch_shape,
            ),
            right_precision=_flatten_time_batch(
                pair_potentials.right_precision,
                batch_shape,
            ),
            lower_precision=_flatten_time_batch(
                pair_potentials.lower_precision,
                batch_shape,
            ),
            left_information=_flatten_time_batch(
                pair_potentials.left_information,
                batch_shape,
            ),
            right_information=_flatten_time_batch(
                pair_potentials.right_information,
                batch_shape,
            ),
            log_constant=_flatten_time_batch(
                pair_potentials.log_constant,
                batch_shape,
            ),
        )

        flat_chains = jax.vmap(
            GaussianChain.from_pair_potentials,
        )(
            flat_initial,
            flat_pairs,
        )

        return cls._from_flat(
            flat_chains,
            batch_shape,
        )

    @classmethod
    def _from_flat(
        cls,
        chain: GaussianChain,
        batch_shape: tuple[int, ...],
    ) -> typing.Self:
        """Restore structural batch dimensions from flattened chains."""
        return cls(
            diagonal_precision_blocks=_unflatten_time_batch(
                chain.diagonal_precision_blocks,
                batch_shape,
            ),
            lower_precision_blocks=_unflatten_time_batch(
                chain.lower_precision_blocks,
                batch_shape,
            ),
            information_vectors=_unflatten_time_batch(
                chain.information_vectors,
                batch_shape,
            ),
            log_constant=_unflatten_batch(
                chain.log_constant,
                batch_shape,
            ),
        )

    def _flatten(self) -> GaussianChain:
        """Flatten structural batch dimensions for `jax.vmap`."""
        batch_shape = self.batch_shape

        if not batch_shape:
            raise ValueError('GaussianChainBatch requires at least one batch dimension')

        return GaussianChain(
            diagonal_precision_blocks=_flatten_time_batch(
                self.diagonal_precision_blocks,
                batch_shape,
            ),
            lower_precision_blocks=_flatten_time_batch(
                self.lower_precision_blocks,
                batch_shape,
            ),
            information_vectors=_flatten_time_batch(
                self.information_vectors,
                batch_shape,
            ),
            log_constant=_flatten_batch(
                self.log_constant,
                batch_shape,
            ),
        )

    def select(
        self,
        index,
    ) -> typing.Self:
        """Index into structural batch dimensions.

        Selection is closed over `GaussianChainBatch`: at least one structural
        batch dimension must remain after indexing.
        """
        if not self.batch_shape:
            raise ValueError('cannot select from an unbatched GaussianChainBatch')

        if not isinstance(index, tuple):
            index = (index,)

        time_index = (slice(None),) + index

        selected = self.__class__(
            diagonal_precision_blocks=self.diagonal_precision_blocks[time_index],
            lower_precision_blocks=self.lower_precision_blocks[time_index],
            information_vectors=self.information_vectors[time_index],
            log_constant=self.log_constant[index],
        )

        if not selected.batch_shape:
            # TODO: Add `get()` returning `GaussianChain` if explicit scalar
            # batch extraction becomes useful.
            raise ValueError(
                'select must leave at least one structural batch dimension'
            )

        return selected

    def add_local_potential(
        self,
        potential: GaussianPotential,
    ) -> typing.Self:
        """Add one time-major unary potential to every chain.

        `potential` must have batch shape `(T, *B)`.
        """
        expected_batch_shape = (
            self.num_steps,
            *self.batch_shape,
        )

        if potential.batch_shape != expected_batch_shape:
            raise ValueError(
                'potential must have batch shape '
                f'{expected_batch_shape}; '
                f'got {potential.batch_shape}'
            )

        if potential.variable_dim != self.variable_dim:
            raise ValueError(
                'potential variable dimension must match chain variable '
                f'dimension; expected {self.variable_dim}, '
                f'got {potential.variable_dim}'
            )

        batch_shape = self.batch_shape

        flat_potential = GaussianPotential(
            precision_blocks=_flatten_time_batch(
                potential.precision_blocks,
                batch_shape,
            ),
            information_vectors=_flatten_time_batch(
                potential.information_vectors,
                batch_shape,
            ),
            log_constant=_flatten_time_batch(
                potential.log_constant,
                batch_shape,
            ),
        )

        flat_chain = jax.vmap(
            lambda chain, local: chain.add_local_potential(local),
        )(
            self._flatten(),
            flat_potential,
        )

        return self._from_flat(
            flat_chain,
            batch_shape,
        )

    def forward_backward(
        self,
    ) -> tuple[GaussianChainMarginalsBatch, jax.Array]:
        """Infer all chains independently with the single-chain solver."""
        batch_shape = self.batch_shape

        posterior, log_normalizer = jax.vmap(
            lambda chain: chain.forward_backward(),
        )(
            self._flatten(),
        )

        return (
            GaussianChainMarginalsBatch._from_flat(
                posterior,
                batch_shape,
            ),
            _unflatten_batch(
                log_normalizer,
                batch_shape,
            ),
        )


class GaussianChainMarginalsBatch(typing.NamedTuple):
    r"""Marginals for a batch of independent Gaussian chains.

    External arrays are time-major:

    - `means`: `(T, *B, D)`
    - `covariances`: `(T, *B, D, D)`
    - `cross_covariances`: `(T - 1, *B, D, D)`

    Structural batch dimensions `*B` index independent Gaussian chains.
    """

    means: jax.Array
    covariances: jax.Array
    cross_covariances: jax.Array

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Structural batch shape."""
        mean_shape = self.means.shape[1:-1]
        covariance_shape = self.covariances.shape[1:-2]
        cross_shape = self.cross_covariances.shape[1:-2]

        assert mean_shape == covariance_shape == cross_shape

        return mean_shape

    @property
    def num_steps(self) -> int:
        """Number of time steps $T$."""
        return self.means.shape[0]

    @property
    def variable_dim(self) -> int:
        """Dimension of each Gaussian variable."""
        return self.means.shape[-1]

    @classmethod
    def _from_flat(
        cls,
        posterior: GaussianChainMarginals,
        batch_shape: tuple[int, ...],
    ) -> typing.Self:
        """Restore structural batch dimensions from flattened marginals."""
        return cls(
            means=_unflatten_time_batch(
                posterior.means,
                batch_shape,
            ),
            covariances=_unflatten_time_batch(
                posterior.covariances,
                batch_shape,
            ),
            cross_covariances=_unflatten_time_batch(
                posterior.cross_covariances,
                batch_shape,
            ),
        )

    def _flatten(self) -> GaussianChainMarginals:
        """Flatten structural batch dimensions for `jax.vmap`."""
        batch_shape = self.batch_shape

        if not batch_shape:
            raise ValueError(
                'GaussianChainMarginalsBatch requires at least one batch dimension'
            )

        return GaussianChainMarginals(
            means=_flatten_time_batch(
                self.means,
                batch_shape,
            ),
            covariances=_flatten_time_batch(
                self.covariances,
                batch_shape,
            ),
            cross_covariances=_flatten_time_batch(
                self.cross_covariances,
                batch_shape,
            ),
        )

    def select(
        self,
        index,
    ) -> typing.Self:
        """Index into structural batch dimensions.

        Selection is closed over `GaussianChainMarginalsBatch`: at least one
        structural batch dimension must remain after indexing.
        """
        if not self.batch_shape:
            raise ValueError(
                'cannot select from an unbatched GaussianChainMarginalsBatch'
            )

        if not isinstance(index, tuple):
            index = (index,)

        time_index = (slice(None),) + index

        selected = self.__class__(
            means=self.means[time_index],
            covariances=self.covariances[time_index],
            cross_covariances=self.cross_covariances[time_index],
        )

        if not selected.batch_shape:
            # TODO: Add `get()` returning `GaussianChainMarginals` if
            # explicit scalar batch extraction becomes useful.
            raise ValueError(
                'select must leave at least one structural batch dimension'
            )

        return selected

    def raw_second_moments(self) -> jax.Array:
        r"""Return $\mathbb{E}[x_t x_t^\top]$ with shape `(T, *B, D, D)`."""
        return self.covariances + jnp.einsum(
            't...i,t...j->t...ij',
            self.means,
            self.means,
        )

    def raw_cross_moments(self) -> jax.Array:
        r"""Return $\mathbb{E}[x_t x_{t+1}^\top]$ with shape `(T - 1, *B, D, D)`."""
        return self.cross_covariances + jnp.einsum(
            't...i,t...j->t...ij',
            self.means[:-1],
            self.means[1:],
        )

    def entropy(self) -> jax.Array:
        """Compute the entropy of every Gaussian chain."""
        entropy = jax.vmap(
            lambda posterior: posterior.entropy(),
        )(
            self._flatten(),
        )

        return _unflatten_batch(
            entropy,
            self.batch_shape,
        )

    def expected_log_potential(
        self,
        chain: GaussianChainBatch,
    ) -> jax.Array:
        r"""Compute $\mathbb{E}_q[\log f(x)]$ independently for every chain."""
        if chain.batch_shape != self.batch_shape:
            raise ValueError(
                'chain and posterior must have the same batch shape; '
                f'got {chain.batch_shape} and {self.batch_shape}'
            )

        if chain.num_steps != self.num_steps:
            raise ValueError(
                'chain and posterior must have the same number of steps; '
                f'got {chain.num_steps} and {self.num_steps}'
            )

        if chain.variable_dim != self.variable_dim:
            raise ValueError(
                'chain and posterior must have the same variable dimension; '
                f'got {chain.variable_dim} and {self.variable_dim}'
            )

        expected = jax.vmap(
            lambda posterior, single_chain: posterior.expected_log_potential(
                single_chain
            ),
        )(
            self._flatten(),
            chain._flatten(),
        )

        return _unflatten_batch(
            expected,
            self.batch_shape,
        )

    def paired_marginals(self) -> PairedGaussian:
        """Return time-major adjacent marginals for every chain."""
        return PairedGaussian(
            left=Gaussian(
                mean=self.means[:-1],
                covariance=self.covariances[:-1],
            ),
            right=Gaussian(
                mean=self.means[1:],
                covariance=self.covariances[1:],
            ),
            cross_covariance=jnp.swapaxes(
                self.cross_covariances,
                -2,
                -1,
            ),
        )
