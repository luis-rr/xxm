"""Unary and chain potentials for finite-state discrete variables."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.core import batch
from xxm.core.dists.discrete import Categorical, DiscreteMarginals
from xxm.core.lengths import Lengths


class DiscretePotential(typing.NamedTuple):
    r"""Unary log potential.

    $$\ell_t(k) = \log \phi_t(z_t=k).$$

    `log_values` stores $\ell$. Leading dimensions are batch dimensions.
    """

    log_values: jax.Array  # (..., K)

    def normalize(self) -> Categorical:
        """Return the normalized categorical distribution defined by this potential.

        Requires at least one category with finite log potential in each batch.
        """
        return Categorical(
            probs=jax.nn.softmax(self.log_values, axis=-1),
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.log_values.shape[:-1]

    @property
    def num_states(self) -> int:
        """Number of states."""
        return self.log_values.shape[-1]

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

    def expected_log_potential(
        self,
        posterior: DiscreteMarginals,
    ) -> jax.Array:
        """Expected unary log potentials."""

        states = posterior.lengths.clear(posterior.state_probs)

        log_values = jnp.where(
            states != 0,
            posterior.lengths.clear(self.log_values),
            0,
        )

        return jnp.sum(states * log_values, axis=(-2, -1))


# def _masked_initial_probs(chain: DiscreteChain) -> jax.Array:
#     """Return initial probabilities `(*B, K)`, replacing absent starts with ones.

#     Ones give a neutral log potential for empty valid prefixes. The result is
#     for evaluating log potentials, not a distribution for absent starts.
#     """
#     return jnp.where(chain.lengths.values[..., None] > 0, chain.initial_probs, 1)


class DiscreteChainPotentials(typing.NamedTuple):
    """Initial and pairwise factors of a discrete Markov chain."""

    initial: jax.Array  # (*B, K)
    pairs: jax.Array  # (*B, T-1, K, K)
    lengths: Lengths

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.initial.shape[:-1]

    def broadcast(
        self,
        shape,
        axis: int = 0,
    ) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def expected_log_potential(
        self,
        posterior: DiscreteMarginals,
    ) -> jax.Array:
        """Expected initial and transition log potentials."""

        states = posterior.lengths.clear(posterior.state_probs)

        initial_probs = posterior.lengths.clear(
            self.initial[..., None, :],
            fill=1,
        )[..., 0, :]

        initial = jnp.sum(
            jsp.special.xlogy(states[..., 0, :], initial_probs),
            axis=-1,
        )

        transitions = posterior.lengths.transitions()
        pairs = transitions.clear(posterior.pair_probs)
        transition_probs = transitions.clear(self.pairs, fill=1)

        dynamics = jnp.sum(
            jsp.special.xlogy(pairs, transition_probs),
            axis=(-3, -2, -1),
        )

        return initial + dynamics
