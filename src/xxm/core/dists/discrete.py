"""Categorical distributions over finite categories."""

import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.core import batch
from xxm.core.lengths import Lengths


class Categorical(typing.NamedTuple):
    r"""Categorical distribution.

    $$\pi_k = p(z=k).$$

    `probs` stores the category probabilities. Leading dimensions are batch dimensions,
    shared across all attributes.
    """

    probs: jax.Array  # (..., K)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.probs.shape[:-1]

    @property
    def num_categories(self) -> int:
        """Number of categories."""
        return self.probs.shape[-1]

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def permute_categories(
        self,
        permutation: jax.Array,  # (K,)
    ) -> 'Categorical':
        """Relabel categories by permutation."""
        return self._replace(
            probs=self.probs[..., permutation],
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def astype(
        self,
        dtype: jax.typing.DTypeLike,
    ) -> 'Categorical':
        """Convert to a different data type."""
        return self._replace(
            probs=self.probs.astype(dtype),
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Sample indices with shape `(*batch_shape, *sample_shape)`."""
        shape = self.batch_shape + (1,) * len(sample_shape) + (self.num_categories,)
        return jax.random.categorical(
            key,
            logits=jnp.log(self.probs).reshape(shape),
            shape=self.batch_shape + sample_shape,
        )

    def log_prob(self, values: jax.Array) -> jax.Array:
        """Evaluate category indices with aligned batch and subsequent query axes."""
        probs = batch.align_array(self.probs, self.batch_shape, values.shape)
        return jnp.take_along_axis(jnp.log(probs), values[..., None], axis=-1)[..., 0]

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)


class DiscreteChainMarginals(typing.NamedTuple):
    r"""
    Marginals of the normalized finite-state chain distribution.

    For $q(z) = f(z) / Z$,

    - `state_probs[t,k]` stores $\gamma_t(k) = q(z_t=k)$
    - `pair_probs[t,i,j]` stores $\xi_t(i,j) = q(z_t=i,z_{t+1}=j)$

    Leading batch dimensions index independent chains.
    """

    state_probs: jax.Array  # (*B, T, K)
    pair_probs: jax.Array  # (*B, T - 1, K, K)
    lengths: Lengths

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading independent-chain dimensions."""
        return self.state_probs.shape[:-2]

    @property
    def max_steps(self) -> int:
        """Number of time steps."""
        return self.state_probs.shape[-2]

    @property
    def num_states(self) -> int:
        """Number of discrete states."""
        return self.state_probs.shape[-1]

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

    def incoming_state_probs(
        self,
    ) -> jax.Array:
        """State probabilities associated with incoming transitions."""
        return self.state_probs[..., 1:, :]

    def expected_initial_counts(self) -> jax.Array:
        """Expected initial counts, pooled over the sequence axis."""
        counts = jnp.where(
            self.lengths.values[..., None] > 0,
            self.state_probs[..., 0, :],
            0,
        )

        return counts.sum(axis=-2) if self.batch_shape else counts

    def expected_transition_counts(self) -> jax.Array:
        """Expected transition counts, pooled over sequences and time."""
        transitions = self.lengths.transitions().clear(self.pair_probs)

        return (
            transitions.sum(axis=(-4, -3))
            if self.batch_shape
            else transitions.sum(axis=-3)
        )

    def entropy(self) -> jax.Array:
        """
        Entropy of the normalized chain distribution.
        """

        states = self.lengths.clear(self.state_probs)

        initial_entropy = -jnp.sum(
            jsp.special.xlogy(
                states[..., 0, :],
                states[..., 0, :],
            ),
            axis=-1,
        )

        transitions = self.lengths.transitions()
        pairs = transitions.clear(self.pair_probs)

        pair_entropy = -jnp.sum(
            jsp.special.xlogy(
                pairs,
                pairs,
            ),
            axis=(-2, -1),
        )

        conditioning_entropy = jnp.sum(
            jsp.special.xlogy(
                states[..., :-1, :],
                states[..., :-1, :],
            ),
            axis=-1,
        )

        return initial_entropy + jnp.sum(
            transitions.clear(pair_entropy + conditioning_entropy),
            axis=-1,
        )

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """Relabel discrete states by permutation."""
        permutation = batch.permutation_indices(permutation, self.num_states)
        return self._replace(
            state_probs=self.state_probs[..., permutation],
            pair_probs=self.pair_probs[..., permutation, :][..., permutation],
        )


class DiscreteMarginals(typing.Protocol):
    r"""Protocol for discrete marginals.

    Represents marginal probabilities $\gamma_t(k)$ of discrete latent states
    and pair marginals $\xi_t(i,j)$ under a represented distribution $q(z)$.

    It may hold, for instance, the posterior distribution after inference.

    Independent batch dimensions precede the time and state dimensions.
    """

    @property
    def lengths(self) -> Lengths: ...

    @property
    def max_steps(self) -> int:
        r"""Number of time steps $T$."""
        ...

    @property
    def num_states(self) -> int:
        r"""Number of discrete states $K$."""
        ...

    @property
    def batch_shape(self) -> tuple[int, ...]:
        r"""Shape of independent batch dimensions (*B)."""
        ...

    @property
    def state_probs(self) -> jax.Array:  # (*B, T, K)
        r"""State marginals $\gamma_t(k)=q(z_t=k)$."""
        ...

    @property
    def pair_probs(self) -> jax.Array:  # (*B, T-1, K, K)
        r"""Adjacent-state marginals $\xi_t(i,j)=q(z_t=i,z_{t+1}=j)$."""
        ...
