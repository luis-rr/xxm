import typing

import jax


class BatchedLatent(typing.Protocol):
    """Vector-valued latent representation."""

    @property
    def variable_dim(self) -> int:
        """Dimension of each latent vector."""
        ...

    def reorient_variables(
        self,
        signs: jax.Array,
    ) -> typing.Self:
        """Reorient variable coordinates."""
        ...

    def permute_variables(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel variable coordinates."""
        ...


class TemporalLatent(
    BatchedLatent,
    typing.Protocol,
):
    """Autonomous temporal latent process."""

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """Sample exactly `num_steps` temporal values."""
        ...


class SwitchingTemporalLatent(
    BatchedLatent,
    typing.Protocol,
):
    """State-indexed temporal latent process, optionally conditioned on gates."""

    @property
    def num_states(self) -> int:
        """Number of discrete-state-indexed temporal processes."""
        ...

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
        gates: jax.Array,
    ) -> jax.Array:
        """Sample exactly `num_steps` state-indexed temporal values."""
        ...

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel discrete-state-indexed temporal parameters."""
        ...
