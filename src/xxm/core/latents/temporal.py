import typing

import jax


class TemporalLatent(typing.Protocol):
    """Autonomous vector-valued temporal latent process."""

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

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """Sample exactly `num_steps` temporal values."""
        ...
