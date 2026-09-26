import typing

import jax

from xxm.core.data import Sequences
from xxm.core.lengths import LengthsSpec


class TemporalLatent(typing.Protocol):
    """Autonomous vector-valued temporal latent process."""

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Independent structural batch dimensions."""
        ...

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
        ls: LengthsSpec,
    ) -> Sequences:
        """Sample temporal values."""
        ...
