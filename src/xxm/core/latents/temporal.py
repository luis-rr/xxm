import typing

import jax

from xxm.core import batch
from xxm.core.data import Sequences
from xxm.core.lengths import LengthsSpec


class TemporalVariable(
    batch.VariableBatchable,
    typing.Protocol,
):
    """Autonomous vector-valued temporal latent process."""

    def sample(
        self,
        key: jax.Array,
        ls: LengthsSpec,
    ) -> Sequences:
        """Sample temporal values."""
        ...


class ShiftableVariable(
    batch.VariableBatchable,
    typing.Protocol,
):
    """Variable-batchable object closed under constant coordinate shifts."""

    def shift_variables(
        self,
        offset: jax.Array,
    ) -> typing.Self: ...
