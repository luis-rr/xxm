import typing

import jax


class DiscretePosterior(typing.Protocol):
    @property
    def state_probs(self) -> jax.Array:  # (T, K)
        ...

    @property
    def pair_probs(self) -> jax.Array:  # (T, K, K)
        ...


class ContinuousPosterior(typing.Protocol):
    @property
    def means(self) -> jax.Array:  # (T, D)
        ...

    @property
    def covariances(self) -> jax.Array:  # (T, D, D)
        ...

    def raw_second_moments(self) -> jax.Array:  # (T, D, D)
        ...

    def raw_cross_moments(self) -> jax.Array:  # (T-1, D, D)
        ...
