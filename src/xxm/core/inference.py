"""Common inference result containers."""

import typing

import jax
import jax.numpy as jnp

ModelT = typing.TypeVar('ModelT')
PosteriorT = typing.TypeVar('PosteriorT')


class Inferred(
    typing.NamedTuple,
    typing.Generic[ModelT, PosteriorT],
):
    """Model, posterior, and objective produced by inference."""

    model: ModelT
    posterior: PosteriorT
    objective: jax.Array


class Fitted(
    typing.NamedTuple,
    typing.Generic[ModelT, PosteriorT],
):
    """Result of fitting a model."""

    model: ModelT
    posterior: PosteriorT
    objective_trace: jax.Array


class FittedCollection(
    typing.NamedTuple,
    typing.Generic[ModelT, PosteriorT],
):
    """Results from fitting multiple model initializations."""

    models: tuple[ModelT, ...]
    posteriors: tuple[PosteriorT, ...]
    objective_traces: jax.Array

    def is_valid(self) -> jax.Array:
        """Return which fits have NaN-free objective traces."""
        return ~jnp.isnan(self.objective_traces).any(axis=1)

    def best_index(self) -> int:
        """Return the index of the fit with the highest final objective."""

        valid = self.is_valid()

        if not jnp.any(valid):
            raise ValueError('No fit has a NaN-free objective trace')

        valid_indices = jnp.where(valid)[0]

        best_valid_index = jnp.argmax(self.objective_traces[valid, -1])

        return int(valid_indices[best_valid_index])

    def get(
        self,
        index: int,
    ) -> Fitted[ModelT, PosteriorT]:
        """Return one fitted result."""

        return Fitted(
            model=self.models[index],
            posterior=self.posteriors[index],
            objective_trace=self.objective_traces[index],
        )

    def best(
        self,
    ) -> Fitted[ModelT, PosteriorT]:
        """Return the fit with the highest final objective."""
        return self.get(self.best_index())
