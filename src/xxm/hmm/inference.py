"""Exact inference for Hidden Markov Models."""

import typing

import jax
import jax.numpy as jnp

from xxm.core.data import DatasetCollection
from xxm.core.inference import InferenceState
from xxm.core.inference.discrete import infer as discrete_infer
from xxm.core.latents.discrete import compute_chain_potentials
from xxm.core.potentials.discrete import DiscreteChainPotentials, DiscretePotential

from .core import EmissionsT, Model, Posterior


class Potentials(typing.NamedTuple):
    latent: DiscreteChainPotentials
    observation: DiscretePotential

    @classmethod
    def from_data(
        cls,
        model: Model[EmissionsT],
        data: DatasetCollection,
    ) -> typing.Self:

        return cls(
            latent=compute_chain_potentials(
                model.initial,
                model.transitions,
                data.pool_lengths_spec(),
            ),
            observation=model.emissions.compute_potential(
                data.weighted_observations(),
            ),
        )


InferenceT = InferenceState[
    Model[EmissionsT],
    Posterior,
]


def infer_exact(
    model: Model[EmissionsT],
    data: DatasetCollection,
) -> InferenceT:
    """Run exact inference across all sessions."""

    potentials = Potentials.from_data(model, data)

    inferred = discrete_infer(
        potentials.latent,
        potentials.observation,
    )

    return InferenceState(
        model=model,
        posterior=inferred.posterior,
        objective=inferred.log_normalizer_total(),
    )


def elbo(
    model: Model[EmissionsT],
    data: DatasetCollection,
    posterior: Posterior,
) -> jax.Array:
    potentials = Potentials.from_data(model, data)

    terms = (
        potentials.latent.expected_log_potential(posterior),
        potentials.observation.expected_log_potential(posterior),
        posterior.entropy(),
    )

    return jnp.sum(sum(terms), axis=-1)
