"""Exact inference conditional on fixed autoregressive history."""

import typing

import jax
import jax.numpy as jnp

from xxm.core.inference import InferenceState
from xxm.core.inference.discrete import infer as discrete_infer
from xxm.core.latents.discrete import compute_chain_potentials
from xxm.core.potentials.discrete import (
    DiscreteChainPotentials,
    DiscretePotential,
)

from .core import Model, Posterior
from .data import ARDatasetCollection
from .emissions import ConditionalDistT


class Potentials(typing.NamedTuple):
    latent: DiscreteChainPotentials
    observation: DiscretePotential

    @classmethod
    def from_data(
        cls,
        model: Model[ConditionalDistT],
        data: ARDatasetCollection,
    ) -> typing.Self:
        """Construct latent and observation factors on the compact AR timeline."""

        if data.input_dim != 0:
            raise ValueError('AR-HMM does not currently support known external inputs')

        if data.num_lags != model.num_lags:
            raise ValueError('Mismatch between data.num_lags and model.num_lags')

        latent = compute_chain_potentials(
            model.initial,
            model.transitions,
            data.pool_lengths_spec(),
        )

        observation = model.emissions.compute_potential(
            data.weighted_observations(),
        )

        return cls(
            latent=latent,
            observation=observation,
        )


def infer_exact(
    model: Model[ConditionalDistT],
    data: ARDatasetCollection,
) -> InferenceState[Model[ConditionalDistT], Posterior]:
    """Infer all independent AR sequences on their compact timeline."""

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
    model: Model[ConditionalDistT],
    data: ARDatasetCollection,
    posterior: Posterior,
) -> jax.Array:
    """Evaluate the conditional AR-HMM ELBO."""

    potentials = Potentials.from_data(model, data)

    terms = (
        potentials.latent.expected_log_potential(posterior),
        potentials.observation.expected_log_potential(posterior),
        posterior.entropy(),
    )

    return jnp.sum(sum(terms), axis=-1)
