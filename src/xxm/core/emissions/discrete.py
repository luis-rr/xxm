"""Emission models for discrete latent variables (HMM emissions)."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.chains.discrete import DiscretePotential
from xxm.core.data import WeightedObservations
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.batch import filter_valid_batches
from xxm.core.posteriors import DiscretePosterior


def _validate_weighted_samples(
    observations: WeightedObservations,
    posterior: DiscretePosterior,
    model_batch_shape: tuple[int, ...],
    num_states: int,
    observation_dim: int,
):
    """Require observations and posterior states to align with the emissions."""

    observations.validate()

    batch.require_same_shape(observations.batch_shape, posterior.batch_shape)
    batch.split_prefix(
        observations.batch_shape, model_batch_shape, name='observation batch shape'
    )

    if observations.observation_dim != observation_dim:
        raise ValueError('observations must match the emission dimension')

    if posterior.state_probs.shape != (*observations.weights.shape, num_states):
        raise ValueError('posterior states must match observation batch and time axes')


def _prepare_weighted_samples(
    observations: WeightedObservations,
    posterior: DiscretePosterior,
    model_batch_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Prepare validated observations and effective state weights for pooling.

    Combine observation weights with posterior state probabilities, then pool
    replicate and time axes into S samples. Return values `(*M, S, D)` and
    state weights `(*M, S, K)`, preserving the model batch `*M`.
    """
    state_weights = observations.safe_aligned(posterior.state_probs)
    state_weights = state_weights * observations.weights[..., None]

    replicate_shape = batch.split_prefix(
        observations.batch_shape,
        model_batch_shape,
    )

    return batch.pool_samples(
        (observations.safe_values(), state_weights),
        model_batch_shape,
        (*replicate_shape, observations.num_steps),
    )


class Emissions(typing.Protocol):
    """Protocol for memoryless emissions conditional on discrete latent states."""

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def num_states(self) -> int: ...

    @property
    def observation_dim(self) -> int: ...

    def select(self, index: batch.SelT) -> typing.Self: ...

    def broadcast(self, shape, axis: int = 0) -> typing.Self: ...

    def squeeze(self, axis=None) -> typing.Self: ...

    def permute(self, permutation, axis: int = 0) -> typing.Self: ...

    def move_axis(self, source: int, destination: int) -> typing.Self: ...

    def log_likelihoods(
        self,
        observations: WeightedObservations,
    ) -> jax.Array:
        """Evaluate state-conditional log likelihoods over time."""
        ...

    def compute_potential(
        self,
        observations: WeightedObservations,
    ) -> DiscretePotential:
        """Construct discrete state potentials from observation likelihoods."""
        ...

    def fit_params(
        self,
        observations: WeightedObservations,
        posterior: DiscretePosterior,
    ) -> typing.Self:
        """Fit emission parameters from observations and posterior state weights."""
        ...

    def observation_mean(self, posterior: DiscretePosterior) -> jax.Array:
        """Compute posterior mean observations with aligned structural batches."""
        ...

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel state-specific emission parameters."""
        ...

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        """Sample observations conditional on discrete states."""
        ...


class GaussianEmissions(typing.NamedTuple):
    r"""State-dependent Gaussian emissions.

    $$y_t|z_t=k \sim \mathcal{N}(\mu_k, \Sigma_k).$$
    """

    dist: Gaussian  # underlying batch (*B, K); K is the intrinsic state axis

    @property
    def batch_shape(self) -> tuple[int, ...]:
        assert self.dist.batch_shape
        return self.dist.batch_shape[:-1]

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        assert self.dist.batch_shape
        return self.dist.batch_shape[-1]

    @property
    def observation_dim(self) -> int:
        return self.dist.variable_dim

    def log_likelihoods(self, observations: WeightedObservations) -> jax.Array:
        r"""Evaluate weighted $\log p(y_t|z_t=k)$ for all states and time."""
        observations.validate()
        batch.require_same_shape(self.batch_shape, observations.batch_shape)

        values = batch.broadcast_array(
            observations.safe_values(),
            (self.num_states,),
            axis=len(self.batch_shape),
        )  # (*B, K, *Q, D)
        terms = jnp.moveaxis(self.dist.log_prob(values), len(self.batch_shape), -1)

        return observations.safe_aligned(terms) * observations.weights[..., None]

    def compute_potential(
        self, observations: WeightedObservations
    ) -> DiscretePotential:
        """Construct one state potential from each observation log likelihood."""
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self,
        observations: WeightedObservations,
        posterior: DiscretePosterior,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        """Fit Gaussian parameters from posterior state weights."""

        batch_shape = self.batch_shape
        _validate_weighted_samples(
            observations,
            posterior,
            batch_shape,
            self.num_states,
            self.observation_dim,
        )

        values, state_weights = _prepare_weighted_samples(
            observations, posterior, batch_shape
        )

        def fit_one(observations_i, state_weights_i, current_i):
            weights = jnp.moveaxis(state_weights_i, -1, 0)
            fitted = gaussian_fit.from_samples_weighted(
                observations_i,
                weights,
                covariance_floor=covariance_floor,
            )
            return filter_valid_batches(
                fitted, current_i, weights, min_expected_count=1.0
            )

        fitted = batch.vmap_batch(
            fit_one,
            values,
            state_weights,
            self.dist,
            batch_shape=batch_shape,
        )
        return self._replace(
            dist=fitted,
        )

    def observation_mean(self, posterior: DiscretePosterior) -> jax.Array:
        """Average state means under the posterior at every timestep."""
        batch.require_same_shape(self.batch_shape, posterior.batch_shape)
        states = self.dist.broadcast(posterior.num_steps, axis=len(self.batch_shape))
        return states.mixture_mean(posterior.state_probs, axis=-1)

    def sample(self, key: jax.Array, states: jax.Array) -> jax.Array:
        """Sample observations conditional on discrete state indices."""
        return batch.take_along_last_batch(
            self.dist, self.dist.batch_shape, states
        ).sample(key)

    def permute_states(self, permutation: jax.Array) -> 'GaussianEmissions':
        """Relabel state-specific emission parameters."""
        return self._replace(
            dist=self.dist.permute(permutation, axis=-1),
        )


class PoissonEmissions(typing.NamedTuple):
    r"""State-dependent Poisson emissions parameterized by log rates.

    $$y_t\mid z_t=k \sim \operatorname{Poisson}(\lambda_k).$$

    `dist.log_rates` stores $\eta_k=\log\lambda_k$.
    """

    dist: Poisson  # underlying batch (*B, K); K is the intrinsic state axis

    @property
    def batch_shape(self) -> tuple[int, ...]:
        assert self.dist.batch_shape
        return self.dist.batch_shape[:-1]

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        assert self.dist.batch_shape
        return self.dist.batch_shape[-1]

    @property
    def observation_dim(self) -> int:
        return self.dist.variable_dim

    def log_likelihoods(self, observations: WeightedObservations) -> jax.Array:
        r"""Evaluate weighted $\log p(y_t\mid z_t=k)$ for all states and time."""
        observations.validate()
        batch.require_same_shape(self.batch_shape, observations.batch_shape)
        values = batch.broadcast_array(
            observations.safe_values(), (self.num_states,), axis=len(self.batch_shape)
        )  # (*B, K, *Q, D)
        terms = jnp.moveaxis(self.dist.log_prob(values), len(self.batch_shape), -1)
        return observations.safe_aligned(terms) * observations.weights[..., None]

    def compute_potential(
        self, observations: WeightedObservations
    ) -> DiscretePotential:
        """Construct one state potential from each observation log likelihood."""
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self,
        observations: WeightedObservations,
        posterior: DiscretePosterior,
    ) -> typing.Self:
        """Fit state-specific Poisson log rates from posterior state weights."""

        batch_shape = self.batch_shape
        _validate_weighted_samples(
            observations, posterior, batch_shape, self.num_states, self.observation_dim
        )
        values, state_weights = _prepare_weighted_samples(
            observations, posterior, batch_shape
        )

        def fit_one(observations_i, state_weights_i, current_i):
            weights = jnp.moveaxis(state_weights_i, -1, 0)
            fitted = poisson_fit.from_samples_weighted(
                values=observations_i, weights=weights
            )
            return filter_valid_batches(
                fitted, current_i, weights, min_expected_count=1.0
            )

        fitted = batch.vmap_batch(
            fit_one,
            values,
            state_weights,
            self.dist,
            batch_shape=batch_shape,
        )
        return self._replace(
            dist=fitted,
        )

    def observation_mean(self, posterior: DiscretePosterior) -> jax.Array:
        """Average state rates under the posterior at every timestep."""
        batch.require_same_shape(self.batch_shape, posterior.batch_shape)
        states = self.dist.broadcast(posterior.num_steps, axis=len(self.batch_shape))
        return states.mixture_mean(posterior.state_probs, axis=-1)

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> 'PoissonEmissions':
        """Relabel state-specific emission parameters."""
        return self._replace(
            dist=self.dist.permute(permutation, axis=-1),
        )

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        """Sample observations conditional on discrete state indices."""
        return batch.take_along_last_batch(
            self.dist, self.dist.batch_shape, states
        ).sample(key)
