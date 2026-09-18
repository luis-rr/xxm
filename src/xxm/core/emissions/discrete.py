"""Emission models for discrete latent variables (HMM emissions)."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import _batch
from xxm.core.chains.discrete import DiscretePotential
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson
from xxm.core.optim import categorical as categorical_fit
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.posteriors import DiscretePosterior


class Emissions(typing.Protocol):
    """Protocol for memoryless emissions conditional on discrete latent states."""

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def num_states(self) -> int: ...

    def select(self, index) -> typing.Self: ...

    def broadcast(self, shape, axis: int = 0) -> typing.Self: ...

    def squeeze(self, axis=None) -> typing.Self: ...

    def permute(self, permutation, axis: int = 0) -> typing.Self: ...

    def move_axis(self, source: int, destination: int) -> typing.Self: ...

    def log_likelihoods(
        self,
        observations: jax.Array,
    ) -> jax.Array:
        """Evaluate state-conditional log likelihoods over time."""
        ...

    def compute_potential(
        self,
        observations: jax.Array,
    ) -> DiscretePotential:
        """Construct discrete state potentials from observation likelihoods."""
        ...

    def fit_params(
        self,
        observations: jax.Array,
        posterior: DiscretePosterior,
    ) -> typing.Self:
        """Fit emission parameters from observations and posterior state weights."""
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

    def select(self, index) -> typing.Self:
        index = _batch.selection(index, len(self.batch_shape))
        return self._replace(dist=self.dist.select(index + (slice(None),)))

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self._replace(dist=self.dist.broadcast(shape, axis=axis))

    def squeeze(self, axis=None) -> typing.Self:
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self._replace(dist=self.dist.squeeze(axes))

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        axis = _batch.axis_index(axis, len(self.batch_shape))
        return self._replace(dist=self.dist.permute(permutation, axis=axis))

    def move_axis(self, source: int, destination: int) -> typing.Self:
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self._replace(dist=self.dist.move_axis(source, destination))

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        assert self.dist.batch_shape
        return self.dist.batch_shape[-1]

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        r"""Evaluate log probabilities $\log p(y_t|z_t=k)$ for all states and time."""
        values = _batch.broadcast_array(
            observations, (self.num_states,), axis=len(self.batch_shape)
        )  # (*B, K, *Q, D)
        return jnp.moveaxis(self.dist.log_prob(values), len(self.batch_shape), -1)

    def compute_potential(self, observations: jax.Array) -> DiscretePotential:
        """Construct one state potential from each observation log likelihood."""
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self,
        observations: jax.Array,
        posterior: DiscretePosterior,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        """Fit Gaussian parameters from posterior state weights."""

        batch_shape = self.batch_shape

        def fit_one(observations_i, state_probs_i, current_i):
            weights = jnp.moveaxis(state_probs_i, -1, 0)
            fitted = gaussian_fit.from_samples_weighted(
                observations_i,
                weights,
                covariance_floor=covariance_floor,
            )
            return categorical_fit.filter_valid_batches(fitted, current_i, weights)

        flat_fitted = jax.vmap(fit_one)(
            _batch.flatten_batch(observations, batch_shape),
            _batch.flatten_batch(posterior.state_probs, batch_shape),
            _batch.flatten_batch(self.dist, batch_shape),
        )
        return self._replace(
            dist=_batch.unflatten_batch(flat_fitted, batch_shape),
        )

    def sample(self, key: jax.Array, states: jax.Array) -> jax.Array:
        """Sample observations conditional on discrete state indices."""
        return _batch.take_along_last_batch(
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

    def select(self, index) -> typing.Self:
        index = _batch.selection(index, len(self.batch_shape))
        return self._replace(dist=self.dist.select(index + (slice(None),)))

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self._replace(dist=self.dist.broadcast(shape, axis=axis))

    def squeeze(self, axis=None) -> typing.Self:
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self._replace(dist=self.dist.squeeze(axes))

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        axis = _batch.axis_index(axis, len(self.batch_shape))
        return self._replace(dist=self.dist.permute(permutation, axis=axis))

    def move_axis(self, source: int, destination: int) -> typing.Self:
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self._replace(dist=self.dist.move_axis(source, destination))

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        assert self.dist.batch_shape
        return self.dist.batch_shape[-1]

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        r"""Evaluate $\log p(y_t\mid z_t=k)$ for all states and time."""
        values = _batch.broadcast_array(
            observations, (self.num_states,), axis=len(self.batch_shape)
        )  # (*B, K, *Q, D)
        return jnp.moveaxis(self.dist.log_prob(values), len(self.batch_shape), -1)

    def compute_potential(self, observations: jax.Array) -> DiscretePotential:
        """Construct one state potential from each observation log likelihood."""
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self,
        observations: jax.Array,
        posterior: DiscretePosterior,
    ) -> typing.Self:
        """Fit state-specific Poisson log rates from posterior state weights."""

        batch_shape = self.batch_shape

        def fit_one(observations_i, state_probs_i, current_i):
            weights = jnp.moveaxis(state_probs_i, -1, 0)
            fitted = poisson_fit.from_samples_weighted(
                values=observations_i, weights=weights
            )
            return categorical_fit.filter_valid_batches(fitted, current_i, weights)

        flat_fitted = jax.vmap(fit_one)(
            _batch.flatten_batch(observations, batch_shape),
            _batch.flatten_batch(posterior.state_probs, batch_shape),
            _batch.flatten_batch(self.dist, batch_shape),
        )
        return self._replace(
            dist=_batch.unflatten_batch(flat_fitted, batch_shape),
        )

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
        return _batch.take_along_last_batch(
            self.dist, self.dist.batch_shape, states
        ).sample(key)
