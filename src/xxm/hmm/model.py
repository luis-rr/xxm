"""HMM facade classes for user-facing API."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.emissions.discrete import (
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.emissions.discrete_ar import AREmissions
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.optim.loop import Fit, FitCollection

from .core import Model, Posterior
from .inference import infer_exact
from .init import (
    init_gaussian,
    init_gaussian_ar,
    init_poisson,
    init_poisson_ar,
)
from .learning import fit_em, fit_em_many


def _categorical_components_from_params(
    initial_probs: jax.Array,
    transition_probs: jax.Array,
) -> tuple[CategoricalInitial, CategoricalTransitions]:
    return (
        CategoricalInitial(
            dist=Categorical(
                probs=initial_probs,
            )
        ),
        CategoricalTransitions(
            dist=Categorical(
                probs=transition_probs,
            )
        ),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianHMM:
    r"""Hidden Markov model with Gaussian state-conditional emissions.

    $$y_t\mid z_t=k \sim \mathcal{N}(\mu_k, \Sigma_k).$$
    """

    _model: Model[GaussianEmissions]

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    def permute(self, permutation: jax.Array) -> GaussianHMM:
        """Relabel discrete states by permutation."""
        return GaussianHMM(
            _model=self._model.permute(permutation),
        )

    @property
    def states(self) -> Gaussian:
        """State-conditional emission distributions."""
        return self._model.emissions.dist

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely state at each time point."""
        return jnp.argmax(
            posterior.state_probs,
            axis=-1,
        )

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean observation at each time point."""
        return self.states.mixture_mean(
            posterior.state_probs,
        )

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emission_means: jax.Array,
        emission_covariances: jax.Array,
    ) -> typing.Self:
        r"""Construct a Gaussian HMM from its initial, transition, and emission parameters."""
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            _model=Model(
                initial=initial,
                transitions=transitions,
                emissions=GaussianEmissions(
                    dist=Gaussian(
                        mean=emission_means,
                        covariance=emission_covariances,
                    )
                ),
            ),
        )

    @classmethod
    def from_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        *,
        self_transition_prob: float = 0.9,
    ) -> typing.Self:
        """Initialize a Gaussian HMM by clustering the observations with K-means."""
        model = init_gaussian(
            key=key,
            observations=observations,
            num_states=num_states,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

    def sample(self, key: jax.Array, num_steps: int) -> tuple[jax.Array, jax.Array]:
        r"""Sample states $z_{0:T-1}$ and observations $y_{0:T-1}$ from the model."""
        return self._model.sample(
            key,
            num_steps,
        )

    def infer(
        self,
        observations: jax.Array,
    ) -> tuple[Posterior, jax.Array]:
        """Compute the exact posterior over states and the observation log likelihood."""
        return infer_exact(
            self._model,
            observations,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[typing.Self]:
        """Fit model parameters by expectation-maximization."""
        fit = fit_em(
            self._model,
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            model=self.__class__(fit.model),
            objective_trace=fit.objective_trace,
        )

    @classmethod
    def fit_many(
        cls,
        models: tuple[typing.Self, ...],
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'Multi-EM',
    ) -> FitCollection[typing.Self]:
        """Fit multiple HMM initializations to the same observations by EM."""
        fit = fit_em_many(
            tuple(model._model for model in models),
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return FitCollection(
            models=tuple(cls(model) for model in fit.models),
            objective_traces=fit.objective_traces,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonHMM:
    r"""Hidden Markov model with Poisson emissions.

    $$p(y_t|z_t=k) = \operatorname{Poisson}(\exp(\eta_k)).$$
    """

    _model: Model[PoissonEmissions]

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    def permute(self, permutation: jax.Array) -> PoissonHMM:
        """Relabel discrete states by permutation."""
        return PoissonHMM(
            _model=self._model.permute(permutation),
        )

    @property
    def states(self) -> Poisson:
        """State-conditional Poisson emission distributions."""
        return self._model.emissions.dist

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely state at each time point."""
        return jnp.argmax(
            posterior.state_probs,
            axis=-1,
        )

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean observation at each time point."""
        return self.states.mixture_mean(
            posterior.state_probs,
        )

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emission_log_rates: jax.Array,
    ) -> typing.Self:
        r"""Construct a Poisson HMM from categorical and log-rate parameters."""
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            _model=Model(
                initial=initial,
                transitions=transitions,
                emissions=PoissonEmissions(
                    dist=Poisson(
                        log_rates=emission_log_rates,
                    )
                ),
            ),
        )

    @classmethod
    def from_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        *,
        self_transition_prob: float = 0.9,
    ) -> typing.Self:
        """Initialize a Poisson HMM by clustering the observations with K-means."""
        model = init_poisson(
            key=key,
            observations=observations,
            num_states=num_states,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

    def sample(self, key: jax.Array, num_steps: int) -> tuple[jax.Array, jax.Array]:
        r"""Sample states $z_{0:T-1}$ and observations $y_{0:T-1}$ from the model."""
        return self._model.sample(
            key,
            num_steps,
        )

    def infer(self, observations: jax.Array) -> tuple[Posterior, jax.Array]:
        """Compute the exact posterior over states and the observation log likelihood."""
        return infer_exact(
            self._model,
            observations,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[typing.Self]:
        """Fit model parameters by expectation-maximization."""
        fit = fit_em(
            self._model,
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            model=self.__class__(fit.model),
            objective_trace=fit.objective_trace,
        )

    @classmethod
    def fit_many(
        cls,
        models: tuple[typing.Self, ...],
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'Multi-EM',
    ) -> FitCollection[typing.Self]:
        """Fit multiple HMM initializations to the same observations by EM."""
        fit = fit_em_many(
            tuple(model._model for model in models),
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return FitCollection(
            models=tuple(cls(model) for model in fit.models),
            objective_traces=fit.objective_traces,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianARHMM:
    r"""
    Autoregressive HMM with Gaussian conditional emissions.

    The first ``num_lags`` observations of a fitted or inferred sequence are
    treated as fixed conditioning history and have no associated latent states.
    The first latent state is drawn independently from ``initial_probs`` and
    selects the regression generating the first observation after that history.

    Conditional predictors have shape ``(L, N)`` and are ordered from most
    recent to oldest observation. Affine coefficients therefore have shape
    ``(K, N, L, N)``.

    Autonomous sampling starts from a zero prehistory. An explicit
    ``initial_history`` may instead be supplied to generate a continuation from
    observed values.
    """

    model: Model[AREmissions[LinearGaussian]]

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.model.num_states

    @property
    def output_dim(self) -> int:
        """Observation dimension $N$."""
        return self.model.emissions.output_dim

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self.model.emissions.num_lags

    def permute(
        self,
        permutation: jax.Array,
    ) -> GaussianARHMM:
        """Relabel discrete states by permutation."""
        return GaussianARHMM(
            model=self.model.permute(permutation),
        )

    @property
    def states(self) -> LinearGaussian:
        """State-conditional autoregressive Gaussian distributions."""
        return self.model.emissions.dist

    def states_conditional(
        self,
        observations: jax.Array,
    ) -> Gaussian:
        """
        Conditional distributions for observations following the AR history.

        For an input sequence of length ``T``, the returned distributions have
        ``T - num_lags`` time steps.
        """
        return self.model.emissions.conditional(
            observations,
        )

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely state at each modeled time point."""
        return jnp.argmax(
            posterior.state_probs,
            axis=-1,
        )

    def observation_mean(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> jax.Array:
        """Return the posterior mean observation conditional on the AR history."""
        return self.states_conditional(
            observations,
        ).mixture_mean(
            posterior.state_probs,
        )

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emission_coefficients: jax.Array,  # (K, N, L, N)
        emission_bias: jax.Array,  # (K, N)
        emission_covariances: jax.Array,  # (K, N, N)
    ) -> typing.Self:
        r"""Construct a Gaussian AR-HMM from state, transition, and AR parameters.

        The coefficients have shape $(K,N,L,N)$ and the first $L$ observations
        serve as fixed conditioning history.
        """
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            model=Model(
                initial=initial,
                transitions=transitions,
                emissions=AREmissions(
                    dist=LinearGaussian(
                        affine=Affine(
                            coefficients=emission_coefficients,
                            bias=emission_bias,
                        ),
                        covariance=emission_covariances,
                    )
                ),
            ),
        )

    @classmethod
    def from_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        num_lags: int,
        *,
        self_transition_prob: float = 0.9,
    ) -> typing.Self:
        """
        Initialize from autoregressive regression pairs.

        The first ``num_lags`` observations are used only as fixed conditioning
        history.
        """
        model = init_gaussian_ar(
            key=key,
            observations=observations,
            num_states=num_states,
            num_lags=num_lags,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
        *,
        initial_history: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Sample states and observations from the AR-HMM.

        If ``initial_history`` is omitted, generation uses a zero prehistory.
        Otherwise it must have shape ``(L, N)`` and be ordered chronologically,
        from oldest to most recent. Only the newly generated observations are
        returned.
        """
        if initial_history is None:
            return self.model.sample(
                key,
                num_steps,
            )

        return self.model.sample_continuation(
            key,
            num_steps,
            initial_history,
        )

    def infer(
        self,
        observations: jax.Array,
    ) -> tuple[Posterior, jax.Array]:
        """
        Infer states conditional on the first ``num_lags`` observations.

        For an input sequence of length ``T``, the posterior has
        ``T - num_lags`` latent steps.
        """
        return infer_exact(
            self.model,
            observations,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[typing.Self]:
        """Fit the conditional AR-HMM with expectation maximization."""
        fit = fit_em(
            self.model,
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            model=self.__class__(fit.model),
            objective_trace=fit.objective_trace,
        )

    @classmethod
    def fit_many(
        cls,
        models: tuple[typing.Self, ...],
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'Multi-EM',
    ) -> FitCollection[typing.Self]:
        """Fit multiple AR-HMM initializations to the same sequence."""
        fit = fit_em_many(
            tuple(model.model for model in models),
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return FitCollection(
            models=tuple(cls(model) for model in fit.models),
            objective_traces=fit.objective_traces,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonARHMM:
    r"""
    Autoregressive HMM with Poisson conditional emissions.

    The first ``num_lags`` observations of a fitted or inferred sequence are
    treated as fixed conditioning history and have no associated latent states.
    The first latent state is drawn independently from ``initial_probs`` and
    selects the regression generating the first observation after that history.

    Conditional predictors have shape ``(L, N)`` and are ordered from most
    recent to oldest observation. Affine coefficients therefore have shape
    ``(K, N, L, N)``.

    Autonomous sampling starts from a zero prehistory. An explicit
    ``initial_history`` may instead be supplied to generate a continuation from
    observed values.
    """

    model: Model[AREmissions[LinearPoisson]]

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.model.num_states

    @property
    def output_dim(self) -> int:
        """Observation dimension $N$."""
        return self.model.emissions.output_dim

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self.model.emissions.num_lags

    def permute(
        self,
        permutation: jax.Array,
    ) -> PoissonARHMM:
        """Relabel discrete states by permutation."""
        return PoissonARHMM(
            model=self.model.permute(permutation),
        )

    @property
    def states(self) -> LinearPoisson:
        """State-conditional autoregressive Poisson distributions."""
        return self.model.emissions.dist

    def states_conditional(
        self,
        observations: jax.Array,
    ) -> Poisson:
        """
        Conditional distributions for observations following the AR history.

        For an input sequence of length ``T``, the returned distributions have
        ``T - num_lags`` time steps.
        """
        return self.model.emissions.conditional(
            observations,
        )

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely state at each modeled time point."""
        return jnp.argmax(
            posterior.state_probs,
            axis=-1,
        )

    def observation_mean(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> jax.Array:
        """Return the posterior mean observation conditional on the AR history."""
        return self.states_conditional(
            observations,
        ).mixture_mean(
            posterior.state_probs,
        )

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emission_coefficients: jax.Array,  # (K, N, L, N)
        emission_bias: jax.Array,  # (K, N)
    ) -> typing.Self:
        r"""Construct a Poisson AR-HMM from state, transition, and log-rate parameters.

        The coefficients have shape $(K,N,L,N)$ and the first $L$ observations
        serve as fixed conditioning history.
        """
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            model=Model(
                initial=initial,
                transitions=transitions,
                emissions=AREmissions(
                    dist=LinearPoisson(
                        affine=Affine(
                            coefficients=emission_coefficients,
                            bias=emission_bias,
                        )
                    )
                ),
            ),
        )

    @classmethod
    def from_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        num_lags: int,
        *,
        self_transition_prob: float = 0.9,
    ) -> typing.Self:
        """
        Initialize from autoregressive regression pairs.

        The first ``num_lags`` observations are used only as fixed conditioning
        history.
        """
        model = init_poisson_ar(
            key=key,
            observations=observations,
            num_states=num_states,
            num_lags=num_lags,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
        *,
        initial_history: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Sample states and observations from the AR-HMM.

        If ``initial_history`` is omitted, generation uses a zero prehistory.
        Otherwise it must have shape ``(L, N)`` and be ordered chronologically,
        from oldest to most recent. Only the newly generated observations are
        returned.
        """
        if initial_history is None:
            return self.model.sample(
                key,
                num_steps,
            )

        return self.model.sample_continuation(
            key,
            num_steps,
            initial_history,
        )

    def infer(self, observations: jax.Array) -> tuple[Posterior, jax.Array]:
        """
        Infer states conditional on the first ``num_lags`` observations.

        For an input sequence of length ``T``, the posterior has
        ``T - num_lags`` latent steps.
        """
        return infer_exact(
            self.model,
            observations,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[typing.Self]:
        """Fit the conditional AR-HMM with expectation maximization."""
        fit = fit_em(
            self.model,
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            model=self.__class__(fit.model),
            objective_trace=fit.objective_trace,
        )

    @classmethod
    def fit_many(
        cls,
        models: tuple[typing.Self, ...],
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'Multi-EM',
    ) -> FitCollection[typing.Self]:
        """Fit multiple AR-HMM initializations to the same sequence."""
        fit = fit_em_many(
            tuple(model.model for model in models),
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return FitCollection(
            models=tuple(cls(model) for model in fit.models),
            objective_traces=fit.objective_traces,
        )
