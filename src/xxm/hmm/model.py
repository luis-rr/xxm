"""HMM facade classes for user-facing API."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson
from xxm.core.emissions.discrete import (
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.inference import Fitted, InferenceState
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)

from .core import Model, Posterior
from .inference import infer_exact
from .init import (
    DEFAULT_SELF_TRANSITION_PROB,
    init_gaussian_via_kmeans,
    init_poisson_via_kmeans,
)
from .learning import fit_em

_infer_exact_jit = jax.jit(infer_exact)

_fit_em_jit = jax.jit(
    fit_em,
    static_argnames=(
        'num_iters',
        'progress',
    ),
)


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

    $$z_0\sim\operatorname{Categorical}(\pi),\qquad
    p(z_{t+1}=j\mid z_t=i)=P(i,j),$$

    $$y_t\mid z_t=k \sim \mathcal{N}(\mu_k, \Sigma_k).$$

    `_model.initial.dist.probs` stores $\pi$, `_model.transitions.dist.probs`
    stores $P$, and `states` stores the Gaussian emission parameters.
    """

    _model: Model[GaussianEmissions]

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    def permute_states(self, permutation: jax.Array) -> GaussianHMM:
        """Relabel discrete states by permutation."""
        return GaussianHMM(
            _model=self._model.permute_states(permutation),
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
        return self.states.broadcast(posterior.state_probs.shape[0]).mixture_mean(
            posterior.state_probs,
            axis=1,
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
        r"""Construct a Gaussian HMM from its parameters."""
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
    def via_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        *,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    ) -> typing.Self:
        """Initialize a Gaussian HMM by clustering the observations with K-means."""
        model = init_gaussian_via_kmeans(
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
    ) -> InferenceState[typing.Self, Posterior]:
        """Compute the exact posterior over states and the observation log likelihood."""
        inferred = _infer_exact_jit(
            self._model,
            observations,
        )

        return InferenceState(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            objective=inferred.objective,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fitted[typing.Self, Posterior]:
        """Fit model parameters by expectation-maximization."""
        fit = _fit_em_jit(
            self._model,
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return Fitted(
            model=self.__class__(fit.state.model),
            posterior=fit.state.posterior,
            objective_trace=fit.objective_trace,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonHMM:
    r"""
    Hidden Markov model with Poisson emissions.

    $$z_0\sim\operatorname{Categorical}(\pi),\qquad
    p(z_{t+1}=j\mid z_t=i)=P(i,j),$$

    $$y_t\mid z_t=k \sim \operatorname{Poisson}(\lambda_k).$$

    `_model.initial.dist.probs` stores $\pi$, `_model.transitions.dist.probs`
    stores $P$, and `states.log_rates` stores $\eta_k=\log\lambda_k$.
    """

    _model: Model[PoissonEmissions]

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    def permute_states(self, permutation: jax.Array) -> PoissonHMM:
        """Relabel discrete states by permutation."""
        return PoissonHMM(
            _model=self._model.permute_states(permutation),
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
        return self.states.broadcast(posterior.state_probs.shape[0]).mixture_mean(
            posterior.state_probs,
            axis=1,
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
    def via_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        *,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    ) -> typing.Self:
        """Initialize a Poisson HMM by clustering the observations with K-means."""
        model = init_poisson_via_kmeans(
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
    ) -> InferenceState[typing.Self, Posterior]:
        """Compute the exact posterior over states and the observation log likelihood."""
        inferred = infer_exact(
            self._model,
            observations,
        )

        return InferenceState(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            objective=inferred.objective,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fitted[typing.Self, Posterior]:
        """Fit model parameters by expectation-maximization."""
        fit = fit_em(
            self._model,
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return Fitted(
            model=self.__class__(fit.state.model),
            posterior=fit.state.posterior,
            objective_trace=fit.objective_trace,
        )
