"""AR-HMM facade classes for user-facing API."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.inference import Fitted, InferenceState
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)

from .core import Model, Posterior
from .data import ARObservations
from .emissions import AREmissions
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
class GaussianARHMM:
    r"""
    Autoregressive HMM with Gaussian conditional emissions.

    $$z_0\sim\operatorname{Categorical}(\pi),\qquad
    p(z_{s+1}=j\mid z_s=i)=P(i,j),$$

    $$y_{L+s}\mid z_s=k,y_{L+s-1:L+s-L}
    \sim\mathcal{N}\left(b_k+\sum_{\ell=1}^L A_{k,\ell}y_{L+s-\ell},R_k\right).$$

    For inference on $T$ observations, $s=0,\ldots,T-L-1$ is the compact
    posterior index: `posterior.state_probs[s]` corresponds to `observations[L+s]`.
    `states.affine` stores the AR coefficients and biases; `states.covariance`
    stores $R_k$.

    The first ``num_lags`` observations of a fitted or inferred sequence are
    treated as fixed conditioning history and have no associated latent states.
    The first latent state is drawn independently from ``initial_probs`` and
    selects the regression generating the first observation after that history.

    Conditional predictors have shape $(L,D_y)$ and are ordered from most
    recent to oldest observation. Affine coefficients therefore have shape
    $(K,D_y,L,D_y)$.

    Autonomous sampling starts from a zero prehistory. An explicit
    ``initial_history`` may instead be supplied to generate a continuation from
    observed values.
    """

    _model: Model[LinearGaussian]

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    @property
    def output_dim(self) -> int:
        """Observation dimension $D_y$."""
        return self._model.emissions.output_dim

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self._model.emissions.num_lags

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> GaussianARHMM:
        """Relabel discrete states by permutation."""
        return GaussianARHMM(
            _model=self._model.permute_states(permutation),
        )

    @property
    def states(self) -> LinearGaussian:
        """State-conditional autoregressive Gaussian distributions."""
        return self._model.emissions.dist

    def states_conditional(
        self,
        observations: jax.Array,
    ) -> Gaussian:
        """
        Conditional distributions for observations following the AR history.

        For an input sequence of length ``T``, the returned distributions have
        ``T - num_lags`` time steps.
        """
        data = ARObservations.from_observations(observations, self.num_lags)
        return self._model.emissions.conditional(
            data.predictors,
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
            axis=1,
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

        The coefficients have shape $(K,D_y,L,D_y)$ and the first $L$ observations
        serve as fixed conditioning history.
        """
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            _model=Model(
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
    def via_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        num_lags: int,
        *,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    ) -> typing.Self:
        """
        Initialize from autoregressive regression pairs.

        The first ``num_lags`` observations are used only as fixed conditioning
        history.
        """
        model = init_gaussian_via_kmeans(
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
        Otherwise it must have shape $(L,D_y)$ and be ordered chronologically,
        from oldest to most recent. Only the newly generated observations are
        returned.
        """
        if initial_history is None:
            return self._model.sample(
                key,
                num_steps,
            )

        return self._model.sample_continuation(
            key,
            num_steps,
            initial_history,
        )

    def infer(
        self,
        observations: jax.Array,
    ) -> InferenceState[typing.Self, Posterior]:
        """
        Infer states conditional on the first ``num_lags`` observations.

        For an input sequence of length ``T``, the posterior has
        ``T - num_lags`` latent steps.
        """
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
        """Fit the conditional AR-HMM with expectation maximization."""
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonARHMM:
    r"""
    Autoregressive HMM with Poisson conditional emissions.

    $$z_0\sim\operatorname{Categorical}(\pi),\qquad
    p(z_{s+1}=j\mid z_s=i)=P(i,j),$$

    $$y_{L+s}\mid z_s=k,y_{L+s-1:L+s-L}
    \sim\operatorname{Poisson}(\lambda_{L+s}),\qquad
    \log\lambda_{L+s}=b_k+\sum_{\ell=1}^L A_{k,\ell}y_{L+s-\ell}.$$

    For inference on $T$ observations, $s=0,\ldots,T-L-1$ is the compact
    posterior index: `posterior.state_probs[s]` corresponds to `observations[L+s]`.
    `states.affine` stores the AR log-rate coefficients and biases.

    The first ``num_lags`` observations of a fitted or inferred sequence are
    treated as fixed conditioning history and have no associated latent states.
    The first latent state is drawn independently from ``initial_probs`` and
    selects the regression generating the first observation after that history.

    Conditional predictors have shape $(L,D_y)$ and are ordered from most
    recent to oldest observation. Affine coefficients therefore have shape
    $(K,D_y,L,D_y)$.

    Autonomous sampling starts from a zero prehistory. An explicit
    ``initial_history`` may instead be supplied to generate a continuation from
    observed values.
    """

    _model: Model[LinearPoisson]

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    @property
    def output_dim(self) -> int:
        """Observation dimension $D_y$."""
        return self._model.emissions.output_dim

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self._model.emissions.num_lags

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> PoissonARHMM:
        """Relabel discrete states by permutation."""
        return PoissonARHMM(
            _model=self._model.permute_states(permutation),
        )

    @property
    def states(self) -> LinearPoisson:
        """State-conditional autoregressive Poisson distributions."""
        return self._model.emissions.dist

    def states_conditional(
        self,
        observations: jax.Array,
    ) -> Poisson:
        """
        Conditional distributions for observations following the AR history.

        For an input sequence of length ``T``, the returned distributions have
        ``T - num_lags`` time steps.
        """
        data = ARObservations.from_observations(observations, self.num_lags)
        return self._model.emissions.conditional(
            data.predictors,
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
            axis=1,
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

        The coefficients have shape $(K,D_y,L,D_y)$ and the first $L$ observations
        serve as fixed conditioning history.
        """
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            _model=Model(
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
    def via_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        num_lags: int,
        *,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    ) -> typing.Self:
        """
        Initialize from autoregressive regression pairs.

        The first ``num_lags`` observations are used only as fixed conditioning
        history.
        """
        model = init_poisson_via_kmeans(
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
        Otherwise it must have shape $(L,D_y)$ and be ordered chronologically,
        from oldest to most recent. Only the newly generated observations are
        returned.
        """
        if initial_history is None:
            return self._model.sample(
                key,
                num_steps,
            )

        return self._model.sample_continuation(
            key,
            num_steps,
            initial_history,
        )

    def infer(self, observations: jax.Array) -> InferenceState[typing.Self, Posterior]:
        """
        Infer states conditional on the first ``num_lags`` observations.

        For an input sequence of length ``T``, the posterior has
        ``T - num_lags`` latent steps.
        """
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
        """Fit the conditional AR-HMM with expectation maximization."""
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
