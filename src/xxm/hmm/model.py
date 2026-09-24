"""HMM facade classes for user-facing API."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core.data import Dataset, Sequences
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson
from xxm.core.emissions.discrete import (
    GaussianEmissions,
    PoissonEmissions,
)
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

    def __post_init__(self) -> None:
        if self._model.batch_shape != ():
            raise ValueError('GaussianHMM facade requires an unbatched core model')

    @property
    def model(self) -> Model[GaussianEmissions]:
        """Underlying core model."""
        return self._model

    @property
    def observation_dim(self) -> int:
        return self._model.emissions.observation_dim

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
        data: Dataset,
        num_states: int,
        *,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    ) -> typing.Self:
        """Initialize a Gaussian HMM by clustering the observations with K-means."""
        model = init_gaussian_via_kmeans(
            key=key,
            data=data,
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
        data: Dataset,
    ) -> Inferred[GaussianHMM]:
        """Infer states for all Dataset sequences and retain their source structure."""
        inferred = _infer_exact_jit(
            self._model,
            data,
        )

        return Inferred(
            model=self,
            posterior=inferred.posterior,
            data=data,
        )

    def fit(
        self,
        data: Dataset,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[GaussianHMM]:
        """Fit model parameters by expectation-maximization."""
        fit = _fit_em_jit(
            self._model,
            data,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            inferred=Inferred(
                model=self.__class__(fit.state.model),
                posterior=fit.state.posterior,
                data=data,
            ),
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

    def __post_init__(self) -> None:
        if self._model.batch_shape != ():
            raise ValueError('PoissonHMM facade requires an unbatched core model')

    @property
    def model(self) -> Model[PoissonEmissions]:
        """Underlying core model."""
        return self._model

    @property
    def observation_dim(self) -> int:
        return self._model.emissions.observation_dim

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
        data: Dataset,
        num_states: int,
        *,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    ) -> typing.Self:
        """Initialize a Poisson HMM by clustering the observations with K-means."""
        model = init_poisson_via_kmeans(
            key=key,
            data=data,
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
        data: Dataset,
    ) -> Inferred[PoissonHMM]:
        """Infer states for all Dataset sequences and retain their source structure."""
        inferred = infer_exact(
            self._model,
            data,
        )

        return Inferred(
            model=self,
            posterior=inferred.posterior,
            data=data,
        )

    def fit(
        self,
        data: Dataset,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[PoissonHMM]:
        """Fit model parameters by expectation-maximization."""
        fit = fit_em(
            self._model,
            data,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            inferred=Inferred(
                model=self.__class__(fit.state.model),
                posterior=fit.state.posterior,
                data=data,
            ),
            objective_trace=fit.objective_trace,
        )


HMMFacadeT = typing.TypeVar('HMMFacadeT', GaussianHMM, PoissonHMM)


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(typing.Generic[HMMFacadeT]):
    """Singular HMM, posterior, and exact source Dataset."""

    model: HMMFacadeT
    posterior: Posterior
    data: Dataset

    def __post_init__(self) -> None:
        if self.posterior.batch_shape != self.data.batch_shape:
            raise ValueError('posterior batch must match the source dataset')
        if self.posterior.num_steps != self.data.num_steps:
            raise ValueError('posterior must match the padded time dimension')
        if self.posterior.num_states != self.model.num_states:
            raise ValueError('posterior must match the model number of states')

    def state_probs(self) -> Sequences:
        """State probabilities with the source batch shape and sequence lengths."""
        return Sequences.from_padded(
            self.posterior.state_probs,
            self.data.lengths,
        )

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean observation at each time point."""
        return self.model.model.emissions.broadcast(
            posterior.batch_shape
        ).observation_mean(
            posterior,
        )


class Fit(typing.NamedTuple, typing.Generic[HMMFacadeT]):
    """Fitted result and objective trace, including the initial objective."""

    inferred: Inferred[HMMFacadeT]
    objective_trace: jax.Array
