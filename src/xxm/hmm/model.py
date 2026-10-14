"""HMM facade classes for user-facing API."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core.data import Dataset, DatasetCollection, Sequences
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson
from xxm.core.emissions.collection import DiscreteEmissionsCollection
from xxm.core.emissions.discrete import (
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.lengths import LengthsLike, LengthsSpec

from .core import Model, Posterior, Sample
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

    def observation_dim(self, session: int | None = None) -> int:
        return self._model.emissions.get(session).observation_dim

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    def permute_states(self, permutation: jax.Array) -> GaussianHMM:
        """Relabel discrete states by permutation."""
        return GaussianHMM(
            _model=self._model.permute_states(permutation),
        )

    def states(self, session: int | None = None) -> Gaussian:
        """State-conditional emission distributions."""
        return self._model.emissions.get(session).dist

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
                emissions=DiscreteEmissionsCollection(
                    (
                        GaussianEmissions(
                            dist=Gaussian(
                                mean=emission_means,
                                covariance=emission_covariances,
                            )
                        ),
                    )
                ),
            ),
        )

    @classmethod
    def via_kmeans(
        cls,
        key: jax.Array,
        data: DatasetCollection,
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

    def sample(
        self,
        key: jax.Array,
        steps: LengthsLike,
        *,
        session: int | None = None,
    ) -> Sample:
        r"""Sample state and observation sequences with requested lengths."""
        ls = LengthsSpec.from_params(steps, min_length=1)
        return self._model.sample(key, ls=ls, session=session)

    def infer(
        self,
        data: DatasetCollection,
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
        data: DatasetCollection,
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

    def observation_dim(self, session: int | None = None) -> int:
        return self._model.emissions.get(session).observation_dim

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    def permute_states(self, permutation: jax.Array) -> PoissonHMM:
        """Relabel discrete states by permutation."""
        return PoissonHMM(
            _model=self._model.permute_states(permutation),
        )

    def states(self, session: int | None = None) -> Poisson:
        """State-conditional Poisson emission distributions."""
        return self._model.emissions.get(session).dist

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
                emissions=DiscreteEmissionsCollection(
                    (
                        PoissonEmissions(
                            dist=Poisson(
                                log_rates=emission_log_rates,
                            )
                        ),
                    )
                ),
            ),
        )

    @classmethod
    def via_kmeans(
        cls,
        key: jax.Array,
        data: DatasetCollection,
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

    def sample(
        self,
        key: jax.Array,
        steps: LengthsLike,
        *,
        session: int | None = None,
    ) -> Sample:
        r"""Sample state and observation sequences with requested lengths."""
        ls = LengthsSpec.from_params(steps, min_length=1)
        return self._model.sample(key, ls=ls, session=session)

    def infer(
        self,
        data: DatasetCollection,
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
        data: DatasetCollection,
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
    """HMM, pooled posterior, and source session collection."""

    model: HMMFacadeT
    posterior: Posterior
    data: DatasetCollection

    def __post_init__(self) -> None:
        expected_batch = (self.data.partition.total,)

        if self.posterior.batch_shape != expected_batch:
            raise ValueError(
                f'posterior batch must be {expected_batch}; '
                f'got {self.posterior.batch_shape}'
            )

        if self.posterior.max_steps != self.data.max_steps:
            raise ValueError('posterior must match the padded time dimension')

        if self.posterior.num_states != self.model.num_states:
            raise ValueError('posterior must match the model number of states')

    def _session(self, session: int | None) -> tuple[Posterior, Dataset]:
        """Select one session from the pooled posterior."""

        dataset = self.data.get(session)
        index = 0 if session is None else session

        posterior = self.posterior.select(self.data.partition.get_slice(index))

        return posterior, dataset

    def state_probs(self, session: int | None = None) -> Sequences:
        """Return state probabilities for one session."""

        posterior, dataset = self._session(session)

        return Sequences.from_padded(
            posterior.state_probs,
            dataset.lengths,
        )

    def observation_mean(self, session: int | None = None) -> Sequences:
        """Return posterior mean observations for one session."""

        posterior, dataset = self._session(session)

        emissions = self.model.model.emissions.get(session)
        emissions = emissions.broadcast(posterior.batch_shape)

        means = emissions.observation_mean(posterior)

        return Sequences.from_padded(
            means,
            dataset.lengths,
        )

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """Relabel discrete states by permutation."""

        return type(self)(
            model=self.model.permute_states(permutation),
            posterior=self.posterior.permute_states(permutation),
            data=self.data,
        )

    def most_likely_states(self, session: int | None = None) -> Sequences:
        """Return marginally most likely states for one session."""

        posterior, dataset = self._session(session)

        return Sequences.from_padded(
            jnp.argmax(posterior.state_probs, axis=-1),
            dataset.lengths,
        )


class Fit(typing.NamedTuple, typing.Generic[HMMFacadeT]):
    """Fitted result and objective trace, including the initial objective."""

    inferred: Inferred[HMMFacadeT]
    objective_trace: jax.Array
