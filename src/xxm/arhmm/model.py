"""AR-HMM facade classes for user-facing API."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.data import Dataset, DatasetCollection, Sequences
from xxm.core.dists.discrete import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.lengths import Lengths, LengthsLike, LengthsSpec

from .core import Model, Posterior, Sample
from .data import ARDataset, ARDatasetCollection
from .emissions import AREmissions, AREmissionsCollection
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


def _sampling_history(
    model: Model,
    lengths: Lengths,
    initial_history: jax.Array,
    session: int | None = None,
) -> jax.Array:

    return jnp.broadcast_to(
        initial_history,
        (
            *lengths.batch_shape,
            model.num_lags,
            model.observation_dim(session),
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

    def __post_init__(self) -> None:
        _ = self._model.batch_shape

    @property
    def model(self) -> Model[LinearGaussian]:
        """Underlying core model."""
        return self._model

    def observation_dim(self, session: int | None = None) -> int:
        return self._model.emissions.get(session).observation_dim

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self._model.num_lags

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> GaussianARHMM:
        """Relabel discrete states by permutation."""
        return GaussianARHMM(
            _model=self._model.permute_states(permutation),
        )

    def states(self, session: int | None = None) -> LinearGaussian:
        """State-conditional autoregressive Gaussian distributions."""
        return self._model.emissions.get(session).dist

    def states_conditional(
        self,
        data: DatasetCollection,
        *,
        session: int | None = None,
    ) -> Gaussian:
        """
        Conditional distributions for observations following the AR history.

        For an input sequence of length ``T``, the returned distributions have
        ``T - num_lags`` time steps.
        """
        dataset = data.get(session)
        prepared = ARDataset.from_dataset(dataset, self.num_lags)
        return self._model.emissions.get(session).conditional(
            prepared.predictors,
        )

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely state at each modeled time point."""
        return jnp.argmax(
            posterior.state_probs,
            axis=-1,
        )

    def observation_mean(
        self,
        data: DatasetCollection,
        posterior: Posterior,
        *,
        session: int | None = None,
    ) -> jax.Array:
        """Return the posterior mean observation conditional on the AR history."""
        return self.states_conditional(
            data,
            session=session,
        ).mixture_mean(
            posterior.state_probs,
            axis=-1,
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
                emissions=AREmissionsCollection(
                    (
                        AREmissions(
                            dist=LinearGaussian(
                                affine=Affine(
                                    coefficients=emission_coefficients,
                                    bias=emission_bias,
                                ),
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
        num_lags: int,
        *,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    ) -> typing.Self:
        """
        Initialize from autoregressive regression pairs.

        The first ``num_lags`` observations are used only as fixed conditioning
        history.
        """

        prepared = ARDatasetCollection.from_datasets(
            data,
            num_lags,
        )

        model = init_gaussian_via_kmeans(
            key=key,
            data=prepared,
            num_states=num_states,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

    def sample(
        self,
        key: jax.Array,
        steps: LengthsLike,
        *,
        initial_history: jax.Array | None = None,
        session: int | None = None,
    ) -> Sample:

        ls = LengthsSpec.from_params(steps, min_length=1)

        if initial_history is None:
            return self._model.sample(key, ls, session=session)

        return self._model.sample_continuation(
            key,
            ls,
            _sampling_history(self._model, ls.lengths, initial_history, session),
            session=session,
        )

    def infer(self, data: DatasetCollection) -> Inferred[typing.Self]:
        """
        Infer states conditional on the first ``num_lags`` observations.

        For an input sequence of length ``T``, the posterior has
        ``T - num_lags`` latent steps.
        """

        prepared = ARDatasetCollection.from_datasets(
            data,
            self.num_lags,
        )

        inferred = infer_exact(
            self._model,
            data=prepared,
        )

        return Inferred(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            data=data,
        )

    def fit(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[typing.Self]:
        """Fit the conditional AR-HMM with expectation maximization."""

        prepared = ARDatasetCollection.from_datasets(
            data,
            self.num_lags,
        )

        fit = fit_em(
            self._model,
            data=prepared,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            Inferred(
                model=self.__class__(fit.state.model),
                posterior=fit.state.posterior,
                data=data,
            ),
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

    def __post_init__(self) -> None:
        _ = self._model.batch_shape

    @property
    def model(self) -> Model[LinearPoisson]:
        """Underlying core model."""
        return self._model

    def observation_dim(self, session: int | None = None) -> int:
        return self._model.emissions.get(session).observation_dim

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self._model.num_states

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self._model.num_lags

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> PoissonARHMM:
        """Relabel discrete states by permutation."""
        return PoissonARHMM(
            _model=self._model.permute_states(permutation),
        )

    def states(self, session: int | None = None) -> LinearPoisson:
        """State-conditional autoregressive Poisson distributions."""
        return self._model.emissions.get(session).dist

    def states_conditional(
        self,
        data: DatasetCollection,
        *,
        session: int | None = None,
    ) -> Poisson:
        """
        Conditional distributions for observations following the AR history.

        For an input sequence of length ``T``, the returned distributions have
        ``T - num_lags`` time steps.
        """
        dataset = data.get(session)
        prepared = ARDataset.from_dataset(dataset, self.num_lags)
        return self._model.emissions.get(session).conditional(
            prepared.predictors,
        )

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely state at each modeled time point."""
        return jnp.argmax(
            posterior.state_probs,
            axis=-1,
        )

    def observation_mean(
        self,
        data: DatasetCollection,
        posterior: Posterior,
        *,
        session: int | None = None,
    ) -> jax.Array:
        """Return the posterior mean observation conditional on the AR history."""
        return self.states_conditional(
            data,
            session=session,
        ).mixture_mean(
            posterior.state_probs,
            axis=-1,
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
                emissions=AREmissionsCollection(
                    (
                        AREmissions(
                            dist=LinearPoisson(
                                affine=Affine(
                                    coefficients=emission_coefficients,
                                    bias=emission_bias,
                                )
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
            data=data,
            num_states=num_states,
            num_lags=num_lags,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

    def sample(
        self,
        key: jax.Array,
        steps: LengthsLike,
        *,
        initial_history: jax.Array | None = None,
        session: int | None = None,
    ) -> Sample:

        ls = LengthsSpec.from_params(steps, min_length=1)

        if initial_history is None:
            return self._model.sample(key, ls, session=session)

        return self._model.sample_continuation(
            key,
            ls,
            _sampling_history(self._model, ls.lengths, initial_history, session),
            session=session,
        )

    def infer(self, data: DatasetCollection) -> Inferred[typing.Self]:
        """
        Infer states conditional on the first ``num_lags`` observations.

        For an input sequence of length ``T``, the posterior has
        ``T - num_lags`` latent steps.
        """

        prepared = ARDatasetCollection.from_datasets(
            data,
            self.num_lags,
        )

        inferred = _infer_exact_jit(
            model=self._model,
            data=prepared,
        )

        return Inferred(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            data=data,
        )

    def fit(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[typing.Self]:
        """Fit the conditional AR-HMM with expectation maximization."""

        prepared = ARDatasetCollection.from_datasets(
            data,
            self.num_lags,
        )

        fit = _fit_em_jit(
            model=self._model,
            data=prepared,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            Inferred(
                model=self.__class__(fit.state.model),
                posterior=fit.state.posterior,
                data=data,
            ),
            objective_trace=fit.objective_trace,
        )


ARHMMFacadeT = typing.TypeVar('ARHMMFacadeT', GaussianARHMM, PoissonARHMM)


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(typing.Generic[ARHMMFacadeT]):
    """AR-HMM, pooled posterior, and source session collection."""

    model: ARHMMFacadeT
    posterior: Posterior
    data: DatasetCollection

    def __post_init__(self) -> None:
        expected_batch = (self.data.partition.total,)

        if self.posterior.batch_shape != expected_batch:
            raise ValueError(
                f'posterior batch must be {expected_batch}; '
                f'got {self.posterior.batch_shape}'
            )

        expected_steps = self.data.max_steps - self.model.num_lags
        if self.posterior.max_steps != expected_steps:
            raise ValueError('posterior must match the compact AR time dimension')

        if self.posterior.num_states != self.model.num_states:
            raise ValueError('posterior must match the model number of states')

    def _session(self, session: int | None) -> tuple[Posterior, Dataset]:
        """Select one session from the pooled posterior."""

        dataset = self.data.get(session)
        index = 0 if session is None else session

        posterior = self.posterior.select(self.data.partition.get_slice(index))

        return posterior, dataset

    def state_probs(self, session: int | None = None) -> Sequences:
        posterior, dataset = self._session(session)

        return Sequences.from_padded(
            posterior.state_probs,
            dataset.lengths.shift(-self.model.num_lags),
        )

    def observation_mean(self, session: int | None = None) -> Sequences:
        posterior, dataset = self._session(session)

        ar_data = ARDataset.from_dataset(
            dataset,
            self.model.num_lags,
        )

        emissions = self.model.model.emissions.get(session)
        emissions = emissions.broadcast(posterior.batch_shape)

        conditional = emissions.conditional(ar_data.safe_predictors())

        means = conditional.mixture_mean(
            posterior.state_probs,
            axis=-1,
        )

        return Sequences.from_padded(
            means,
            ar_data.lengths,
        )

    def most_likely_states(self, session: int | None = None) -> Sequences:
        posterior, dataset = self._session(session)

        return Sequences.from_padded(
            jnp.argmax(posterior.state_probs, axis=-1),
            dataset.lengths.shift(-self.model.num_lags),
        )

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        return type(self)(
            model=self.model.permute_states(permutation),
            posterior=self.posterior.permute_states(permutation),
            data=self.data,
        )

    def states_conditional(self, session: int | None = None) -> Poisson | Gaussian:
        return self.model.states_conditional(
            self.data,
            session=session,
        )


class Fit(typing.NamedTuple, typing.Generic[ARHMMFacadeT]):
    """Fitted result and objective trace, including the initial objective."""

    inferred: Inferred[ARHMMFacadeT]
    objective_trace: jax.Array
