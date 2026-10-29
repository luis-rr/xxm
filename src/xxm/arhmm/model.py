"""AR-HMM facade classes for user-facing API."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core import facade as _facade
from xxm.core.affine import Affine
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.dists.discrete import Categorical
from xxm.core.dists.gaussian import LinearGaussian, TemporalGaussian
from xxm.core.dists.poisson import LinearPoisson, TemporalPoisson
from xxm.core.facade import Fit
from xxm.core.io import chains, hdf5
from xxm.core.io.hdf5 import PathLike, load_model_params
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.lengths import Lengths, LengthsLike, LengthsSpec
from xxm.hmm.io import HMMParams

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
class GaussianARHMM(_facade.PermutableFacade[Model[LinearGaussian]]):
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

    def observation_dim(self, session: int | None = None) -> int:
        return self.core.emissions.get(session).observation_dim

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.core.num_states

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self.core.num_lags

    def states(self, session: int | None = None) -> LinearGaussian:
        """State-conditional autoregressive Gaussian distributions."""
        return self.core.emissions.get(session).dist

    def states_conditional(
        self,
        data: DatasetCollection,
        *,
        session: int | None = None,
    ) -> TemporalGaussian:
        """
        Conditional distributions for observations following the AR history.

        For an input sequence of length ``T``, the returned distributions have
        ``T - num_lags`` time steps.
        """
        dataset = data.get(session)
        prepared = ARDataset.from_dataset(dataset, self.num_lags)

        conditional = self.core.emissions.get(session).conditional(
            prepared.predictors,
        )

        temporal = TemporalGaussian(
            dist=conditional,
            lengths=dataset.lengths.shift(-self.num_lags),
        )

        temporal.validate()

        return temporal

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
        ).dist.mixture_mean(
            posterior.state_probs,
            axis=-1,
        )

    def to_params(self) -> HMMParams:
        """Return stored parameters, with one dictionary per emission session."""
        return {
            'initial_probs': self.core.initial.dist.probs,
            'transition_probs': self.core.transitions.dist.probs,
            'emissions': tuple(
                {
                    'coefficients': e.dist.affine.coefficients,
                    'bias': e.dist.affine.bias,
                    'covariance': e.dist.covariance,
                }
                for e in self.core.emissions.emissions
            ),
        }

    @classmethod
    def from_hdf5(cls, path: PathLike) -> typing.Self:
        """Restore the model or result without running inference."""
        params = load_model_params(path)
        return cls.from_params(**typing.cast(HMMParams, params))

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emissions: tuple[dict[str, jax.Array], ...],
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
            core=Model(
                initial=initial,
                transitions=transitions,
                emissions=AREmissionsCollection(
                    tuple(
                        AREmissions(
                            LinearGaussian(
                                affine=Affine(item['coefficients'], item['bias']),
                                covariance=item['covariance'],
                            )
                        )
                        for item in emissions
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
    ) -> GaussianARHMMCollection:
        """Initialize an independent model for every session."""
        return GaussianARHMMCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_gaussian_via_kmeans(
                    key=session_key,
                    data=ARDatasetCollection.from_datasets(
                        DatasetCollection((dataset,)),
                        num_lags,
                    ),
                    num_states=num_states,
                    self_transition_prob=self_transition_prob,
                )
            ),
        )

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
            return self.core.sample(key, ls, session=session)

        return self.core.sample_continuation(
            key,
            ls,
            _sampling_history(self.core, ls.lengths, initial_history, session),
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
            self.core,
            data=prepared,
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
    ) -> Fit[Inferred[typing.Self]]:
        """Fit the conditional AR-HMM with expectation maximization."""

        prepared = ARDatasetCollection.from_datasets(
            data,
            self.num_lags,
        )

        fit = fit_em(
            self.core,
            data=prepared,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            Inferred(
                model=self._with_core(fit.state.model),
                posterior=fit.state.posterior,
                data=data,
            ),
            objective_trace=fit.objective_trace,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonARHMM(_facade.PermutableFacade[Model[LinearPoisson]]):
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

    def observation_dim(self, session: int | None = None) -> int:
        return self.core.emissions.get(session).observation_dim

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.core.num_states

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self.core.num_lags

    def states(self, session: int | None = None) -> LinearPoisson:
        """State-conditional autoregressive Poisson distributions."""
        return self.core.emissions.get(session).dist

    def states_conditional(
        self,
        data: DatasetCollection,
        *,
        session: int | None = None,
    ) -> TemporalPoisson:
        """
        Conditional distributions for observations following the AR history.

        For an input sequence of length ``T``, the returned distributions have
        ``T - num_lags`` time steps.
        """
        dataset = data.get(session)
        prepared = ARDataset.from_dataset(dataset, self.num_lags)
        conditional = self.core.emissions.get(session).conditional(
            prepared.predictors,
        )

        temporal = TemporalPoisson(
            dist=conditional,
            lengths=dataset.lengths.shift(-self.num_lags),
        )

        temporal.validate()

        return temporal

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
        ).dist.mixture_mean(
            posterior.state_probs,
            axis=-1,
        )

    def to_params(self) -> HMMParams:
        """Return stored parameters, with one dictionary per emission session."""
        return {
            'initial_probs': self.core.initial.dist.probs,
            'transition_probs': self.core.transitions.dist.probs,
            'emissions': tuple(
                {
                    'coefficients': e.dist.affine.coefficients,
                    'bias': e.dist.affine.bias,
                }
                for e in self.core.emissions.emissions
            ),
        }

    @classmethod
    def from_hdf5(cls, path: PathLike) -> typing.Self:
        """Restore the model or result without running inference."""
        params = load_model_params(path)
        return cls.from_params(**typing.cast(HMMParams, params))

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emissions: tuple[dict[str, jax.Array], ...],
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
            core=Model(
                initial=initial,
                transitions=transitions,
                emissions=AREmissionsCollection(
                    tuple(
                        AREmissions(
                            LinearPoisson(
                                affine=Affine(item['coefficients'], item['bias']),
                            )
                        )
                        for item in emissions
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
    ) -> PoissonARHMMCollection:
        """Initialize an independent model for every session."""
        return PoissonARHMMCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_poisson_via_kmeans(
                    key=session_key,
                    data=DatasetCollection((dataset,)),
                    num_states=num_states,
                    num_lags=num_lags,
                    self_transition_prob=self_transition_prob,
                )
            ),
        )

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
            return self.core.sample(key, ls, session=session)

        return self.core.sample_continuation(
            key,
            ls,
            _sampling_history(self.core, ls.lengths, initial_history, session),
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
            model=self.core,
            data=prepared,
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
    ) -> Fit[Inferred[typing.Self]]:
        """Fit the conditional AR-HMM with expectation maximization."""

        prepared = ARDatasetCollection.from_datasets(
            data,
            self.num_lags,
        )

        fit = _fit_em_jit(
            model=self.core,
            data=prepared,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            Inferred(
                model=self._with_core(fit.state.model),
                posterior=fit.state.posterior,
                data=data,
            ),
            objective_trace=fit.objective_trace,
        )


ARHMMFacadeT = typing.TypeVar('ARHMMFacadeT', GaussianARHMM, PoissonARHMM)


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(
    _facade.PermutableInferred[ARHMMFacadeT, Posterior],
    typing.Generic[ARHMMFacadeT],
):
    """AR-HMM, pooled posterior, and source session collection."""

    def to_hdf5(self, path: PathLike, *, overwrite: bool = False) -> None:
        hdf5.save_inferred(
            path,
            family='arhmm',
            model_type=f'xxm.arhmm.{type(self.model).__name__}',
            params=self.model.to_params(),
            posterior=chains.discrete_to_storage(self.posterior),
            data=self.data._to_storage(),
            overwrite=overwrite,
        )

    @classmethod
    def from_hdf5(cls, path: PathLike) -> typing.Self:
        """Restore the model or result without running inference."""
        model_type, raw, stored_posterior, stored_data = hdf5.load_inferred(path)
        data = DatasetCollection._from_storage(stored_data)
        params = typing.cast(HMMParams, raw)
        if model_type == 'xxm.arhmm.GaussianARHMM':
            model = GaussianARHMM.from_params(**params)
        elif model_type == 'xxm.arhmm.PoissonARHMM':
            model = PoissonARHMM.from_params(**params)
        else:
            raise ValueError(f'Unsupported AR-HMM facade: {model_type!r}')
        posterior = chains.discrete_from_storage(stored_posterior)
        return cls(
            model=typing.cast(ARHMMFacadeT, model), posterior=posterior, data=data
        )

    def __post_init__(self) -> None:
        super().__post_init__()

        expected_steps = self.data.max_steps - self.model.num_lags
        if self.posterior.max_steps != expected_steps:
            raise ValueError('posterior must match the compact AR time dimension')

        if self.posterior.num_states != self.model.num_states:
            raise ValueError('posterior must match the model number of states')

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

        emissions = self.model.core.emissions.get(session)
        emissions = emissions.broadcast(posterior.batch_shape)

        means = batch.weighted_sum(
            emissions.conditional_mean(ar_data.safe_predictors()),  # (..., T-L, K, D_y)
            posterior.state_probs,
            posterior.state_probs.shape,
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

    def states_conditional(
        self, session: int | None = None
    ) -> TemporalPoisson | TemporalGaussian:
        return self.model.states_conditional(
            self.data,
            session=session,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class ARHMMInferredCollection(
    _facade.PermutableInferredCollection[
        'ARHMMCollection[ARHMMFacadeT]',
        Posterior,
    ],
    typing.Generic[ARHMMFacadeT],
):
    """Independent AR-HMM posteriors with coordinated state permutations."""


@dataclasses.dataclass(frozen=True, eq=False)
class ARHMMCollection(
    _facade.FacadeCollection[
        ARHMMFacadeT,
        Inferred[ARHMMFacadeT],
        ARHMMInferredCollection[ARHMMFacadeT],
    ],
    typing.Generic[ARHMMFacadeT],
):
    """Independent AR-HMM facades with shared inference/fit orchestration."""

    @classmethod
    def _inferred_collection_cls(
        cls,
    ) -> type[ARHMMInferredCollection[ARHMMFacadeT]]:
        return ARHMMInferredCollection

    @classmethod
    def _inferred_cls(cls, model: ARHMMFacadeT) -> type[Inferred[ARHMMFacadeT]]:
        return Inferred


GaussianARHMMCollection = ARHMMCollection[GaussianARHMM]
PoissonARHMMCollection = ARHMMCollection[PoissonARHMM]
