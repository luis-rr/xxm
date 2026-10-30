"""AR-HMM facade classes for user-facing API."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core import facade as _facade
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.dists import io as dists_io
from xxm.core.dists.gaussian import LinearGaussian, TemporalGaussian
from xxm.core.dists.poisson import LinearPoisson, TemporalPoisson
from xxm.core.facade import Fit
from xxm.core.io import chains, hdf5
from xxm.core.io.hdf5 import PathLike, load_model_params
from xxm.core.latents.discrete import categorical_from_params
from xxm.core.lengths import Lengths, LengthsLike, LengthsSpec
from xxm.hmm.io import HMMParams

from .core import Model, Posterior, Sample
from .data import ARDataset, ARDatasetCollection
from .emissions import AREmissions, AREmissionsCollection, ConditionalDistT
from .inference import elbo, infer_exact
from .init import (
    DEFAULT_SELF_TRANSITION_PROB,
    init_gaussian_via_kmeans,
    init_poisson_via_kmeans,
)
from .learning import fit_em

_infer_exact_jit = jax.jit(infer_exact)
_elbo_jit = jax.jit(elbo)

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


def _observation_mean(
    emissions: AREmissions[typing.Any],
    predictors: jax.Array,  # (*B, T-L, L, D_y)
    state_probs: jax.Array,  # (*B, T-L, K)
) -> jax.Array:  # (*B, T-L, D_y)
    """Mix per-state conditional means by state probabilities."""
    emissions = emissions.broadcast(state_probs.shape[:-2])
    return batch.weighted_sum(
        emissions.conditional_mean(predictors),  # (*B, T-L, K, D_y)
        state_probs,
        state_probs.shape,
        axis=-1,
    )


class ARHMM(
    _facade.PermutableFacade[Model[ConditionalDistT]],
    typing.Generic[ConditionalDistT],
):
    r"""
    Autoregressive HMM facade shared by all conditional emission types.

    $$z_0\sim\operatorname{Categorical}(\pi),\qquad
    p(z_{s+1}=j\mid z_s=i)=P(i,j).$$

    For inference on $T$ observations, $s=0,\ldots,T-L-1$ is the compact
    posterior index: `posterior.state_probs[s]` corresponds to `observations[L+s]`.

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
        """Dimension $D_y$ of each observation."""
        return self.core.emissions.get(session).observation_dim

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.core.num_states

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self.core.num_lags

    def states_conditional(
        self,
        data: DatasetCollection,
        *,
        session: int | None = None,
    ) -> TemporalGaussian | TemporalPoisson:
        """Conditional distributions for observations following the AR history."""
        raise NotImplementedError

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
        prepared = ARDataset.from_dataset(data.get(session), self.num_lags)
        return _observation_mean(
            self.core.emissions.get(session),
            prepared.predictors,
            posterior.state_probs,
        )

    def to_params(self) -> HMMParams:
        """Return stored parameters, with one dictionary per emission session."""
        return {
            'initial_probs': self.core.initial.dist.probs,
            'transition_probs': self.core.transitions.dist.probs,
            'emissions': tuple(
                self._emissions_to_params(e) for e in self.core.emissions.emissions
            ),
        }

    @classmethod
    def _emissions_to_params(
        cls, emissions: AREmissions[ConditionalDistT]
    ) -> typing.Any:
        """Store one session's emissions; emission facades define this."""
        raise NotImplementedError

    @classmethod
    def _emissions_from_params(
        cls, params: typing.Any
    ) -> AREmissions[ConditionalDistT]:
        """Construct one session's emissions; emission facades define this."""
        raise NotImplementedError

    @classmethod
    def from_hdf5(cls, path: PathLike) -> typing.Self:
        """Restore the model or result without running inference."""
        params = load_model_params(path)
        return cls.from_params(typing.cast(HMMParams, params))

    @classmethod
    def from_params(cls, params: HMMParams) -> typing.Self:
        r"""Construct an AR-HMM from state, transition, and AR parameters.

        The coefficients have shape $(K,D_y,L,D_y)$ and the first $L$ observations
        serve as fixed conditioning history.
        """
        initial, transitions = categorical_from_params(
            params['initial_probs'], params['transition_probs']
        )

        return cls(
            core=Model(
                initial=initial,
                transitions=transitions,
                emissions=AREmissionsCollection(
                    tuple(
                        cls._emissions_from_params(item) for item in params['emissions']
                    )
                ),
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
        """Sample states and observations, optionally continuing a history."""

        ls = LengthsSpec.from_steps(steps, min_length=1)

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
            self.core,
            prepared,
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
            self.core,
            prepared,
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
class GaussianARHMM(ARHMM[LinearGaussian]):
    r"""
    Autoregressive HMM with Gaussian conditional emissions.

    $$y_{L+s}\mid z_s=k,y_{L+s-1:L+s-L}
    \sim\mathcal{N}\left(b_k+\sum_{\ell=1}^L A_{k,\ell}y_{L+s-\ell},R_k\right).$$

    `states.affine` stores the AR coefficients and biases; `states.covariance`
    stores $R_k$.
    """

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

        temporal = TemporalGaussian(
            dist=self.core.emissions.get(session).conditional(prepared.predictors),
            lengths=dataset.lengths.shift(-self.num_lags),
        )

        temporal.validate()

        return temporal

    @classmethod
    def _emissions_to_params(
        cls, emissions: AREmissions[LinearGaussian]
    ) -> dists_io.LinearGaussianParams:
        return dists_io.linear_gaussian_to_params(emissions.dist)

    @classmethod
    def _emissions_from_params(
        cls, params: dists_io.LinearGaussianParams
    ) -> AREmissions[LinearGaussian]:
        return AREmissions(dists_io.linear_gaussian_from_params(params))

    @classmethod
    def via_kmeans(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        num_states: int,
        num_lags: int,
        *,
        self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    ) -> GaussianARHMMCollection:
        """Initialize an independent model for every session with K-means."""

        return GaussianARHMMCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_gaussian_via_kmeans(
                    session_key,
                    ARDataset.from_dataset(dataset, num_lags),
                    num_states,
                    self_transition_prob=self_transition_prob,
                )
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonARHMM(ARHMM[LinearPoisson]):
    r"""
    Autoregressive HMM with Poisson conditional emissions.

    $$y_{L+s}\mid z_s=k,y_{L+s-1:L+s-L}
    \sim\operatorname{Poisson}(\lambda_{L+s}),\qquad
    \log\lambda_{L+s}=b_k+\sum_{\ell=1}^L A_{k,\ell}y_{L+s-\ell}.$$

    `states.affine` stores the AR log-rate coefficients and biases.
    """

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

        temporal = TemporalPoisson(
            dist=self.core.emissions.get(session).conditional(prepared.predictors),
            lengths=dataset.lengths.shift(-self.num_lags),
        )

        temporal.validate()

        return temporal

    @classmethod
    def _emissions_to_params(
        cls, emissions: AREmissions[LinearPoisson]
    ) -> dists_io.LinearPoissonParams:
        return dists_io.linear_poisson_to_params(emissions.dist)

    @classmethod
    def _emissions_from_params(
        cls, params: dists_io.LinearPoissonParams
    ) -> AREmissions[LinearPoisson]:
        return AREmissions(dists_io.linear_poisson_from_params(params))

    @classmethod
    def via_kmeans(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        num_states: int,
        num_lags: int,
        *,
        self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    ) -> PoissonARHMMCollection:
        """Initialize an independent model for every session with K-means."""

        return PoissonARHMMCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_poisson_via_kmeans(
                    session_key,
                    ARDataset.from_dataset(dataset, num_lags),
                    num_states,
                    self_transition_prob=self_transition_prob,
                )
            ),
        )


ARHMMFacadeT = typing.TypeVar('ARHMMFacadeT', bound='ARHMM[typing.Any]')


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
        facade_cls = _FACADES.get(model_type)
        if facade_cls is None:
            raise ValueError(f'Unsupported AR-HMM facade: {model_type!r}')
        model = facade_cls.from_params(params)
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

    def elbo(self) -> jax.Array:
        r"""Conditional log-likelihood, the ELBO at the exact posterior
        $q(z) = p(z \mid y)$ given the first $L$ observations:

        $$\log p(y_{L+1:T} \mid y_{1:L})
        = \mathbb{E}_{q(z)}[\log p(y_{L+1:T}, z \mid y_{1:L})] + \mathbb{H}[q(z)].$$
        """
        prepared = ARDatasetCollection.from_datasets(self.data, self.model.num_lags)
        return _elbo_jit(self.model.core, prepared, self.posterior)

    def observation_mean(self, session: int | None = None) -> Sequences:
        posterior, dataset = self._session(session)

        ar_data = ARDataset.from_dataset(
            dataset,
            self.model.num_lags,
        )

        means = _observation_mean(
            self.model.core.emissions.get(session),
            ar_data.safe_predictors(),
            posterior.state_probs,
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

    def infer(
        self,
        data: DatasetCollection,
        *,
        max_workers: int = 1,
    ) -> ARHMMInferredCollection[ARHMMFacadeT]:
        """Infer every session independently."""
        return self._infer_each(
            data,
            lambda model, subset: model.infer(subset),
            max_workers=max_workers,
        )

    def fit(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        progress: bool | str = False,
        max_workers: int = 1,
    ) -> _facade.FitCollection[ARHMMInferredCollection[ARHMMFacadeT]]:
        """Fit every session independently by EM."""
        return self._fit_each(
            data,
            lambda model, subset: model.fit(
                subset, num_iters=num_iters, progress=progress
            ),
            max_workers=max_workers,
        )


GaussianARHMMCollection = ARHMMCollection[GaussianARHMM]
PoissonARHMMCollection = ARHMMCollection[PoissonARHMM]

_FACADES: dict[str, type[ARHMM[typing.Any]]] = {
    f'xxm.arhmm.{facade.__name__}': facade for facade in (GaussianARHMM, PoissonARHMM)
}
