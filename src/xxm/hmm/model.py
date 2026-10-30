"""HMM facade classes for user-facing API."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core import facade as _facade
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.dists import io as dists_io
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson
from xxm.core.emissions.collection import DiscreteEmissionsCollection
from xxm.core.emissions.discrete import (
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.facade import Fit
from xxm.core.io import chains, hdf5
from xxm.core.io.hdf5 import PathLike, load_model_params
from xxm.core.latents.discrete import categorical_from_params
from xxm.core.lengths import LengthsLike, LengthsSpec

from .collection import collection_align, collection_compose
from .core import Model, Posterior, Sample
from .inference import infer_exact
from .init import (
    DEFAULT_SELF_TRANSITION_PROB,
    EmissionT,
    init_gaussian_via_kmeans,
    init_poisson_via_kmeans,
)
from .io import HMMParams
from .learning import fit_em

_infer_exact_jit = jax.jit(infer_exact)

_fit_em_jit = jax.jit(
    fit_em,
    static_argnames=(
        'num_iters',
        'progress',
    ),
)


class HMM(_facade.PermutableFacade[Model[EmissionT]], typing.Generic[EmissionT]):
    r"""Hidden Markov model facade shared by all emission types.

    $$z_0\sim\operatorname{Categorical}(\pi),\qquad
    p(z_{t+1}=j\mid z_t=i)=P(i,j).$$

    `core.initial.dist.probs` stores $\pi$ and `core.transitions.dist.probs`
    stores $P$. Emission-specific facades bind the emission type and its
    initialization.
    """

    def observation_dim(self, session: int | None = None) -> int:
        """Dimension $D_y$ of each observation."""
        return self.core.emissions.get(session).observation_dim

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.core.num_states

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely state at each time point."""
        return jnp.argmax(posterior.state_probs, axis=-1)

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
    def _emissions_to_params(cls, emissions: EmissionT) -> typing.Any:
        """Store one session's emissions; emission facades define this."""
        raise NotImplementedError

    @classmethod
    def _emissions_from_params(cls, params: typing.Any) -> EmissionT:
        """Construct one session's emissions; emission facades define this."""
        raise NotImplementedError

    @classmethod
    def from_hdf5(cls, path: PathLike) -> typing.Self:
        """Restore the model or result without running inference."""
        params = load_model_params(path)
        return cls.from_params(typing.cast(HMMParams, params))

    @classmethod
    def from_params(cls, params: HMMParams) -> typing.Self:
        r"""Construct a model from its parameters."""
        initial, transitions = categorical_from_params(
            params['initial_probs'], params['transition_probs']
        )

        return cls(
            core=Model(
                initial=initial,
                transitions=transitions,
                emissions=DiscreteEmissionsCollection(
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
        session: int | None = None,
    ) -> Sample:
        r"""Sample state and observation sequences with requested lengths."""
        ls = LengthsSpec.from_steps(steps, min_length=1)
        return self.core.sample(key, ls=ls, session=session)

    def infer(self, data: DatasetCollection) -> Inferred[typing.Self]:
        """Infer states for all Dataset sequences and retain their source structure."""
        inferred = _infer_exact_jit(
            self.core,
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
    ) -> Fit[Inferred[typing.Self]]:
        """Fit model parameters by expectation-maximization."""
        fit = _fit_em_jit(
            self.core,
            data,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            inferred=Inferred(
                model=self._with_core(fit.state.model),
                posterior=fit.state.posterior,
                data=data,
            ),
            objective_trace=fit.objective_trace,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianHMM(HMM[GaussianEmissions]):
    r"""Hidden Markov model with Gaussian state-conditional emissions.

    $$y_t\mid z_t=k \sim \mathcal{N}(\mu_k, \Sigma_k).$$
    """

    def states(self, session: int | None = None) -> Gaussian:
        """State-conditional emission distributions."""
        return self.core.emissions.get(session).dist

    @classmethod
    def _emissions_to_params(
        cls, emissions: GaussianEmissions
    ) -> dists_io.GaussianParams:
        return dists_io.gaussian_to_params(emissions.dist)

    @classmethod
    def _emissions_from_params(
        cls, params: dists_io.GaussianParams
    ) -> GaussianEmissions:
        return GaussianEmissions(dists_io.gaussian_from_params(params))

    @classmethod
    def via_kmeans(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        num_states: int,
        *,
        self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    ) -> GaussianHMMCollection:
        """Initialize an independent HMM for every session with weighted K-means."""

        return GaussianHMMCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_gaussian_via_kmeans(
                    session_key,
                    dataset,
                    num_states,
                    self_transition_prob=self_transition_prob,
                )
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonHMM(HMM[PoissonEmissions]):
    r"""Hidden Markov model with Poisson emissions.

    $$y_t\mid z_t=k \sim \operatorname{Poisson}(\lambda_k).$$

    `states.log_rates` stores $\eta_k=\log\lambda_k$.
    """

    def states(self, session: int | None = None) -> Poisson:
        """State-conditional Poisson emission distributions."""
        return self.core.emissions.get(session).dist

    @classmethod
    def _emissions_to_params(
        cls, emissions: PoissonEmissions
    ) -> dists_io.PoissonParams:
        return dists_io.poisson_to_params(emissions.dist)

    @classmethod
    def _emissions_from_params(cls, params: dists_io.PoissonParams) -> PoissonEmissions:
        return PoissonEmissions(dists_io.poisson_from_params(params))

    @classmethod
    def via_kmeans(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        num_states: int,
        *,
        self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    ) -> PoissonHMMCollection:
        """Initialize an independent HMM for every session with weighted K-means."""

        return PoissonHMMCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_poisson_via_kmeans(
                    session_key,
                    dataset,
                    num_states,
                    self_transition_prob=self_transition_prob,
                )
            ),
        )


HMMFacadeT = typing.TypeVar('HMMFacadeT', bound='HMM[typing.Any]')


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(
    _facade.PermutableInferred[HMMFacadeT, Posterior],
    typing.Generic[HMMFacadeT],
):
    """HMM, pooled posterior, and source session collection."""

    def to_hdf5(self, path: PathLike, *, overwrite: bool = False) -> None:
        hdf5.save_inferred(
            path,
            family='hmm',
            model_type=f'xxm.hmm.{type(self.model).__name__}',
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
            raise ValueError(f'Unsupported HMM facade: {model_type!r}')
        model = facade_cls.from_params(params)
        posterior = chains.discrete_from_storage(stored_posterior)
        return cls(model=typing.cast(HMMFacadeT, model), posterior=posterior, data=data)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.posterior.max_steps != self.data.max_steps:
            raise ValueError('posterior must match the padded time dimension')

        if self.posterior.num_states != self.model.num_states:
            raise ValueError('posterior must match the model number of states')

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

        emissions = self.model.core.emissions.get(session)
        emissions = emissions.broadcast(posterior.batch_shape)

        means = emissions.observation_mean(posterior)

        return Sequences.from_padded(
            means,
            dataset.lengths,
        )

    def most_likely_states(self, session: int | None = None) -> Sequences:
        """Return marginally most likely states for one session."""

        posterior, dataset = self._session(session)

        return Sequences.from_padded(
            jnp.argmax(posterior.state_probs, axis=-1),
            dataset.lengths,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class HMMInferredCollection(
    _facade.PermutableInferredCollection[
        'HMMCollection[HMMFacadeT]',
        Posterior,
    ],
    typing.Generic[HMMFacadeT],
):
    def align_states(
        self,
        key: jax.Array,
        *,
        num_restarts: int | None = 10,
    ) -> typing.Self:
        if len(self.posteriors) == 1:
            return self

        permutations = collection_align(
            key,
            self.posteriors,
            num_restarts=num_restarts,
        )

        return self.permute_states(permutations)

    def compose(self) -> HMMFacadeT:
        """Pool sufficient statistics from already aligned sessions."""
        if len(self.posteriors) == 1:
            return self.models.get()

        cores = typing.cast(
            tuple[Model[typing.Any], ...],
            tuple(model.core for model in self.models.models),
        )

        combined = collection_compose(cores, self.posteriors)

        return self.models.get(0)._with_core(combined)

    def combine(
        self,
        key: jax.Array,
        *,
        num_restarts: int | None = 10,
    ) -> HMMFacadeT:
        return self.align_states(
            key,
            num_restarts=num_restarts,
        ).compose()


@dataclasses.dataclass(frozen=True, eq=False)
class HMMCollection(
    _facade.FacadeCollection[
        HMMFacadeT,
        Inferred[HMMFacadeT],
        HMMInferredCollection[HMMFacadeT],
    ],
    typing.Generic[HMMFacadeT],
):
    """Independent HMM facades with shared inference/fit orchestration."""

    @classmethod
    def _inferred_collection_cls(
        cls,
    ) -> type[HMMInferredCollection[HMMFacadeT]]:
        return HMMInferredCollection

    @classmethod
    def _inferred_cls(cls, model: HMMFacadeT) -> type[Inferred[HMMFacadeT]]:
        return Inferred

    def infer(
        self,
        data: DatasetCollection,
        *,
        max_workers: int = 1,
    ) -> HMMInferredCollection[HMMFacadeT]:
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
    ) -> _facade.FitCollection[HMMInferredCollection[HMMFacadeT]]:
        """Fit every session independently by EM."""
        return self._fit_each(
            data,
            lambda model, subset: model.fit(
                subset, num_iters=num_iters, progress=progress
            ),
            max_workers=max_workers,
        )


GaussianHMMCollection = HMMCollection[GaussianHMM]
PoissonHMMCollection = HMMCollection[PoissonHMM]

_FACADES: dict[str, type[HMM[typing.Any]]] = {
    f'xxm.hmm.{facade.__name__}': facade for facade in (GaussianHMM, PoissonHMM)
}
