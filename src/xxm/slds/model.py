"""Facade classes for switching linear dynamical systems."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core import facade as _facade
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.dists import io as dists_io
from xxm.core.dists.discrete import Categorical
from xxm.core.dists.gaussian import (
    FullGaussian,
    FullLinearGaussian,
    LinearGaussian,
    TemporalGaussian,
)
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.collection import ContinuousEmissionsCollection
from xxm.core.emissions.continuous import (
    EmissionsT,
    GaussianEmissions,
    LaplaceEmissionsT,
    PoissonEmissions,
    QuadraticEmissionsT,
)
from xxm.core.emissions.initialization import initial_latent_trajectory
from xxm.core.facade import Fit
from xxm.core.io import hdf5
from xxm.core.io.hdf5 import PathLike, load_model_params
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.latents.gaussian import StateConditionedGaussian
from xxm.core.latents.switching import GaussianLinearSwitchingDynamics
from xxm.core.lengths import LengthsLike, LengthsSpec
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.gaussian import CovarianceType
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import (
    Model,
    Posterior,
    Sample,
)
from .inference import elbo, infer_laplace, infer_variational
from .init import (
    DEFAULT_SELF_TRANSITION_PROB,
    init_gaussian_via_arhmm,
    init_gaussian_via_pca,
    init_poisson_via_arhmm,
    init_poisson_via_pca,
)
from .io import SLDSParams
from .learning import (
    fit_laplace_em,
    fit_variational_em,
)
from .serialization import posterior_from_storage, posterior_to_storage

_elbo_jit = jax.jit(elbo)
_infer_variational_jit = jax.jit(
    infer_variational,
    static_argnames=('num_iters',),
)

_infer_laplace_jit = jax.jit(
    infer_laplace,
    static_argnames=('num_iters',),
)

_fit_variational_em_jit = jax.jit(
    fit_variational_em,
    static_argnames=(
        'num_iters',
        'num_inference_iters',
        'progress',
    ),
)


_fit_laplace_em_jit = jax.jit(
    fit_laplace_em,
    static_argnames=(
        'num_iters',
        'num_inference_iters',
        'progress',
    ),
)


def _initial_latents(
    emissions: ContinuousEmissionsCollection[EmissionsT],
    data: DatasetCollection,
    initial_latents: jax.Array | None,
) -> jax.Array:
    """Initialize one trajectory per session, then pool sequences."""

    if initial_latents is not None:
        return initial_latents

    trajectories = []

    for emission, dataset in zip(
        emissions.emissions,
        data.datasets,
        strict=True,
    ):
        working_emission = emission.broadcast(dataset.batch_shape)

        trajectory = initial_latent_trajectory(
            working_emission,
            dataset.observations.values,
            dataset.observation_weights()[..., None],
        )

        trajectories.append(dataset.lengths.clear(trajectory))

    return jnp.concatenate(trajectories, axis=0)


def _sampling_inputs(
    model: Model,
    ls: LengthsSpec,
    inputs: jax.Array | None,
) -> Sequences:

    if inputs is not None:
        values = jnp.asarray(inputs)
    else:
        if model.dynamics.input_dim != 0:
            raise ValueError('controlled SLDS sampling requires explicit inputs')
        values = jnp.empty(
            (*ls.batch_shape, ls.max_steps, 0),
            dtype=model.dynamics.dist.affine.bias.dtype,
        )
    expected = (*ls.batch_shape, ls.max_steps, model.dynamics.input_dim)
    if values.shape != expected:
        raise ValueError(f'inputs must have shape {expected}; got {values.shape}')
    return Sequences.from_padded(values, ls.lengths)


def _conditioned_dynamics(model: Model, inputs: jax.Array | None) -> FullLinearGaussian:
    if inputs is None:
        if model.dynamics.input_dim != 0:
            raise ValueError('controlled SLDS dynamics require an input value')
        inputs = jnp.empty(
            (*model.batch_shape, 0), dtype=model.dynamics.dist.affine.bias.dtype
        )
    return model.dynamics.conditional(jnp.asarray(inputs))


class SLDS(
    _facade.PermutableFacade[Model[EmissionsT]],
    _facade.AlignableFacade[Model[EmissionsT]],
    typing.Generic[EmissionsT],
):
    r"""
    Switching linear dynamical system facade shared by all emission types.

    $$z_0 \sim \operatorname{Categorical}(\pi), \qquad
    z_t \mid z_{t-1} \sim \operatorname{Categorical}(P_{z_{t-1},:}),$$

    $$x_0 \mid z_0=k \sim \mathcal{N}(m_k,S_k), \qquad
    x_t \mid x_{t-1},z_t=k
    \sim \mathcal{N}(A_kx_{t-1}+B_k u_{t-1} + b_k,Q_k).$$

    For $t>0$, $z_t$ selects the dynamics generating $x_t$ from $x_{t-1}$.
    Emission-specific facades bind the emission type and its initialization.
    """

    @property
    def input_dim(self) -> int:
        """Dimension $D_u$ of known dynamics inputs."""
        return self.core.dynamics.input_dim

    @property
    def input_coefficients(self) -> jax.Array:
        """Known-input dynamics coefficients $B_k$."""
        return self.core.dynamics.input_coefficients

    @property
    def num_states(self) -> int:
        """Number of discrete dynamical states $K$."""
        return self.core.num_states

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self.core.dynamics.dist.output_dim

    def observation_dim(self, session: int | None = None) -> int:
        """Dimension $D_y$ of each observation."""
        return self.core.emissions.get(session).observation_dim

    def dynamics(self, inputs: jax.Array | None = None) -> FullLinearGaussian:
        """State-specific latent transitions conditioned on known inputs."""
        return _conditioned_dynamics(self.core, inputs)

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely discrete state at each time point."""
        return jnp.argmax(
            posterior.discrete.state_probs,
            axis=-1,
        )

    def to_params(self) -> SLDSParams:
        """Return stored parameters, with one dictionary per emission session."""
        return {
            'initial_probs': self.core.state_initial.dist.probs,
            'transition_probs': self.core.transitions.dist.probs,
            'latent_initial_means': self.core.latent_initial.dist.mean,
            'latent_initial_covariances': self.core.latent_initial.dist.covariance,
            'dynamics_coefficients': self.core.dynamics.latent_coefficients,
            'dynamics_input_coefficients': self.core.dynamics.input_coefficients,
            'dynamics_bias': self.core.dynamics.dist.affine.bias,
            'dynamics_covariances': self.core.dynamics.dist.covariance,
            'emissions': tuple(
                self._emissions_to_params(e) for e in self.core.emissions.emissions
            ),
        }

    @classmethod
    def _emissions_to_params(cls, emissions: EmissionsT) -> typing.Any:
        """Store one session's emissions; emission facades define this."""
        raise NotImplementedError

    @classmethod
    def _emissions_from_params(cls, params: typing.Any) -> EmissionsT:
        """Construct one session's emissions; emission facades define this."""
        raise NotImplementedError

    @classmethod
    def from_hdf5(cls, path: PathLike) -> typing.Self:
        """Restore the model or result without running inference."""
        params = load_model_params(path)
        return cls.from_params(typing.cast(SLDSParams, params))

    @classmethod
    def from_params(cls, params: SLDSParams) -> typing.Self:
        """Construct an SLDS from its stored parameters."""
        return cls(
            core=Model(
                state_initial=CategoricalInitial(
                    dist=Categorical(
                        probs=params['initial_probs'],
                    )
                ),
                transitions=CategoricalTransitions(
                    dist=Categorical(
                        probs=params['transition_probs'],
                    )
                ),
                latent_initial=StateConditionedGaussian(
                    dist=FullGaussian.from_moments(
                        params['latent_initial_means'],
                        params['latent_initial_covariances'],
                    )
                ),
                dynamics=GaussianLinearSwitchingDynamics.from_components(
                    latent_coefficients=params['dynamics_coefficients'],
                    input_coefficients=params.get('dynamics_input_coefficients'),
                    bias=params['dynamics_bias'],
                    covariance=params['dynamics_covariances'],
                ),
                emissions=ContinuousEmissionsCollection(
                    tuple(
                        cls._emissions_from_params(item) for item in params['emissions']
                    )
                ),
            )
        )

    def sample(
        self,
        key: jax.Array,
        steps: LengthsLike,
        *,
        inputs: jax.Array | None = None,
        session: int | None = None,
    ) -> Sample:
        r"""Sample discrete states, continuous latents, and observations."""

        ls = LengthsSpec.from_steps(steps, min_length=1)

        return self.core.sample(
            key,
            inputs=_sampling_inputs(self.core, ls, inputs),
            session=session,
        )


class VariationalSLDS(SLDS[QuadraticEmissionsT], typing.Generic[QuadraticEmissionsT]):
    """SLDS with conjugate emissions: structured variational inference and EM."""

    def infer(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        initial_latents: jax.Array | None = None,
    ) -> Inferred[typing.Self]:
        """Infer structured posteriors with a visibility-respecting initial trajectory."""
        initial_latents = _initial_latents(self.core.emissions, data, initial_latents)

        inferred = _infer_variational_jit(
            self.core,
            data,
            num_iters=num_iters,
            initial_latents=initial_latents,
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
        num_inference_iters: int,
        initial_latents: jax.Array | None = None,
        progress: bool | str = 'Variational EM',
    ) -> Fit[Inferred[typing.Self]]:
        """Fit model parameters with variational expectation-maximization."""
        initial_latents = _initial_latents(self.core.emissions, data, initial_latents)

        fit = _fit_variational_em_jit(
            self.core,
            data,
            num_iters=num_iters,
            num_inference_iters=num_inference_iters,
            initial_latents=initial_latents,
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


class LaplaceSLDS(SLDS[LaplaceEmissionsT], typing.Generic[LaplaceEmissionsT]):
    """SLDS with nonconjugate emissions: structured Laplace inference and EM."""

    def infer(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        initial_latents: jax.Array | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Inferred[typing.Self]:
        """Infer structured posteriors with a visibility-respecting initial trajectory."""
        initial_latents = _initial_latents(self.core.emissions, data, initial_latents)

        inferred = _infer_laplace_jit(
            self.core,
            data,
            num_iters=num_iters,
            initial_latents=initial_latents,
            params=laplace_params,
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
        num_inference_iters: int,
        initial_latents: jax.Array | None = None,
        progress: bool | str = 'Laplace EM',
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Fit[Inferred[typing.Self]]:
        """Fit model parameters with Laplace EM."""

        initial_latents = _initial_latents(self.core.emissions, data, initial_latents)

        fit = _fit_laplace_em_jit(
            self.core,
            data,
            num_iters=num_iters,
            num_inference_iters=num_inference_iters,
            laplace_params=laplace_params,
            initial_latents=initial_latents,
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
class GaussianSLDS(VariationalSLDS[GaussianEmissions]):
    r"""Switching linear dynamical system with Gaussian emissions.

    $$y_t \mid x_t \sim \mathcal{N}(Cx_t+d,R).$$
    """

    def emissions(self, session: int | None = None) -> LinearGaussian:
        """Linear-Gaussian observation distribution."""
        return self.core.emissions.get(session).dist

    @classmethod
    def _emissions_to_params(
        cls, emissions: GaussianEmissions
    ) -> dists_io.LinearGaussianParams:
        return dists_io.linear_gaussian_to_params(emissions.dist)

    @classmethod
    def _emissions_from_params(
        cls,
        params: dists_io.LinearGaussianParams,
    ) -> GaussianEmissions:
        return GaussianEmissions(dists_io.linear_gaussian_from_params(params))

    @classmethod
    def via_pca(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        num_states: int,
        latent_dim: int,
        *,
        self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
        covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        noise: CovarianceType = 'diagonal',
    ) -> GaussianSLDSCollection:
        """Initialize an independent model for every session from PCA latents."""

        return GaussianSLDSCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_gaussian_via_pca(
                    session_key,
                    dataset,
                    num_states,
                    latent_dim,
                    self_transition_prob=self_transition_prob,
                    covariance_floor=covariance_floor,
                    noise=noise,
                )
            ),
        )

    @classmethod
    def via_arhmm(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        num_states: int,
        latent_dim: int,
        *,
        num_arhmm_iters: int = 10,
        self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
        covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        progress: bool | str = 'AR-HMM',
        noise: CovarianceType = 'diagonal',
    ) -> GaussianSLDSCollection:
        """Initialize an independent model for every session from a fitted AR-HMM."""

        return GaussianSLDSCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_gaussian_via_arhmm(
                    session_key,
                    dataset,
                    num_states,
                    latent_dim,
                    num_arhmm_iters=num_arhmm_iters,
                    self_transition_prob=self_transition_prob,
                    covariance_floor=covariance_floor,
                    progress=progress,
                    noise=noise,
                )
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonSLDS(LaplaceSLDS[PoissonEmissions]):
    r"""Switching linear dynamical system with Poisson emissions.

    $$y_t \mid x_t \sim \operatorname{Poisson}(\exp(Cx_t+d)).$$
    """

    def emissions(self, session: int | None = None) -> LinearPoisson:
        """Linear-Poisson observation distribution in log-rate form."""
        return self.core.emissions.get(session).dist

    @classmethod
    def _emissions_to_params(
        cls, emissions: PoissonEmissions
    ) -> dists_io.LinearPoissonParams:
        return dists_io.linear_poisson_to_params(emissions.dist)

    @classmethod
    def _emissions_from_params(
        cls, params: dists_io.LinearPoissonParams
    ) -> PoissonEmissions:
        return PoissonEmissions(dists_io.linear_poisson_from_params(params))

    @classmethod
    def via_pca(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        num_states: int,
        latent_dim: int,
        *,
        self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
        covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
    ) -> PoissonSLDSCollection:
        """Initialize an independent model for every session from log-count PCA."""

        return PoissonSLDSCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_poisson_via_pca(
                    session_key,
                    dataset,
                    num_states,
                    latent_dim,
                    self_transition_prob=self_transition_prob,
                    covariance_floor=covariance_floor,
                    count_floor=count_floor,
                )
            ),
        )

    @classmethod
    def via_arhmm(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        num_states: int,
        latent_dim: int,
        *,
        num_arhmm_iters: int = 10,
        self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
        covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
        progress: bool | str = 'AR-HMM',
    ) -> PoissonSLDSCollection:
        """Initialize an independent model for every session from a fitted AR-HMM."""

        return PoissonSLDSCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda session_key, dataset: cls(
                init_poisson_via_arhmm(
                    session_key,
                    dataset,
                    num_states,
                    latent_dim,
                    num_arhmm_iters=num_arhmm_iters,
                    self_transition_prob=self_transition_prob,
                    covariance_floor=covariance_floor,
                    count_floor=count_floor,
                    progress=progress,
                )
            ),
        )


SLDSFacadeT = typing.TypeVar('SLDSFacadeT', bound='SLDS[typing.Any]')
VariationalSLDSFacadeT = typing.TypeVar(
    'VariationalSLDSFacadeT', bound='VariationalSLDS[typing.Any]'
)
LaplaceSLDSFacadeT = typing.TypeVar(
    'LaplaceSLDSFacadeT', bound='LaplaceSLDS[typing.Any]'
)


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(
    _facade.PermutableInferred[SLDSFacadeT, Posterior],
    _facade.AlignableInferred[SLDSFacadeT, Posterior],
    typing.Generic[SLDSFacadeT],
):
    """Singular SLDS, structured posterior, and the exact source data."""

    def to_hdf5(self, path: PathLike, *, overwrite: bool = False) -> None:
        hdf5.save_inferred(
            path,
            family='slds',
            model_type=f'xxm.slds.{type(self.model).__name__}',
            params=self.model.to_params(),
            posterior=posterior_to_storage(self.posterior),
            data=self.data._to_storage(),
            overwrite=overwrite,
        )

    @classmethod
    def from_hdf5(cls, path: PathLike) -> typing.Self:
        """Restore the model or result without running inference."""
        model_type, raw, stored_posterior, stored_data = hdf5.load_inferred(path)
        data = DatasetCollection._from_storage(stored_data)
        params = typing.cast(SLDSParams, raw)
        facade_cls = _FACADES.get(model_type)
        if facade_cls is None:
            raise ValueError(f'Unsupported SLDS facade: {model_type!r}')
        model = facade_cls.from_params(params)
        posterior = posterior_from_storage(stored_posterior)
        return cls(
            model=typing.cast(SLDSFacadeT, model), posterior=posterior, data=data
        )

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.posterior.max_steps != self.data.max_steps:
            raise ValueError('posterior must match the padded time dimension')

        if self.posterior.continuous.variable_dim != self.model.latent_dim:
            raise ValueError('posterior must match the model latent dimension')

        if self.posterior.discrete.num_states != self.model.num_states:
            raise ValueError('posterior must match the model number of states')

    def latent_posterior(self, session: int | None = None) -> TemporalGaussian:
        """Return marginal continuous latent posteriors with batch (N_s,)."""
        posterior, _ = self._session(session)
        continuous = posterior.continuous

        return TemporalGaussian(
            FullGaussian.from_moments(continuous.means, continuous.covariances),
            continuous.lengths,
        )

    def state_probs(self, session: int | None = None) -> Sequences:
        """Return posterior discrete-state probabilities."""
        posterior, dataset = self._session(session)

        return Sequences.from_padded(
            posterior.discrete.state_probs,
            dataset.lengths,
        )

    def elbo(self) -> jax.Array:
        r"""Evidence lower bound at the structured posterior $q(z)\,q(x)$:

        $$\mathcal{L} = \mathbb{E}_{q(z)\,q(x)}[\log p(y, z, x)]
        + \mathbb{H}[q(z)] + \mathbb{H}[q(x)] \le \log p(y).$$
        """
        return _elbo_jit(self.model.core, self.data, self.posterior)

    def observation_mean(self, session: int | None = None) -> Sequences:
        """Return posterior mean observations."""
        posterior, dataset = self._session(session)

        emissions = self.model.core.emissions.get(session)
        emissions = emissions.broadcast(posterior.batch_shape)

        return Sequences.from_padded(
            emissions.observation_mean(posterior.continuous),
            dataset.lengths,
        )

    def most_likely_states(self, session: int | None = None) -> Sequences:
        """Return the marginally most likely discrete state at each time point."""
        posterior, dataset = self._session(session)

        states = jnp.argmax(
            posterior.discrete.state_probs,
            axis=-1,
        )

        return Sequences.from_padded(
            states,
            dataset.lengths,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class SLDSInferredCollection(
    _facade.PermutableInferredCollection[
        'SLDSCollection[SLDSFacadeT]',
        Posterior,
    ],
    _facade.AlignableInferredCollection[
        'SLDSCollection[SLDSFacadeT]',
        Posterior,
    ],
    typing.Generic[SLDSFacadeT],
):
    """Independent SLDS posteriors with coordinated latent transformations."""


@dataclasses.dataclass(frozen=True, eq=False)
class SLDSCollection(
    _facade.FacadeCollection[
        SLDSFacadeT,
        Inferred[SLDSFacadeT],
        SLDSInferredCollection[SLDSFacadeT],
    ],
    typing.Generic[SLDSFacadeT],
):
    """Independent SLDS facades with shared inference/fit orchestration."""

    @classmethod
    def _inferred_collection_cls(
        cls,
    ) -> type[SLDSInferredCollection[SLDSFacadeT]]:
        return SLDSInferredCollection

    @classmethod
    def _inferred_cls(cls, model: SLDSFacadeT) -> type[Inferred[SLDSFacadeT]]:
        return Inferred


@dataclasses.dataclass(frozen=True, eq=False)
class VariationalSLDSCollection(
    SLDSCollection[VariationalSLDSFacadeT], typing.Generic[VariationalSLDSFacadeT]
):
    """Independent variational-inference SLDS facades."""

    def infer(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        initial_latents: tuple[jax.Array, ...] | None = None,
        max_workers: int = 1,
    ) -> SLDSInferredCollection[VariationalSLDSFacadeT]:
        """Infer every session independently, optionally from per-session warm starts."""
        return self._infer_each(
            data,
            lambda session_idx, model, subset: model.infer(
                subset,
                num_iters=num_iters,
                initial_latents=None
                if initial_latents is None
                else initial_latents[session_idx],
            ),
            max_workers=max_workers,
        )

    def fit(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        num_inference_iters: int,
        initial_latents: tuple[jax.Array, ...] | None = None,
        progress: bool | str = False,
        max_workers: int = 1,
    ) -> _facade.FitCollection[SLDSInferredCollection[VariationalSLDSFacadeT]]:
        """Fit every session independently by variational EM, optionally from per-session warm starts."""
        return self._fit_each(
            data,
            lambda session_idx, model, subset: model.fit(
                subset,
                num_iters=num_iters,
                num_inference_iters=num_inference_iters,
                initial_latents=None
                if initial_latents is None
                else initial_latents[session_idx],
                progress=progress,
            ),
            max_workers=max_workers,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class LaplaceSLDSCollection(
    SLDSCollection[LaplaceSLDSFacadeT], typing.Generic[LaplaceSLDSFacadeT]
):
    """Independent Laplace-inference SLDS facades."""

    def infer(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        initial_latents: tuple[jax.Array, ...] | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
        max_workers: int = 1,
    ) -> SLDSInferredCollection[LaplaceSLDSFacadeT]:
        """Infer every session independently, optionally from per-session warm starts."""
        return self._infer_each(
            data,
            lambda session_idx, model, subset: model.infer(
                subset,
                num_iters=num_iters,
                initial_latents=None
                if initial_latents is None
                else initial_latents[session_idx],
                laplace_params=laplace_params,
            ),
            max_workers=max_workers,
        )

    def fit(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        num_inference_iters: int,
        initial_latents: tuple[jax.Array, ...] | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
        progress: bool | str = False,
        max_workers: int = 1,
    ) -> _facade.FitCollection[SLDSInferredCollection[LaplaceSLDSFacadeT]]:
        """Fit every session independently by Laplace EM, optionally from per-session warm starts."""
        return self._fit_each(
            data,
            lambda session_idx, model, subset: model.fit(
                subset,
                num_iters=num_iters,
                num_inference_iters=num_inference_iters,
                initial_latents=None
                if initial_latents is None
                else initial_latents[session_idx],
                progress=progress,
                laplace_params=laplace_params,
            ),
            max_workers=max_workers,
        )


GaussianSLDSCollection = VariationalSLDSCollection[GaussianSLDS]
PoissonSLDSCollection = LaplaceSLDSCollection[PoissonSLDS]

_FACADES: dict[str, type[SLDS[typing.Any]]] = {
    f'xxm.slds.{facade.__name__}': facade for facade in (GaussianSLDS, PoissonSLDS)
}
