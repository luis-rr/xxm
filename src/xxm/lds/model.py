"""Facade classes for linear dynamical systems."""

from __future__ import annotations

import dataclasses
import typing

import jax
from jax import numpy as jnp

from xxm.core import facade as _facade
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.dists.gaussian import Gaussian, LinearGaussian, TemporalGaussian
from xxm.core.dists.gaussian import GaussianChainMarginals as Posterior
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.collection import ContinuousEmissionsCollection
from xxm.core.emissions.continuous import (
    EmissionsT,
    GaussianEmissions,
    LaplaceEmissionsT,
    PoissonEmissions,
    QuadraticEmissionsT,
)
from xxm.core.facade import Fit
from xxm.core.io import chains, hdf5
from xxm.core.io.hdf5 import PathLike, load_model_params
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.core.lengths import LengthsLike, LengthsSpec
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model, Sample
from .inference import elbo, infer_exact, infer_laplace
from .init import init_gaussian_via_pca, init_poisson_via_pca
from .io import LDSParams
from .learning import fit_em, fit_laplace_em

_infer_exact_jit = jax.jit(infer_exact)
_infer_laplace_jit = jax.jit(infer_laplace)

_fit_em_jit = jax.jit(
    fit_em,
    static_argnames=(
        'num_iters',
        'progress',
    ),
)

_fit_laplace_em_jit = jax.jit(
    fit_laplace_em,
    static_argnames=(
        'num_iters',
        'progress',
    ),
)

_elbo_jit = jax.jit(elbo)


def _sampling_inputs(
    model: Model,
    ls: LengthsSpec,
    inputs: jax.Array | None,
) -> Sequences:

    if inputs is not None:
        values = jnp.asarray(inputs)
    else:
        if model.dynamics.input_dim != 0:
            raise ValueError('controlled LDS sampling requires explicit inputs')
        values = jnp.empty(
            (*ls.batch_shape, ls.max_steps, 0),
            dtype=model.dynamics.dist.affine.bias.dtype,
        )
    expected = (*ls.batch_shape, ls.max_steps, model.dynamics.input_dim)
    if values.shape != expected:
        raise ValueError(f'inputs must have shape {expected}; got {values.shape}')
    return Sequences.from_padded(values, ls.lengths)


def _conditioned_dynamics(
    model: Model,
    inputs: jax.Array | None,
) -> LinearGaussian:
    input_dim = model.dynamics.input_dim
    batch_shape = model.dynamics.batch_shape

    if inputs is None:
        if input_dim != 0:
            raise ValueError('controlled LDS dynamics require an input value')

        inputs = jnp.empty(
            (*batch_shape, 0),
            dtype=model.dynamics.dist.affine.bias.dtype,
        )
    else:
        inputs = jnp.asarray(inputs)

    return model.dynamics.conditional(inputs)


class LDS(_facade.AlignableFacade[Model[EmissionsT]], typing.Generic[EmissionsT]):
    r"""Linear dynamical system facade shared by all emission types.

    $$x_0 \sim \mathcal{N}(m_0,S_0),$$

    $$x_t\mid x_{t-1},u_t
    \sim \mathcal{N}(Ax_{t-1}+Bu_t+b,Q).$$

    Emission-specific facades bind the emission type and its initialization.
    """

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self.core.dynamics.latent_dim

    def observation_dim(self, session: int | None = None) -> int:
        """Dimension $D_y$ of each observation."""
        return self.core.emissions.get(session).observation_dim

    @property
    def input_dim(self) -> int:
        """Dimension $D_u$ of known dynamics inputs."""
        return self.core.dynamics.input_dim

    @property
    def input_coefficients(self) -> jax.Array:
        """Known-input dynamics coefficients $B$."""
        return self.core.dynamics.input_coefficients

    def dynamics(self, inputs: jax.Array | None = None) -> LinearGaussian:
        r"""Return $p(x_t\mid x_{t-1},u_t)$ conditioned on a known input."""
        return _conditioned_dynamics(self.core, inputs)

    def to_params(self) -> LDSParams:
        """Return stored parameters, with one dictionary per emission session."""
        return {
            'initial_mean': self.core.initial.dist.mean,
            'initial_covariance': self.core.initial.dist.covariance,
            'dynamics_coefficients': self.core.dynamics.latent_coefficients,
            'dynamics_input_coefficients': self.core.dynamics.input_coefficients,
            'dynamics_bias': self.core.dynamics.dist.affine.bias,
            'dynamics_covariance': self.core.dynamics.dist.covariance,
            'emissions': tuple(e.to_params() for e in self.core.emissions.emissions),
        }

    @classmethod
    def _emissions_from_params(cls, params: dict[str, jax.Array]) -> EmissionsT:
        """Construct one session's emissions; emission facades define this."""
        raise NotImplementedError

    @classmethod
    def from_hdf5(cls, path: PathLike) -> typing.Self:
        """Restore the model or result without running inference."""
        params = load_model_params(path)
        return cls.from_params(**typing.cast(LDSParams, params))

    @classmethod
    def from_params(
        cls,
        *,
        initial_mean: jax.Array,
        initial_covariance: jax.Array,
        dynamics_coefficients: jax.Array,
        dynamics_bias: jax.Array,
        dynamics_covariance: jax.Array,
        emissions: tuple[dict[str, jax.Array], ...],
        dynamics_input_coefficients: jax.Array | None = None,
    ) -> typing.Self:
        """Construct an LDS from explicit parameters."""
        initial = GaussianInitial(
            dist=Gaussian.from_moments(initial_mean, initial_covariance)
        )
        dynamics = GaussianLinearDynamics.from_params(
            latent_coefficients=dynamics_coefficients,
            input_coefficients=dynamics_input_coefficients,
            bias=dynamics_bias,
            covariance=dynamics_covariance,
        )

        return cls(
            core=Model(
                initial=initial,
                dynamics=dynamics,
                emissions=ContinuousEmissionsCollection(
                    tuple(cls._emissions_from_params(item) for item in emissions)
                ),
            ),
        )

    def sample(
        self,
        key: jax.Array,
        steps: LengthsLike,
        *,
        inputs: jax.Array | None = None,
        session: int | None = None,
    ) -> Sample:
        r"""Sample latent states and observations."""

        ls = LengthsSpec.from_params(steps, min_length=1)
        input_sequences = _sampling_inputs(self.core, ls, inputs)

        return self.core.sample(key, inputs=input_sequences, session=session)


class ExactLDS(LDS[QuadraticEmissionsT], typing.Generic[QuadraticEmissionsT]):
    """LDS with conjugate emissions: exact inference and EM."""

    def infer(
        self,
        data: DatasetCollection,
    ) -> Inferred[typing.Self]:
        """Compute exact posteriors with the Dataset's batch shape."""

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
        """Fit shared parameters across all Dataset batch entries by EM."""
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


class LaplaceLDS(LDS[LaplaceEmissionsT], typing.Generic[LaplaceEmissionsT]):
    """LDS with nonconjugate emissions: Laplace inference and Laplace EM."""

    def infer(
        self,
        data: DatasetCollection,
        *,
        initial_latents: jax.Array | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Inferred[typing.Self]:
        """Compute Laplace posteriors with the Dataset's batch shape.

        Optional initial latents have shape `(*data.batch_shape, T, D_x)`.
        """
        inferred = _infer_laplace_jit(
            self.core,
            data,
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
        progress: bool | str = 'Laplace EM',
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Fit[Inferred[typing.Self]]:
        """Fit shared parameters across Dataset batch entries by Laplace EM."""
        fit = _fit_laplace_em_jit(
            self.core,
            data,
            num_iters=num_iters,
            progress=progress,
            laplace_params=laplace_params,
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
class GaussianLDS(ExactLDS[GaussianEmissions]):
    r"""Linear dynamical system with Gaussian emissions.

    $$y_t\mid x_t \sim \mathcal{N}(Cx_t+d,R).$$
    """

    def emissions(self, session: int | None = None) -> LinearGaussian:
        """Linear-Gaussian observation distribution."""
        return self.core.emissions.get(session).dist

    @classmethod
    def _emissions_from_params(cls, params: dict[str, jax.Array]) -> GaussianEmissions:
        return GaussianEmissions.from_params(**params)

    @classmethod
    def via_pca(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        latent_dim: int,
        *,
        ridge: float = gaussian_fit.DEFAULT_RIDGE,
        covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> GaussianLDSCollection:
        """Initialize an independent model for every session from PCA latents."""
        return GaussianLDSCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda _session_key, dataset: cls(
                init_gaussian_via_pca(
                    dataset,
                    latent_dim,
                    ridge=ridge,
                    covariance_floor=covariance_floor,
                )
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonLDS(LaplaceLDS[PoissonEmissions]):
    r"""Linear dynamical system with Poisson emissions.

    $$y_t\mid x_t \sim \operatorname{Poisson}(\exp(Cx_t+d)).$$
    """

    def emissions(self, session: int | None = None) -> LinearPoisson:
        """Linear-Poisson observation distribution in log-rate form."""
        return self.core.emissions.get(session).dist

    @classmethod
    def _emissions_from_params(cls, params: dict[str, jax.Array]) -> PoissonEmissions:
        return PoissonEmissions.from_params(**params)

    @classmethod
    def via_pca(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        latent_dim: int,
        *,
        ridge: float = gaussian_fit.DEFAULT_RIDGE,
        covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
    ) -> PoissonLDSCollection:
        """Initialize an independent model for every session from log-count PCA."""
        return PoissonLDSCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda _session_key, dataset: cls(
                init_poisson_via_pca(
                    dataset,
                    latent_dim,
                    ridge=ridge,
                    covariance_floor=covariance_floor,
                    count_floor=count_floor,
                )
            ),
        )


LDSFacadeT = typing.TypeVar('LDSFacadeT', bound='LDS[typing.Any]')
ExactLDSFacadeT = typing.TypeVar('ExactLDSFacadeT', bound='ExactLDS[typing.Any]')
LaplaceLDSFacadeT = typing.TypeVar('LaplaceLDSFacadeT', bound='LaplaceLDS[typing.Any]')


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(
    _facade.AlignableInferred[LDSFacadeT, Posterior],
    typing.Generic[LDSFacadeT],
):
    """Singular LDS facade, posterior, and the exact source data.

    Temporal accessors return `Sequences` with the source batch shape and lengths.
    Use `get(index)` to retrieve one unpadded trajectory on the host.
    Masked observations within each sequence's length retain their timesteps.
    """

    def to_hdf5(self, path: PathLike, *, overwrite: bool = False) -> None:
        hdf5.save_inferred(
            path,
            family='lds',
            model_type=f'xxm.lds.{type(self.model).__name__}',
            params=self.model.to_params(),
            posterior=chains.gaussian_to_storage(self.posterior),
            data=self.data._to_storage(),
            overwrite=overwrite,
        )

    @classmethod
    def from_hdf5(cls, path: PathLike) -> typing.Self:
        """Restore the model or result without running inference."""
        model_type, raw, stored_posterior, stored_data = hdf5.load_inferred(path)
        data = DatasetCollection._from_storage(stored_data)
        params = typing.cast(LDSParams, raw)
        facade_cls = _FACADES.get(model_type)
        if facade_cls is None:
            raise ValueError(f'Unsupported LDS facade: {model_type!r}')
        model = facade_cls.from_params(**params)
        posterior = chains.gaussian_from_storage(stored_posterior)
        return cls(model=typing.cast(LDSFacadeT, model), posterior=posterior, data=data)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.posterior.max_steps != self.data.max_steps:
            raise ValueError('posterior must match the padded time dimension')

        if self.posterior.variable_dim != self.model.latent_dim:
            raise ValueError('posterior variable dimension must match model latent_dim')

    def latent_posterior(self, session: int | None = None) -> TemporalGaussian:
        """Return marginal latent posteriors with batch (N_s,).

        Summaries such as `.mean` and `.std` are `Sequences` with padded
        timesteps cleared.
        """
        posterior, _ = self._session(session)

        return TemporalGaussian(
            Gaussian.from_moments(posterior.means, posterior.covariances),
            posterior.lengths,
        )

    def observation_mean(self, session: int | None = None) -> Sequences:
        """Return expected observations, including masked valid steps."""
        posterior, dataset = self._session(session)
        emissions = self.model.core.emissions.get(session)

        if dataset.batch_shape:
            emissions = emissions.broadcast(
                dataset.batch_shape,
                axis=0,
            )

        values = emissions.observation_mean(posterior)

        return Sequences.from_padded(
            values=values,
            lengths=dataset.lengths,
        )

    def elbo(self) -> jax.Array:
        return _elbo_jit(
            self.model.core,
            self.data,
            self.posterior,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class LDSInferredCollection(
    _facade.AlignableInferredCollection[
        'LDSCollection[LDSFacadeT]',
        Posterior,
    ],
    typing.Generic[LDSFacadeT],
):
    """Independent LDS posteriors with coordinated latent alignments."""


@dataclasses.dataclass(frozen=True, eq=False)
class LDSCollection(
    _facade.FacadeCollection[
        LDSFacadeT,
        Inferred[LDSFacadeT],
        LDSInferredCollection[LDSFacadeT],
    ],
    typing.Generic[LDSFacadeT],
):
    """Independent LDS facades with shared inference/fit orchestration."""

    @classmethod
    def _inferred_collection_cls(
        cls,
    ) -> type[LDSInferredCollection[LDSFacadeT]]:
        return LDSInferredCollection

    @classmethod
    def _inferred_cls(cls, model: LDSFacadeT) -> type[Inferred[LDSFacadeT]]:
        return Inferred


@dataclasses.dataclass(frozen=True, eq=False)
class ExactLDSCollection(
    LDSCollection[ExactLDSFacadeT], typing.Generic[ExactLDSFacadeT]
):
    """Independent exact-inference LDS facades."""

    def infer(
        self,
        data: DatasetCollection,
        *,
        max_workers: int = 1,
    ) -> LDSInferredCollection[ExactLDSFacadeT]:
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
    ) -> _facade.FitCollection[LDSInferredCollection[ExactLDSFacadeT]]:
        """Fit every session independently by EM."""
        return self._fit_each(
            data,
            lambda model, subset: model.fit(
                subset, num_iters=num_iters, progress=progress
            ),
            max_workers=max_workers,
        )


@dataclasses.dataclass(frozen=True, eq=False)
class LaplaceLDSCollection(
    LDSCollection[LaplaceLDSFacadeT], typing.Generic[LaplaceLDSFacadeT]
):
    """Independent Laplace-inference LDS facades."""

    def infer(
        self,
        data: DatasetCollection,
        *,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
        max_workers: int = 1,
    ) -> LDSInferredCollection[LaplaceLDSFacadeT]:
        """Infer every session independently."""
        return self._infer_each(
            data,
            lambda model, subset: model.infer(subset, laplace_params=laplace_params),
            max_workers=max_workers,
        )

    def fit(
        self,
        data: DatasetCollection,
        *,
        num_iters: int,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
        progress: bool | str = False,
        max_workers: int = 1,
    ) -> _facade.FitCollection[LDSInferredCollection[LaplaceLDSFacadeT]]:
        """Fit every session independently by Laplace EM."""
        return self._fit_each(
            data,
            lambda model, subset: model.fit(
                subset,
                num_iters=num_iters,
                progress=progress,
                laplace_params=laplace_params,
            ),
            max_workers=max_workers,
        )


GaussianLDSCollection = ExactLDSCollection[GaussianLDS]
PoissonLDSCollection = LaplaceLDSCollection[PoissonLDS]

_FACADES: dict[str, type[LDS[typing.Any]]] = {
    f'xxm.lds.{facade.__name__}': facade for facade in (GaussianLDS, PoissonLDS)
}
