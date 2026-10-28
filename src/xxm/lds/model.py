"""Facade classes for Gaussian and Poisson linear dynamical systems."""

from __future__ import annotations

import dataclasses
import typing

import jax
from jax import numpy as jnp

from xxm.core import facade as _facade
from xxm.core.affine import Affine
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.dists.gaussian import Gaussian, LinearGaussian, TemporalGaussian
from xxm.core.dists.gaussian import GaussianChainMarginals as Posterior
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.collection import ContinuousEmissionsCollection
from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianLDS(_facade.AlignableFacade[Model[GaussianEmissions]]):
    r"""Linear dynamical system with Gaussian emissions.

    $$x_0 \sim \mathcal{N}(m_0,S_0),$$

    $$x_t\mid x_{t-1},u_t
    \sim \mathcal{N}(Ax_{t-1}+Bu_t+b,Q),$$

    $$y_t\mid x_t \sim \mathcal{N}(Cx_t+d,R).$$
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

    def emissions(self, session: int | None = None) -> LinearGaussian:
        """Linear-Gaussian observation distribution."""
        return self.core.emissions.get(session).dist

    def to_params(self) -> LDSParams:
        """Return stored parameters, with one dictionary per emission session."""
        return {
            'initial_mean': self.core.initial.dist.mean,
            'initial_covariance': self.core.initial.dist.covariance,
            'dynamics_coefficients': self.core.dynamics.latent_coefficients,
            'dynamics_input_coefficients': self.core.dynamics.input_coefficients,
            'dynamics_bias': self.core.dynamics.dist.affine.bias,
            'dynamics_covariance': self.core.dynamics.dist.covariance,
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
        """Construct a Gaussian LDS from explicit parameters."""
        initial = GaussianInitial(
            dist=Gaussian(mean=initial_mean, covariance=initial_covariance)
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
                    tuple(
                        GaussianEmissions(
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
    def via_pca(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        latent_dim: int,
        *,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> GaussianLDSCollection:
        """Initialize an independent model for every session."""
        return GaussianLDSCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda _session_key, dataset: cls(
                init_gaussian_via_pca(
                    DatasetCollection((dataset,)),
                    latent_dim=latent_dim,
                    ridge=ridge,
                    covariance_floor=covariance_floor,
                )
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonLDS(_facade.AlignableFacade[Model[PoissonEmissions]]):
    r"""Linear dynamical system with Poisson emissions.

    $$x_0 \sim \mathcal{N}(m_0,S_0),$$

    $$x_t\mid x_{t-1},u_t
    \sim \mathcal{N}(Ax_{t-1}+Bu_t+b,Q),$$

    $$y_t\mid x_t \sim \operatorname{Poisson}(\exp(Cx_t+d)).$$
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

    def emissions(self, session: int | None = None) -> LinearPoisson:
        """Linear-Poisson observation distribution in log-rate form."""
        return self.core.emissions.get(session).dist

    def to_params(self) -> LDSParams:
        """Return stored parameters, with one dictionary per emission session."""
        return {
            'initial_mean': self.core.initial.dist.mean,
            'initial_covariance': self.core.initial.dist.covariance,
            'dynamics_coefficients': self.core.dynamics.latent_coefficients,
            'dynamics_input_coefficients': self.core.dynamics.input_coefficients,
            'dynamics_bias': self.core.dynamics.dist.affine.bias,
            'dynamics_covariance': self.core.dynamics.dist.covariance,
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
        """Construct a Poisson LDS from explicit parameters."""
        initial = GaussianInitial(
            dist=Gaussian(mean=initial_mean, covariance=initial_covariance)
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
                    tuple(
                        PoissonEmissions(
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
    def via_pca(
        cls,
        key: jax.Array,
        data: DatasetCollection,
        latent_dim: int,
        *,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
    ) -> PoissonLDSCollection:
        """Initialize an independent model for every session."""
        return PoissonLDSCollection.from_datasets(
            key=key,
            data=data,
            init_fn=lambda _session_key, dataset: cls(
                init_poisson_via_pca(
                    DatasetCollection((dataset,)),
                    latent_dim=latent_dim,
                    ridge=ridge,
                    covariance_floor=covariance_floor,
                    count_floor=count_floor,
                )
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


LDSFacadeT = typing.TypeVar('LDSFacadeT', GaussianLDS, PoissonLDS)


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
        if model_type == 'xxm.lds.GaussianLDS':
            model = GaussianLDS.from_params(**params)
        elif model_type == 'xxm.lds.PoissonLDS':
            model = PoissonLDS.from_params(**params)
        else:
            raise ValueError(f'Unsupported LDS facade: {model_type!r}')
        posterior = chains.gaussian_from_storage(stored_posterior)
        return cls(model=typing.cast(LDSFacadeT, model), posterior=posterior, data=data)

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.posterior.max_steps != self.data.max_steps:
            raise ValueError('posterior must match the padded time dimension')

        if self.posterior.variable_dim != self.model.latent_dim:
            raise ValueError('posterior variable dimension must match model latent_dim')

    def latent_mean(self, session: int | None = None) -> Sequences:
        """Return latent means with the source batch shape and lengths."""
        posterior, dataset = self._session(session)
        return Sequences.from_padded(
            values=posterior.means,
            lengths=dataset.lengths,
        )

    def latent_std(self, session: int | None = None) -> Sequences:
        """Return latent standard deviations with the source batch shape and lengths."""
        posterior, dataset = self._session(session)
        stds = jnp.sqrt(jnp.diagonal(posterior.covariances, axis1=-2, axis2=-1))

        return Sequences.from_padded(
            values=stds,
            lengths=dataset.lengths,
        )

    def latent_posterior(self, session: int | None = None) -> TemporalGaussian:
        """Return marginal latent posteriors with the posterior lengths."""
        posterior, _ = self._session(session)

        return TemporalGaussian(
            Gaussian(posterior.means, posterior.covariances),
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


GaussianLDSCollection = LDSCollection[GaussianLDS]
PoissonLDSCollection = LDSCollection[PoissonLDS]
