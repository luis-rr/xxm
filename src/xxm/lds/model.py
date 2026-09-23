"""Facade classes for Gaussian and Poisson linear dynamical systems."""

from __future__ import annotations

import dataclasses
import typing

import jax
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.data import Dataset, Sequences
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model
from .inference import elbo, infer_exact, infer_laplace
from .init import init_gaussian_via_pca, init_poisson_via_pca
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


def _latent_components_from_params(
    initial_mean: jax.Array,
    initial_covariance: jax.Array,
    dynamics_coefficients: jax.Array,
    dynamics_input_coefficients: jax.Array | None,
    dynamics_bias: jax.Array,
    dynamics_covariance: jax.Array,
) -> tuple[GaussianInitial, GaussianLinearDynamics]:
    dynamics_coefficients = jnp.asarray(dynamics_coefficients)

    if dynamics_input_coefficients is None:
        dynamics_input_coefficients = jnp.empty(
            (*dynamics_coefficients.shape[:-1], 0),
            dtype=dynamics_coefficients.dtype,
        )
    else:
        dynamics_input_coefficients = jnp.asarray(dynamics_input_coefficients)

    coefficients = jnp.concatenate(
        [
            dynamics_coefficients,
            dynamics_input_coefficients,
        ],
        axis=-1,
    )

    return (
        GaussianInitial(
            dist=Gaussian(
                mean=initial_mean,
                covariance=initial_covariance,
            )
        ),
        GaussianLinearDynamics(
            dist=LinearGaussian(
                affine=Affine(
                    coefficients=coefficients,
                    bias=dynamics_bias,
                ),
                covariance=dynamics_covariance,
            )
        ),
    )


def _sampling_inputs(
    model: Model,
    num_steps: int,
    inputs: jax.Array | None,
) -> jax.Array:
    batch_shape = model.dynamics.batch_shape
    input_dim = model.dynamics.input_dim

    if inputs is None:
        if input_dim != 0:
            raise ValueError('controlled LDS sampling requires explicit inputs')

        return jnp.empty(
            (*batch_shape, num_steps, 0),
            dtype=model.dynamics.dist.affine.bias.dtype,
        )

    return jnp.asarray(inputs)


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
class GaussianLDS:
    r"""Linear dynamical system with Gaussian emissions.

    $$x_0 \sim \mathcal{N}(m_0,S_0),$$

    $$x_t\mid x_{t-1},u_t
    \sim \mathcal{N}(Ax_{t-1}+Bu_t+b,Q),$$

    $$y_t\mid x_t \sim \mathcal{N}(Cx_t+d,R).$$
    """

    _model: Model[GaussianEmissions]

    def __post_init__(self) -> None:
        if self._model.batch_shape != ():
            raise ValueError('GaussianLDS facade requires an unbatched core model')

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self._model.dynamics.latent_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self._model.emissions.dist.output_dim

    def input_dim(self) -> int:
        """Dimension $D_u$ of known dynamics inputs."""
        return self._model.dynamics.input_dim

    def input_coefficients(self) -> jax.Array:
        """Known-input dynamics coefficients $B$."""
        return self._model.dynamics.input_coefficients

    def dynamics(self, inputs: jax.Array | None = None) -> LinearGaussian:
        r"""Return $p(x_t\mid x_{t-1},u_t)$ conditioned on a known input."""
        return _conditioned_dynamics(self._model, inputs)

    @property
    def emissions(self) -> LinearGaussian:
        """Linear-Gaussian observation distribution."""
        return self._model.emissions.dist

    @classmethod
    def from_params(
        cls,
        *,
        initial_mean: jax.Array,
        initial_covariance: jax.Array,
        dynamics_coefficients: jax.Array,
        dynamics_bias: jax.Array,
        dynamics_covariance: jax.Array,
        emission_coefficients: jax.Array,
        emission_bias: jax.Array,
        emission_covariance: jax.Array,
        dynamics_input_coefficients: jax.Array | None = None,
    ) -> typing.Self:
        """Construct a Gaussian LDS from explicit parameters."""
        initial, dynamics = _latent_components_from_params(
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_coefficients=dynamics_coefficients,
            dynamics_input_coefficients=dynamics_input_coefficients,
            dynamics_bias=dynamics_bias,
            dynamics_covariance=dynamics_covariance,
        )

        return cls(
            _model=Model(
                initial=initial,
                dynamics=dynamics,
                emissions=GaussianEmissions(
                    dist=LinearGaussian(
                        affine=Affine(
                            coefficients=emission_coefficients,
                            bias=emission_bias,
                        ),
                        covariance=emission_covariance,
                    )
                ),
            ),
        )

    @classmethod
    def via_pca(
        cls,
        data: Dataset,
        latent_dim: int,
        *,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Initialize one shared Gaussian LDS from a Dataset."""
        return cls(
            init_gaussian_via_pca(
                data,
                latent_dim=latent_dim,
                ridge=ridge,
                covariance_floor=covariance_floor,
            )
        )

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
        *,
        inputs: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        r"""Sample latent states $x_{0:T-1}$ and observations $y_{0:T-1}$."""
        return self._model.sample(
            key,
            num_steps,
            inputs=_sampling_inputs(
                self._model,
                num_steps,
                inputs,
            ),
        )

    def infer(
        self,
        data: Dataset,
    ) -> Inferred[GaussianLDS]:
        """Compute exact posteriors with the Dataset's batch shape."""
        inferred = _infer_exact_jit(
            self._model,
            data,
        )

        return Inferred(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            data=data,
        )

    def fit(
        self,
        data: Dataset,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[GaussianLDS]:
        """Fit shared parameters across all Dataset batch entries by EM."""
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

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            _model=self._model.align(alignment),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonLDS:
    r"""Linear dynamical system with Poisson emissions.

    $$x_0 \sim \mathcal{N}(m_0,S_0),$$

    $$x_t\mid x_{t-1},u_t
    \sim \mathcal{N}(Ax_{t-1}+Bu_t+b,Q),$$

    $$y_t\mid x_t \sim \operatorname{Poisson}(\exp(Cx_t+d)).$$
    """

    _model: Model[PoissonEmissions]

    def __post_init__(self) -> None:
        if self._model.batch_shape != ():
            raise ValueError('PoissonLDS facade requires an unbatched core model')

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self._model.dynamics.latent_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self._model.emissions.dist.output_dim

    def input_dim(self) -> int:
        """Dimension $D_u$ of known dynamics inputs."""
        return self._model.dynamics.input_dim

    def input_coefficients(self) -> jax.Array:
        """Known-input dynamics coefficients $B$."""
        return self._model.dynamics.input_coefficients

    def dynamics(self, inputs: jax.Array | None = None) -> LinearGaussian:
        r"""Return $p(x_t\mid x_{t-1},u_t)$ conditioned on a known input."""
        return _conditioned_dynamics(self._model, inputs)

    @property
    def emissions(self) -> LinearPoisson:
        """Linear-Poisson observation distribution in log-rate form."""
        return self._model.emissions.dist

    @classmethod
    def from_params(
        cls,
        *,
        initial_mean: jax.Array,
        initial_covariance: jax.Array,
        dynamics_coefficients: jax.Array,
        dynamics_bias: jax.Array,
        dynamics_covariance: jax.Array,
        emission_coefficients: jax.Array,
        emission_bias: jax.Array,
        dynamics_input_coefficients: jax.Array | None = None,
    ) -> typing.Self:
        """Construct a Poisson LDS from explicit parameters."""
        initial, dynamics = _latent_components_from_params(
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_coefficients=dynamics_coefficients,
            dynamics_input_coefficients=dynamics_input_coefficients,
            dynamics_bias=dynamics_bias,
            dynamics_covariance=dynamics_covariance,
        )

        return cls(
            _model=Model(
                initial=initial,
                dynamics=dynamics,
                emissions=PoissonEmissions(
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
    def via_pca(
        cls,
        data: Dataset,
        latent_dim: int,
        *,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
    ) -> typing.Self:
        """Initialize one shared Poisson LDS from a Dataset."""
        return cls(
            init_poisson_via_pca(
                data,
                latent_dim=latent_dim,
                ridge=ridge,
                covariance_floor=covariance_floor,
                count_floor=count_floor,
            )
        )

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
        *,
        inputs: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        r"""Sample latent states $x_{0:T-1}$ and observations $y_{0:T-1}$."""
        return self._model.sample(
            key,
            num_steps,
            inputs=_sampling_inputs(
                self._model,
                num_steps,
                inputs,
            ),
        )

    def infer(
        self,
        data: Dataset,
        *,
        initial_latents: jax.Array | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Inferred[PoissonLDS]:
        """Compute Laplace posteriors with the Dataset's batch shape.

        Optional initial latents have shape `(*data.batch_shape, T, D_x)`.
        """
        inferred = _infer_laplace_jit(
            self._model,
            data,
            initial_latents=initial_latents,
            params=laplace_params,
        )

        return Inferred(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            data=data,
        )

    def fit(
        self,
        data: Dataset,
        *,
        num_iters: int,
        progress: bool | str = 'Laplace EM',
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Fit[PoissonLDS]:
        """Fit shared parameters across Dataset batch entries by Laplace EM."""
        fit = _fit_laplace_em_jit(
            self._model,
            data,
            num_iters=num_iters,
            progress=progress,
            laplace_params=laplace_params,
        )

        return Fit(
            inferred=Inferred(
                model=self.__class__(fit.state.model),
                posterior=fit.state.posterior,
                data=data,
            ),
            objective_trace=fit.objective_trace,
        )

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            _model=self._model.align(alignment),
        )


LDSFacadeT = typing.TypeVar('LDSFacadeT', GaussianLDS, PoissonLDS)


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(typing.Generic[LDSFacadeT]):
    """Singular LDS facade, posterior, and the exact source Dataset.

    Temporal accessors return `Sequences` with the source batch shape and lengths.
    Use `get(index)` to retrieve one unpadded trajectory on the host.
    Masked observations within each sequence's length retain their timesteps.
    """

    model: LDSFacadeT
    posterior: Posterior
    data: Dataset

    def __post_init__(self) -> None:
        expected_batch = self.data.batch_shape

        if self.posterior.batch_shape != expected_batch:
            raise ValueError(
                f'expected posterior batch shape {expected_batch}, '
                f'got {self.posterior.batch_shape}'
            )

        if self.posterior.num_steps != self.data.num_steps:
            raise ValueError('posterior must match the padded time dimension')

        if self.posterior.variable_dim != self.model.latent_dim:
            raise ValueError('posterior variable dimension must match model latent_dim')

    def latent_mean(self) -> Sequences:
        """Return latent means with the source batch shape and lengths."""
        return Sequences(values=self.posterior.means, lengths=self.data.lengths)

    def observation_mean(self) -> Sequences:
        """Return expected observations, including masked valid steps."""
        emissions = self.model._model.emissions

        if self.data.batch_shape:
            emissions = emissions.broadcast(
                self.data.batch_shape,
                axis=0,
            )

        values = emissions.observation_mean(self.posterior)

        return Sequences(values=values, lengths=self.data.lengths)

    def align(self, alignment: Affine) -> Inferred[LDSFacadeT]:
        """Transform model and posterior coordinates exactly, without reinference."""
        model = self.model.align(alignment)
        posterior_alignment = alignment
        if self.data.batch_shape:
            posterior_alignment = alignment.broadcast(self.data.batch_shape, axis=0)

        return self.__class__(
            model=model,
            posterior=self.posterior.affine(posterior_alignment),
            data=self.data,
        )

    def elbo(self) -> jax.Array:
        return _elbo_jit(
            self.model._model,
            self.data,
            self.posterior,
        )


class Fit(
    typing.NamedTuple,
    typing.Generic[LDSFacadeT],
):
    """Result of fitting a model."""

    inferred: Inferred[LDSFacadeT]
    objective_trace: jax.Array
