"""Facade classes for Gaussian and Poisson linear dynamical systems."""

from __future__ import annotations

import dataclasses
import typing

import jax

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.continuous import (
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.inference import Fitted, FittedCollection, Inferred
from xxm.core.latents.gaussian import (
    GaussianInitial,
    GaussianLinearDynamics,
)
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model
from .inference import (
    infer_exact,
    infer_laplace,
)
from .init import (
    init_gaussian_via_pca,
    init_gaussian_via_pca_many,
    init_poisson_via_pca,
    init_poisson_via_pca_many,
)
from .learning import (
    fit_em,
    fit_em_many,
    fit_laplace_em,
    fit_laplace_em_many,
)

_infer_exact_jit = jax.jit(infer_exact)

_infer_laplace_jit = jax.jit(infer_laplace)

_fit_em_jit = jax.jit(
    fit_em,
    static_argnames=(
        'num_iters',
        'progress',
    ),
)

_fit_em_many_jit = jax.jit(
    fit_em_many,
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

_fit_laplace_em_many_jit = jax.jit(
    fit_laplace_em_many,
    static_argnames=(
        'num_iters',
        'progress',
    ),
)


def _latent_components_from_params(
    initial_mean: jax.Array,
    initial_covariance: jax.Array,
    dynamics_coefficients: jax.Array,
    dynamics_bias: jax.Array,
    dynamics_covariance: jax.Array,
) -> tuple[GaussianInitial, GaussianLinearDynamics]:
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
                    coefficients=dynamics_coefficients,
                    bias=dynamics_bias,
                ),
                covariance=dynamics_covariance,
            )
        ),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianLDS:
    r"""Linear dynamical system with Gaussian emissions.

    $$x_0 \sim \mathcal{N}(m_0, S_0), \quad
    x_t|x_{t-1} \sim \mathcal{N}(Ax_{t-1}+b, Q), \quad
    y_t|x_t \sim \mathcal{N}(Cx_t+d, R).$$
    """

    _model: Model[GaussianEmissions]

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self._model.dynamics.dist.output_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self._model.emissions.dist.output_dim

    @property
    def dynamics(self) -> LinearGaussian:
        """Linear-Gaussian latent transition distribution."""
        return self._model.dynamics.dist

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
    ) -> typing.Self:
        """Construct a Gaussian LDS from initial, dynamics, and emission parameters."""
        initial, dynamics = _latent_components_from_params(
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_coefficients=dynamics_coefficients,
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
        observations: jax.Array,
        latent_dim: int,
        *,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Initialize a Gaussian LDS from a principal-component decomposition."""
        model = init_gaussian_via_pca(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floor=covariance_floor,
        )

        return cls(model)

    @classmethod
    def via_pca_many(
        cls,
        observations: jax.Array,
        latent_dim: int,
        *,
        covariance_floors: jax.Array,
    ) -> tuple[typing.Self, ...]:
        """Construct one Gaussian LDS for each requested covariance floor."""
        models = init_gaussian_via_pca_many(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floors=covariance_floors,
        )

        return tuple(cls(model) for model in models)

    def sample(self, key: jax.Array, num_steps: int) -> tuple[jax.Array, jax.Array]:
        r"""Sample latent states $x_{0:T-1}$ and observations $y_{0:T-1}$."""
        return self._model.sample(
            key,
            num_steps,
        )

    def infer(self, observations: jax.Array) -> Inferred[typing.Self, Posterior]:
        """Compute the exact Gaussian posterior and observation log likelihood."""
        inferred = _infer_exact_jit(
            self._model,
            observations,
        )

        return Inferred(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            objective=inferred.objective,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fitted[typing.Self, Posterior]:
        """Fit model parameters by expectation-maximization."""
        fit = _fit_em_jit(
            self._model,
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return Fitted(
            model=self.__class__(fit.state.model),
            posterior=fit.state.posterior,
            objective_trace=fit.objective_trace,
        )

    @classmethod
    def fit_many(
        cls,
        models: tuple[typing.Self, ...],
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'Multi-EM',
    ) -> FittedCollection[typing.Self, Posterior]:
        """Fit multiple LDS initializations to the same observations by EM."""
        fit = _fit_em_many_jit(
            tuple(model._model for model in models),
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return FittedCollection(
            models=tuple(cls(state.model) for state in fit.states),
            posteriors=tuple(state.posterior for state in fit.states),
            objective_traces=fit.objective_traces,
        )

    def latent_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean latent trajectory."""
        return posterior.means

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean observation at each time point."""
        return self._model.emissions.observation_mean(posterior)

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            _model=self._model.align(alignment),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonLDS:
    r"""Linear dynamical system with Poisson emissions.

    $$x_0 \sim \mathcal{N}(m_0, S_0), \quad
    x_t|x_{t-1} \sim \mathcal{N}(Ax_{t-1}+b, Q), \quad
    y_t|x_t \sim \operatorname{Poisson}(\exp(Cx_t+d)).$$
    """

    _model: Model[PoissonEmissions]

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self._model.dynamics.dist.output_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self._model.emissions.dist.output_dim

    @property
    def dynamics(self) -> LinearGaussian:
        """Linear-Gaussian latent transition distribution."""
        return self._model.dynamics.dist

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
    ) -> typing.Self:
        """Construct a Poisson LDS from initial, dynamics, and emission parameters."""
        initial, dynamics = _latent_components_from_params(
            initial_mean=initial_mean,
            initial_covariance=initial_covariance,
            dynamics_coefficients=dynamics_coefficients,
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
        observations: jax.Array,
        latent_dim: int,
        *,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Initialize a Poisson LDS from a principal-component decomposition."""
        model = init_poisson_via_pca(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floor=covariance_floor,
        )

        return cls(model)

    @classmethod
    def via_pca_many(
        cls,
        observations: jax.Array,
        latent_dim: int,
        *,
        covariance_floors: jax.Array,
    ) -> tuple[typing.Self, ...]:
        """Construct one Poisson LDS for each requested covariance floor."""
        models = init_poisson_via_pca_many(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floors=covariance_floors,
        )

        return tuple(cls(model) for model in models)

    def sample(self, key: jax.Array, num_steps: int) -> tuple[jax.Array, jax.Array]:
        r"""Sample latent states $x_{0:T-1}$ and observations $y_{0:T-1}$."""
        return self._model.sample(
            key,
            num_steps,
        )

    def infer(
        self,
        observations: jax.Array,
        *,
        initial_latents: jax.Array | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Inferred[typing.Self, Posterior]:
        """
        Compute the Laplace posterior and approximate observation log likelihood.
        """
        inferred = _infer_laplace_jit(
            self._model,
            observations,
            initial_latents=initial_latents,
            params=laplace_params,
        )

        return Inferred(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            objective=inferred.objective,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'Laplace EM',
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Fitted[typing.Self, Posterior]:
        """Fit model parameters with Laplace-approximated expectation-maximization."""
        fit = _fit_laplace_em_jit(
            self._model,
            observations,
            num_iters=num_iters,
            progress=progress,
            laplace_params=laplace_params,
        )

        return Fitted(
            model=self.__class__(fit.state.model),
            posterior=fit.state.posterior,
            objective_trace=fit.objective_trace,
        )

    @classmethod
    def fit_many(
        cls,
        models: tuple[typing.Self, ...],
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'Multi-Laplace EM',
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> FittedCollection[typing.Self, Posterior]:
        """Fit multiple Poisson LDS initializations with Laplace EM."""
        fit = _fit_laplace_em_many_jit(
            tuple(model._model for model in models),
            observations,
            num_iters=num_iters,
            progress=progress,
            laplace_params=laplace_params,
        )

        return FittedCollection(
            models=tuple(cls(state.model) for state in fit.states),
            posteriors=tuple(state.posterior for state in fit.states),
            objective_traces=fit.objective_traces,
        )

    def latent_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean latent trajectory."""
        return posterior.means

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean observation at each time point."""
        return self._model.emissions.observation_mean(posterior)

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            _model=self._model.align(alignment),
        )
