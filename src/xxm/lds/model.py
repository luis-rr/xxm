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
from xxm.core.latents.gaussian import (
    GaussianInitial,
    GaussianLinearDynamics,
)
from xxm.core.optim.loop import Fit, FitCollection
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model
from .inference import (
    infer_exact,
    infer_laplace,
)
from .init import (
    init_pca_gaussian,
    init_pca_gaussian_many,
    init_pca_poisson,
    init_pca_poisson_many,
)
from .learning import (
    fit_em,
    fit_em_many,
    fit_laplace_em,
    fit_laplace_em_many,
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

    model: Model[GaussianEmissions]

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self.model.dynamics.dist.output_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self.model.emissions.dist.output_dim

    @property
    def dynamics(self) -> LinearGaussian:
        """Linear-Gaussian latent transition distribution."""
        return self.model.dynamics.dist

    @property
    def emissions(self) -> LinearGaussian:
        """Linear-Gaussian observation distribution."""
        return self.model.emissions.dist

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
            model=Model(
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
    def from_pca(
        cls,
        observations: jax.Array,
        latent_dim: int,
        *,
        covariance_floor: float = 1e-2,
    ) -> typing.Self:
        """Initialize a Gaussian LDS from a principal-component decomposition."""
        model = init_pca_gaussian(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floor=covariance_floor,
        )

        return cls(model)

    @classmethod
    def from_pca_many(
        cls,
        observations: jax.Array,
        latent_dim: int,
        *,
        covariance_floors: jax.Array,
    ) -> tuple[typing.Self, ...]:
        """Construct one Gaussian LDS for each requested covariance floor."""
        models = init_pca_gaussian_many(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floors=covariance_floors,
        )

        return tuple(cls(model) for model in models)

    def sample(self, key: jax.Array, num_steps: int) -> tuple[jax.Array, jax.Array]:
        r"""Sample latent states $x_{0:T-1}$ and observations $y_{0:T-1}$."""
        return self.model.sample(
            key,
            num_steps,
        )

    def infer(self, observations: jax.Array) -> tuple[Posterior, jax.Array]:
        """Compute the exact Gaussian posterior and observation log likelihood."""
        return infer_exact(
            self.model,
            observations,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fit[typing.Self]:
        """Fit model parameters by expectation-maximization."""
        fit = fit_em(
            self.model,
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return Fit(
            model=self.__class__(fit.model),
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
    ) -> FitCollection[typing.Self]:
        """Fit multiple LDS initializations to the same observations by EM."""
        fit = fit_em_many(
            tuple(model.model for model in models),
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return FitCollection(
            models=tuple(cls(model) for model in fit.models),
            objective_traces=fit.objective_traces,
        )

    def latent_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean latent trajectory."""
        return posterior.means

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean observation at each time point."""
        return self.model.emissions.observation_mean(posterior)

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            model=self.model.align(alignment),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonLDS:
    r"""Linear dynamical system with Poisson emissions.

    $$x_0 \sim \mathcal{N}(m_0, S_0), \quad
    x_t|x_{t-1} \sim \mathcal{N}(Ax_{t-1}+b, Q), \quad
    y_t|x_t \sim \operatorname{Poisson}(\exp(Cx_t+d)).$$
    """

    model: Model[PoissonEmissions]

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self.model.dynamics.dist.output_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self.model.emissions.dist.output_dim

    @property
    def dynamics(self) -> LinearGaussian:
        """Linear-Gaussian latent transition distribution."""
        return self.model.dynamics.dist

    @property
    def emissions(self) -> LinearPoisson:
        """Linear-Poisson observation distribution in log-rate form."""
        return self.model.emissions.dist

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
            Model(
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
    def from_pca(
        cls,
        observations: jax.Array,
        latent_dim: int,
        *,
        covariance_floor: float = 1e-2,
    ) -> typing.Self:
        """Initialize a Poisson LDS from a principal-component decomposition."""
        model = init_pca_poisson(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floor=covariance_floor,
        )

        return cls(model)

    @classmethod
    def from_pca_many(
        cls,
        observations: jax.Array,
        latent_dim: int,
        *,
        covariance_floors: jax.Array,
    ) -> tuple[typing.Self, ...]:
        """Construct one Poisson LDS for each requested covariance floor."""
        models = init_pca_poisson_many(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floors=covariance_floors,
        )

        return tuple(cls(model) for model in models)

    def sample(self, key: jax.Array, num_steps: int) -> tuple[jax.Array, jax.Array]:
        r"""Sample latent states $x_{0:T-1}$ and observations $y_{0:T-1}$."""
        return self.model.sample(
            key,
            num_steps,
        )

    def infer(
        self,
        observations: jax.Array,
        *,
        initial_latents: jax.Array | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> tuple[Posterior, jax.Array]:
        """Compute a Laplace posterior approximation and its objective value."""
        return infer_laplace(
            self.model,
            observations,
            initial_latents=initial_latents,
            params=laplace_params,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'Laplace EM',
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Fit[typing.Self]:
        """Fit model parameters with Laplace-approximated expectation-maximization."""
        fit = fit_laplace_em(
            self.model,
            observations,
            num_iters=num_iters,
            progress=progress,
            laplace_params=laplace_params,
        )

        return Fit(
            model=self.__class__(fit.model),
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
    ) -> FitCollection[typing.Self]:
        """Fit multiple Poisson LDS initializations with Laplace EM."""
        fit = fit_laplace_em_many(
            tuple(model.model for model in models),
            observations,
            num_iters=num_iters,
            progress=progress,
            laplace_params=laplace_params,
        )

        return FitCollection(
            models=tuple(cls(model) for model in fit.models),
            objective_traces=fit.objective_traces,
        )

    def latent_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean latent trajectory."""
        return posterior.means

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean observation at each time point."""
        return self.model.emissions.observation_mean(posterior)

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            model=self.model.align(alignment),
        )
