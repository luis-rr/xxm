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
from xxm.core.models.gaussian import (
    GaussianInitial,
    GaussianLinearDynamics,
)
from xxm.core.optim.loop import Fit, FitCollection

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
            model=Gaussian(
                mean=initial_mean,
                covariance=initial_covariance,
            )
        ),
        GaussianLinearDynamics(
            model=LinearGaussian(
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
    """Linear dynamical system with Gaussian emissions."""

    model: Model[GaussianEmissions]

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
                    model=LinearGaussian(
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
        models = init_pca_gaussian_many(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floors=covariance_floors,
        )

        return tuple(cls(model) for model in models)

    def sample(self, key: jax.Array, num_steps: int) -> tuple[jax.Array, jax.Array]:
        return self.model.sample(
            key,
            num_steps,
        )

    def infer(self, observations: jax.Array) -> tuple[Posterior, jax.Array]:
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

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        return self.model.emissions.model.conditional(posterior.means).mean


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonLDS:
    """Linear dynamical system with Poisson emissions."""

    model: Model[PoissonEmissions]

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
                    model=LinearPoisson(
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
        models = init_pca_poisson_many(
            observations=observations,
            latent_dim=latent_dim,
            covariance_floors=covariance_floors,
        )

        return tuple(cls(model) for model in models)

    def sample(self, key: jax.Array, num_steps: int) -> tuple[jax.Array, jax.Array]:
        return self.model.sample(
            key,
            num_steps,
        )

    def infer(
        self,
        observations: jax.Array,
        *,
        initial_latents: jax.Array | None = None,
        max_iter: int = 20,
        tol: float = 1e-6,
        max_line_search_iters: int = 20,
    ) -> tuple[Posterior, jax.Array]:
        return infer_laplace(
            self.model,
            observations,
            initial_latents=initial_latents,
            max_iter=max_iter,
            tol=tol,
            max_line_search_iters=max_line_search_iters,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        progress: bool | str = 'Laplace EM',
    ) -> Fit[typing.Self]:
        fit = fit_laplace_em(
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
        progress: bool | str = 'Multi-Laplace EM',
    ) -> FitCollection[typing.Self]:
        fit = fit_laplace_em_many(
            tuple(model.model for model in models),
            observations,
            num_iters=num_iters,
            progress=progress,
        )

        return FitCollection(
            models=tuple(cls(model) for model in fit.models),
            objective_traces=fit.objective_traces,
        )

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        return self.model.emissions.model.conditional(posterior.means).rates
