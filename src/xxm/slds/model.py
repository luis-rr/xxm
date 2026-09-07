from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.latents.gaussian import StateConditionedGaussian
from xxm.core.optim.loop import Fit, FitCollection
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import (
    GaussianLinearSwitchingDynamics,
    Model,
    Posterior,
)
from .inference import infer_laplace, infer_variational
from .init import (
    init_arhmm_gaussian,
    init_arhmm_poisson,
    init_pca_gaussian,
    init_pca_poisson,
)
from .learning import (
    fit_laplace_em,
    fit_laplace_em_many,
    fit_variational_em,
    fit_variational_em_many,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianSLDS:
    r"""Switching linear dynamical system with Gaussian emissions.

    $$z_0 \sim \operatorname{Categorical}(\pi), \quad
    z_t|z_{t-1} \sim \operatorname{Categorical}(P_{z_{t-1},:}), \quad
    x_t|x_{t-1},z_t=k \sim \mathcal{N}(A_kx_{t-1}+b_k,Q_k),$$
    $$y_t\mid x_t \sim \mathcal{N}(Cx_t+d,R).$$
    """

    model: Model[GaussianEmissions]

    @property
    def num_states(self) -> int:
        """Number of discrete dynamical states $K$."""
        return self.model.num_states

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self.model.dynamics.model.output_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self.model.emissions.model.output_dim

    @property
    def dynamics(self) -> LinearGaussian:
        """State-conditioned linear-Gaussian transition distributions."""
        return self.model.dynamics.model

    @property
    def emissions(self) -> LinearGaussian:
        """Linear-Gaussian observation distribution."""
        return self.model.emissions.model

    def permute(self, permutation: jax.Array) -> typing.Self:
        """Relabel the discrete latent states."""
        return self.__class__(
            model=self.model.permute(permutation),
        )

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            model=self.model.align(alignment),
        )

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely discrete state at each time point."""
        return jnp.argmax(
            posterior.discrete.state_probs,
            axis=-1,
        )

    def latent_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean continuous latent trajectory."""
        return posterior.continuous.means

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean observation at each time point."""
        return self.model.emissions.observation_mean(
            posterior.continuous,
        )

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,  # (K,)
        transition_probs: jax.Array,  # (K, K)
        latent_initial_means: jax.Array,  # (K, D)
        latent_initial_covariances: jax.Array,  # (K, D, D)
        dynamics_coefficients: jax.Array,  # (K, D, D)
        dynamics_bias: jax.Array,  # (K, D)
        dynamics_covariances: jax.Array,  # (K, D, D)
        emission_coefficients: jax.Array,  # (N, D)
        emission_bias: jax.Array,  # (N,)
        emission_covariance: jax.Array,  # (N, N)
    ) -> typing.Self:
        """Construct a Gaussian SLDS from discrete, latent, and emission parameters."""
        return cls(
            model=Model(
                state_initial=CategoricalInitial(
                    model=Categorical(
                        probs=initial_probs,
                    )
                ),
                transitions=CategoricalTransitions(
                    model=Categorical(
                        probs=transition_probs,
                    )
                ),
                latent_initial=StateConditionedGaussian(
                    model=Gaussian(
                        mean=latent_initial_means,
                        covariance=latent_initial_covariances,
                    )
                ),
                dynamics=GaussianLinearSwitchingDynamics(
                    model=LinearGaussian(
                        affine=Affine(
                            coefficients=dynamics_coefficients,
                            bias=dynamics_bias,
                        ),
                        covariance=dynamics_covariances,
                    )
                ),
                emissions=GaussianEmissions(
                    model=LinearGaussian(
                        affine=Affine(
                            coefficients=emission_coefficients,
                            bias=emission_bias,
                        ),
                        covariance=emission_covariance,
                    )
                ),
            )
        )

    @classmethod
    def from_pca(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        latent_dim: int,
        *,
        self_transition_prob: float = 0.9,
        covariance_floor: float = 1e-2,
    ) -> typing.Self:
        """Initialize a Gaussian SLDS from a principal-component decomposition."""
        model = init_pca_gaussian(
            key=key,
            observations=observations,
            num_states=num_states,
            latent_dim=latent_dim,
            self_transition_prob=self_transition_prob,
            covariance_floor=covariance_floor,
        )

        return cls(model)

    @classmethod
    def from_arhmm(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        latent_dim: int,
        *,
        num_arhmm_iters: int = 10,
        self_transition_prob: float = 0.9,
        covariance_floor: float = 1e-2,
        progress: bool | str = 'AR-HMM',
    ) -> typing.Self:
        """Initialize a Gaussian SLDS using an autoregressive HMM fit."""
        model = init_arhmm_gaussian(
            key=key,
            observations=observations,
            num_states=num_states,
            latent_dim=latent_dim,
            num_arhmm_iters=num_arhmm_iters,
            self_transition_prob=self_transition_prob,
            covariance_floor=covariance_floor,
            progress=progress,
        )

        return cls(model)

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Sample states $z_{0:T-1}$, latents $x_{0:T-1}$, and observations $y_{0:T-1}$."""
        return self.model.sample(
            key,
            num_steps,
        )

    def infer(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
    ) -> tuple[Posterior, jax.Array]:
        """Compute a variational posterior over discrete states and continuous latents."""
        return infer_variational(
            self.model,
            observations,
            num_iters=num_iters,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        num_inference_iters: int,
        progress: bool | str = 'Variational EM',
    ) -> Fit[typing.Self]:
        """Fit model parameters with variational expectation-maximization."""
        fit = fit_variational_em(
            self.model,
            observations,
            num_iters=num_iters,
            num_inference_iters=num_inference_iters,
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
        num_inference_iters: int,
        progress: bool | str = 'Multi-Variational EM',
    ) -> FitCollection[typing.Self]:
        """Fit multiple Gaussian SLDS initializations with variational EM."""
        fit = fit_variational_em_many(
            tuple(model.model for model in models),
            observations,
            num_iters=num_iters,
            num_inference_iters=num_inference_iters,
            progress=progress,
        )

        return FitCollection(
            models=tuple(cls(model) for model in fit.models),
            objective_traces=fit.objective_traces,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonSLDS:
    r"""Switching linear dynamical system with Poisson emissions.

    $$z_0 \sim \operatorname{Categorical}(\pi), \quad
    z_t|z_{t-1} \sim \operatorname{Categorical}(P_{z_{t-1},:}), \quad
    x_t|x_{t-1},z_t=k \sim \mathcal{N}(A_kx_{t-1}+b_k,Q_k),$$
    $$y_t\mid x_t \sim \operatorname{Poisson}(\exp(Cx_t+d)).$$
    """

    model: Model[PoissonEmissions]

    @property
    def num_states(self) -> int:
        """Number of discrete dynamical states $K$."""
        return self.model.num_states

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self.model.dynamics.model.output_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self.model.emissions.model.output_dim

    @property
    def dynamics(self) -> LinearGaussian:
        """State-conditioned linear-Gaussian transition distributions."""
        return self.model.dynamics.model

    @property
    def emissions(self) -> LinearPoisson:
        """Linear-Poisson observation distribution in log-rate form."""
        return self.model.emissions.model

    def permute(self, permutation: jax.Array) -> typing.Self:
        """Relabel the discrete latent states."""
        return self.__class__(
            model=self.model.permute(permutation),
        )

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            model=self.model.align(alignment),
        )

    def most_likely_states(self, posterior: Posterior) -> jax.Array:
        """Return the marginally most likely discrete state at each time point."""
        return jnp.argmax(
            posterior.discrete.state_probs,
            axis=-1,
        )

    def latent_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean continuous latent trajectory."""
        return posterior.continuous.means

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return the posterior mean observation at each time point."""
        return self.model.emissions.observation_mean(
            posterior.continuous,
        )

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,  # (K,)
        transition_probs: jax.Array,  # (K, K)
        latent_initial_means: jax.Array,  # (K, D)
        latent_initial_covariances: jax.Array,  # (K, D, D)
        dynamics_coefficients: jax.Array,  # (K, D, D)
        dynamics_bias: jax.Array,  # (K, D)
        dynamics_covariances: jax.Array,  # (K, D, D)
        emission_coefficients: jax.Array,  # (N, D)
        emission_bias: jax.Array,  # (N,)
    ) -> typing.Self:
        """Construct a Poisson SLDS from discrete, latent, and emission parameters."""
        return cls(
            model=Model(
                state_initial=CategoricalInitial(
                    model=Categorical(
                        probs=initial_probs,
                    )
                ),
                transitions=CategoricalTransitions(
                    model=Categorical(
                        probs=transition_probs,
                    )
                ),
                latent_initial=StateConditionedGaussian(
                    model=Gaussian(
                        mean=latent_initial_means,
                        covariance=latent_initial_covariances,
                    )
                ),
                dynamics=GaussianLinearSwitchingDynamics(
                    model=LinearGaussian(
                        affine=Affine(
                            coefficients=dynamics_coefficients,
                            bias=dynamics_bias,
                        ),
                        covariance=dynamics_covariances,
                    )
                ),
                emissions=PoissonEmissions(
                    model=LinearPoisson(
                        affine=Affine(
                            coefficients=emission_coefficients,
                            bias=emission_bias,
                        ),
                    )
                ),
            )
        )

    @classmethod
    def from_pca(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        latent_dim: int,
        *,
        self_transition_prob: float = 0.9,
        covariance_floor: float = 1e-2,
    ) -> typing.Self:
        """Initialize a Poisson SLDS from a principal-component decomposition."""
        model = init_pca_poisson(
            key=key,
            observations=observations,
            num_states=num_states,
            latent_dim=latent_dim,
            self_transition_prob=self_transition_prob,
            covariance_floor=covariance_floor,
        )

        return cls(model)

    @classmethod
    def from_arhmm(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        latent_dim: int,
        *,
        num_arhmm_iters: int = 10,
        self_transition_prob: float = 0.9,
        covariance_floor: float = 1e-2,
        progress: bool | str = 'AR-HMM',
    ) -> typing.Self:
        """Initialize a Poisson SLDS using an autoregressive HMM fit."""
        model = init_arhmm_poisson(
            key=key,
            observations=observations,
            num_states=num_states,
            latent_dim=latent_dim,
            num_arhmm_iters=num_arhmm_iters,
            self_transition_prob=self_transition_prob,
            covariance_floor=covariance_floor,
            progress=progress,
        )

        return cls(model)

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Sample states $z_{0:T-1}$, latents $x_{0:T-1}$, and observations $y_{0:T-1}$."""
        return self.model.sample(
            key,
            num_steps,
        )

    def infer(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        initial_latents: jax.Array | None = None,
        params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> tuple[Posterior, jax.Array]:
        """Compute a Laplace posterior approximation over states and latents."""
        return infer_laplace(
            self.model,
            observations,
            num_iters=num_iters,
            initial_latents=initial_latents,
            params=params,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        num_iters: int,
        num_inference_iters: int,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
        progress: bool | str = 'Laplace EM',
    ) -> Fit[typing.Self]:
        """Fit model parameters with Laplace-approximated expectation-maximization."""
        fit = fit_laplace_em(
            self.model,
            observations,
            num_iters=num_iters,
            num_inference_iters=num_inference_iters,
            laplace_params=laplace_params,
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
        num_inference_iters: int,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
        progress: bool | str = 'Multi-Laplace EM',
    ) -> FitCollection[typing.Self]:
        """Fit multiple Poisson SLDS initializations with Laplace EM."""
        fit = fit_laplace_em_many(
            tuple(model.model for model in models),
            observations,
            num_iters=num_iters,
            num_inference_iters=num_inference_iters,
            laplace_params=laplace_params,
            progress=progress,
        )

        return FitCollection(
            models=tuple(cls(model) for model in fit.models),
            objective_traces=fit.objective_traces,
        )
