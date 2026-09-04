from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.core.models.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.models.gaussian import StateConditionedGaussian
from xxm.core.optim.loop import Fit, FitCollection

from .core import (
    GaussianLinearSwitchingDynamics,
    Model,
    Posterior,
)
from .inference import infer_variational
from .init import (
    init_arhmm_gaussian,
    init_pca_gaussian,
)
from .learning import (
    fit_variational_em,
    fit_variational_em_many,
)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianSLDS:
    """Switching linear dynamical system with Gaussian emissions."""

    model: Model[GaussianEmissions]

    @property
    def num_states(self) -> int:
        return self.model.num_states

    @property
    def latent_dim(self) -> int:
        return self.model.dynamics.model.output_dim

    @property
    def observation_dim(self) -> int:
        return self.model.emissions.model.output_dim

    @property
    def dynamics(self) -> LinearGaussian:
        return self.model.dynamics.model

    @property
    def emissions(self) -> LinearGaussian:
        return self.model.emissions.model

    def permute(self, permutation: jax.Array) -> GaussianSLDS:
        """Relabel the discrete latent states."""
        return self.__class__(
            model=self.model.permute(permutation),
        )

    def align(self, alignment: Affine) -> GaussianSLDS:
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
