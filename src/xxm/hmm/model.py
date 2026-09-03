from __future__ import annotations

import dataclasses
import typing

import jax

from xxm.core.affine import Affine
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.emissions.discrete import (
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.emissions.discrete_ar import AREmissions
from xxm.core.models.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.optim.loop import Fit, FitCollection

from .core import Model, Posterior
from .inference import infer_exact
from .init import (
    init_gaussian,
    init_gaussian_ar,
    init_poisson,
    init_poisson_ar,
)
from .learning import fit_em, fit_em_many


def _categorical_components_from_params(
    initial_probs: jax.Array,
    transition_probs: jax.Array,
) -> tuple[CategoricalInitial, CategoricalTransitions]:
    return (
        CategoricalInitial(
            model=Categorical(
                probs=initial_probs,
            )
        ),
        CategoricalTransitions(
            model=Categorical(
                probs=transition_probs,
            )
        ),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianHMM:
    """Hidden Markov model with Gaussian emissions."""

    model: Model[GaussianEmissions]

    @property
    def num_states(self) -> int:
        return self.model.num_states

    def permute(self, permutation: jax.Array) -> GaussianHMM:
        return GaussianHMM(
            model=self.model.permute(permutation),
        )

    @property
    def states(self) -> Gaussian:
        return self.model.emissions.model

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emission_means: jax.Array,
        emission_covariances: jax.Array,
    ) -> typing.Self:
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            Model(
                initial=initial,
                transitions=transitions,
                emissions=GaussianEmissions(
                    model=Gaussian(
                        mean=emission_means,
                        covariance=emission_covariances,
                    )
                ),
            ),
        )

    @classmethod
    def from_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        *,
        self_transition_prob: float = 0.9,
    ) -> typing.Self:
        model = init_gaussian(
            key=key,
            observations=observations,
            num_states=num_states,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

    def sample(self, key: jax.Array, num_steps: int) -> tuple[jax.Array, jax.Array]:
        return self.model.sample(
            key,
            num_steps,
        )

    def infer(
        self,
        observations: jax.Array,
    ) -> tuple[Posterior, jax.Array]:
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonHMM:
    """Hidden Markov model with Poisson emissions."""

    model: Model[PoissonEmissions]

    @property
    def num_states(self) -> int:
        return self.model.num_states

    def permute(self, permutation: jax.Array) -> PoissonHMM:
        return PoissonHMM(
            model=self.model.permute(permutation),
        )

    @property
    def states(self) -> Poisson:
        return self.model.emissions.model

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emission_log_rates: jax.Array,
    ) -> typing.Self:
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            model=Model(
                initial=initial,
                transitions=transitions,
                emissions=PoissonEmissions(
                    model=Poisson(
                        log_rates=emission_log_rates,
                    )
                ),
            ),
        )

    @classmethod
    def from_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        *,
        self_transition_prob: float = 0.9,
    ) -> typing.Self:
        model = init_poisson(
            key=key,
            observations=observations,
            num_states=num_states,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianARHMM:
    """Autoregressive HMM with Gaussian emissions."""

    model: Model[AREmissions[LinearGaussian]]

    @property
    def num_states(self) -> int:
        return self.model.emissions.num_states

    @property
    def output_dim(self) -> int:
        return self.model.emissions.output_dim

    @property
    def num_lags(self) -> int:
        return self.model.emissions.num_lags

    def permute(self, permutation: jax.Array) -> GaussianARHMM:
        return GaussianARHMM(
            model=self.model.permute(permutation),
        )

    @property
    def states(self) -> LinearGaussian:
        return self.model.emissions.model

    def states_conditional(self, observations: jax.Array) -> Gaussian:
        return self.model.emissions.conditional(observations)

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emission_coefficients: jax.Array,  # (K, O, L, I)
        emission_bias: jax.Array,  # (K, O)
        emission_covariances: jax.Array,  # (K, O, O)
    ) -> typing.Self:
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            model=Model(
                initial=initial,
                transitions=transitions,
                emissions=AREmissions(
                    model=LinearGaussian(
                        affine=Affine(
                            coefficients=emission_coefficients,
                            bias=emission_bias,
                        ),
                        covariance=emission_covariances,
                    )
                ),
            ),
        )

    @classmethod
    def from_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        num_lags: int,
        *,
        self_transition_prob: float = 0.9,
    ) -> typing.Self:
        model = init_gaussian_ar(
            key=key,
            observations=observations,
            num_states=num_states,
            num_lags=num_lags,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonARHMM:
    """Autoregressive HMM with Poisson emissions."""

    model: Model[AREmissions[LinearPoisson]]

    @property
    def num_states(self) -> int:
        return self.model.emissions.num_states

    @property
    def output_dim(self) -> int:
        return self.model.emissions.output_dim

    @property
    def num_lags(self) -> int:
        return self.model.emissions.num_lags

    def permute(self, permutation: jax.Array) -> PoissonARHMM:
        return PoissonARHMM(
            model=self.model.permute(permutation),
        )

    @property
    def states(self) -> LinearPoisson:
        return self.model.emissions.model

    def states_conditional(self, observations: jax.Array) -> Poisson:
        return self.model.emissions.conditional(observations)

    @classmethod
    def from_params(
        cls,
        *,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        emission_coefficients: jax.Array,  # (K, O, L, I)
        emission_bias: jax.Array,  # (K, O)
    ) -> typing.Self:
        initial, transitions = _categorical_components_from_params(
            initial_probs,
            transition_probs,
        )

        return cls(
            model=Model(
                initial=initial,
                transitions=transitions,
                emissions=AREmissions(
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
    def from_kmeans(
        cls,
        key: jax.Array,
        observations: jax.Array,
        num_states: int,
        num_lags: int,
        *,
        self_transition_prob: float = 0.9,
    ) -> typing.Self:
        model = init_poisson_ar(
            key=key,
            observations=observations,
            num_states=num_states,
            num_lags=num_lags,
            self_transition_prob=self_transition_prob,
        )

        return cls(model)

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
