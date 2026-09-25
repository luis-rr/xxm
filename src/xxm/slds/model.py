"""Facade classes for Gaussian and Poisson switching linear dynamical systems."""

from __future__ import annotations

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.data import Dataset, Sequences
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions
from xxm.core.emissions.initialization import initial_latent_trajectory
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.latents.gaussian import StateConditionedGaussian
from xxm.core.latents.switching import GaussianLinearSwitchingDynamics
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import (
    Model,
    Posterior,
)
from .inference import infer_laplace, infer_variational
from .init import (
    DEFAULT_SELF_TRANSITION_PROB,
    init_gaussian_via_arhmm,
    init_gaussian_via_pca,
    init_poisson_via_arhmm,
    init_poisson_via_pca,
)
from .learning import (
    fit_laplace_em,
    fit_variational_em,
)

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
    emissions: GaussianEmissions | PoissonEmissions,
    data: Dataset,
    initial_latents: jax.Array | None,
) -> jax.Array:
    """Use an explicit trajectory or initialize without consulting withheld values."""
    if initial_latents is not None:
        return initial_latents
    emissions = emissions.broadcast(data.batch_shape)
    return data.lengths.clear(
        initial_latent_trajectory(
            emissions, data.observations.values, data.observation_weights()[..., None]
        )
    )


def _sampling_inputs(
    model: Model, num_steps: int, inputs: jax.Array | None
) -> jax.Array:
    if inputs is None:
        if model.dynamics.input_dim != 0:
            raise ValueError('controlled SLDS sampling requires explicit inputs')
        return jnp.empty(
            (*model.batch_shape, num_steps, 0),
            dtype=model.dynamics.dist.affine.bias.dtype,
        )
    return jnp.asarray(inputs)


def _conditioned_dynamics(model: Model, inputs: jax.Array | None) -> LinearGaussian:
    if inputs is None:
        if model.dynamics.input_dim != 0:
            raise ValueError('controlled SLDS dynamics require an input value')
        inputs = jnp.empty(
            (*model.batch_shape, 0), dtype=model.dynamics.dist.affine.bias.dtype
        )
    return model.dynamics.conditional(jnp.asarray(inputs))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class GaussianSLDS:
    r"""
    Switching linear dynamical system with Gaussian emissions.

    $$z_0 \sim \operatorname{Categorical}(\pi), \qquad
    z_t \mid z_{t-1} \sim \operatorname{Categorical}(P_{z_{t-1},:}),$$

    $$x_0 \mid z_0=k \sim \mathcal{N}(m_k,S_k), \qquad
    x_t \mid x_{t-1},z_t=k
    \sim \mathcal{N}(A_kx_{t-1}+B_k u_{t-1} + b_k,Q_k),$$

    $$y_t \mid x_t \sim \mathcal{N}(Cx_t+d,R).$$

    For $t>0$, $z_t$ selects the dynamics generating $x_t$ from $x_{t-1}$.
    """

    _model: Model[GaussianEmissions]

    def __post_init__(self) -> None:
        if self._model.batch_shape != ():
            raise ValueError('GaussianSLDS facade requires an unbatched core model')

    @property
    def model(self) -> Model[GaussianEmissions]:
        return self._model

    @property
    def input_dim(self) -> int:
        return self._model.dynamics.input_dim

    @property
    def input_coefficients(self) -> jax.Array:
        return self._model.dynamics.input_coefficients

    @property
    def num_states(self) -> int:
        """Number of discrete dynamical states $K$."""
        return self._model.num_states

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self._model.dynamics.dist.output_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self._model.emissions.dist.output_dim

    def dynamics(self, inputs: jax.Array | None = None) -> LinearGaussian:
        """State-specific latent transitions conditioned on known inputs."""
        return _conditioned_dynamics(self._model, inputs)

    @property
    def emissions(self) -> LinearGaussian:
        """Linear-Gaussian observation distribution."""
        return self._model.emissions.dist

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """Relabel the discrete latent states."""
        return self.__class__(
            _model=self._model.permute_states(permutation),
        )

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            _model=self._model.align(alignment),
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
        dynamics_input_coefficients: jax.Array | None = None,
    ) -> typing.Self:
        """Construct a Gaussian SLDS from discrete, latent, and emission parameters."""
        return cls(
            _model=Model(
                state_initial=CategoricalInitial(
                    dist=Categorical(
                        probs=initial_probs,
                    )
                ),
                transitions=CategoricalTransitions(
                    dist=Categorical(
                        probs=transition_probs,
                    )
                ),
                latent_initial=StateConditionedGaussian(
                    dist=Gaussian(
                        mean=latent_initial_means,
                        covariance=latent_initial_covariances,
                    )
                ),
                dynamics=GaussianLinearSwitchingDynamics.from_params(
                    latent_coefficients=dynamics_coefficients,
                    input_coefficients=dynamics_input_coefficients,
                    bias=dynamics_bias,
                    covariance=dynamics_covariances,
                ),
                emissions=GaussianEmissions(
                    dist=LinearGaussian(
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
    def via_pca(
        cls,
        key: jax.Array,
        data: Dataset,
        num_states: int,
        latent_dim: int,
        *,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Initialize a Gaussian SLDS from a principal-component decomposition."""
        model = init_gaussian_via_pca(
            key=key,
            data=data,
            num_states=num_states,
            latent_dim=latent_dim,
            self_transition_prob=self_transition_prob,
            covariance_floor=covariance_floor,
        )

        return cls(model)

    @classmethod
    def via_arhmm(
        cls,
        key: jax.Array,
        data: Dataset,
        num_states: int,
        latent_dim: int,
        *,
        num_arhmm_iters: int = 10,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        progress: bool | str = 'AR-HMM',
    ) -> typing.Self:
        """Initialize a Gaussian SLDS using an autoregressive HMM fit."""
        model = init_gaussian_via_arhmm(
            key=key,
            data=data,
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
        *,
        inputs: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Sample states $z_{0:T-1}$, latents $x_{0:T-1}$, and observations $y_{0:T-1}$."""
        return self._model.sample(
            key,
            num_steps,
            inputs=_sampling_inputs(self._model, num_steps, inputs),
        )

    def infer(
        self,
        data: Dataset,
        *,
        num_iters: int,
        initial_latents: jax.Array | None = None,
    ) -> Inferred[GaussianSLDS]:
        """Infer structured posteriors with a visibility-respecting initial trajectory."""
        initial_latents = _initial_latents(self._model.emissions, data, initial_latents)

        inferred = _infer_variational_jit(
            self._model,
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
        data: Dataset,
        *,
        num_iters: int,
        num_inference_iters: int,
        initial_latents: jax.Array | None = None,
        progress: bool | str = 'Variational EM',
    ) -> Fit[GaussianSLDS]:
        """Fit model parameters with variational expectation-maximization."""
        initial_latents = _initial_latents(self._model.emissions, data, initial_latents)

        fit = _fit_variational_em_jit(
            self._model,
            data,
            num_iters=num_iters,
            num_inference_iters=num_inference_iters,
            initial_latents=initial_latents,
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PoissonSLDS:
    r"""
    Switching linear dynamical system with Poisson emissions.

    $$z_0 \sim \operatorname{Categorical}(\pi), \qquad
    z_t \mid z_{t-1} \sim \operatorname{Categorical}(P_{z_{t-1},:}),$$

    $$x_0 \mid z_0=k \sim \mathcal{N}(m_k,S_k), \qquad
    x_t \mid x_{t-1},z_t=k
    \sim \mathcal{N}(A_kx_{t-1}+b_k,Q_k),$$

    $$y_t \mid x_t \sim \operatorname{Poisson}(\exp(Cx_t+d)).$$

    For $t>0$, $z_t$ selects the dynamics generating $x_t$ from $x_{t-1}$.
    """

    _model: Model[PoissonEmissions]

    def __post_init__(self) -> None:
        if self._model.batch_shape != ():
            raise ValueError('PoissonSLDS facade requires an unbatched core model')

    @property
    def model(self) -> Model[PoissonEmissions]:
        return self._model

    @property
    def input_dim(self) -> int:
        return self._model.dynamics.input_dim

    @property
    def input_coefficients(self) -> jax.Array:
        return self._model.dynamics.input_coefficients

    @property
    def num_states(self) -> int:
        """Number of discrete dynamical states $K$."""
        return self._model.num_states

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the continuous latent state."""
        return self._model.dynamics.dist.output_dim

    @property
    def observation_dim(self) -> int:
        """Dimension $D_y$ of each observation."""
        return self._model.emissions.dist.output_dim

    def dynamics(self, inputs: jax.Array | None = None) -> LinearGaussian:
        """State-specific latent transitions conditioned on known inputs."""
        return _conditioned_dynamics(self._model, inputs)

    @property
    def emissions(self) -> LinearPoisson:
        """Linear-Poisson observation distribution in log-rate form."""
        return self._model.emissions.dist

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """Relabel the discrete latent states."""
        return self.__class__(
            _model=self._model.permute_states(permutation),
        )

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            _model=self._model.align(alignment),
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
        dynamics_input_coefficients: jax.Array | None = None,
    ) -> typing.Self:
        """Construct a Poisson SLDS from discrete, latent, and emission parameters."""
        return cls(
            _model=Model(
                state_initial=CategoricalInitial(
                    dist=Categorical(
                        probs=initial_probs,
                    )
                ),
                transitions=CategoricalTransitions(
                    dist=Categorical(
                        probs=transition_probs,
                    )
                ),
                latent_initial=StateConditionedGaussian(
                    dist=Gaussian(
                        mean=latent_initial_means,
                        covariance=latent_initial_covariances,
                    )
                ),
                dynamics=GaussianLinearSwitchingDynamics.from_params(
                    latent_coefficients=dynamics_coefficients,
                    input_coefficients=dynamics_input_coefficients,
                    bias=dynamics_bias,
                    covariance=dynamics_covariances,
                ),
                emissions=PoissonEmissions(
                    dist=LinearPoisson(
                        affine=Affine(
                            coefficients=emission_coefficients,
                            bias=emission_bias,
                        ),
                    )
                ),
            )
        )

    @classmethod
    def via_pca(
        cls,
        key: jax.Array,
        data: Dataset,
        num_states: int,
        latent_dim: int,
        *,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Initialize a Poisson SLDS from a principal-component decomposition."""
        model = init_poisson_via_pca(
            key=key,
            data=data,
            num_states=num_states,
            latent_dim=latent_dim,
            self_transition_prob=self_transition_prob,
            covariance_floor=covariance_floor,
        )

        return cls(model)

    @classmethod
    def via_arhmm(
        cls,
        key: jax.Array,
        data: Dataset,
        num_states: int,
        latent_dim: int,
        *,
        num_arhmm_iters: int = 10,
        self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        progress: bool | str = 'AR-HMM',
    ) -> typing.Self:
        """Initialize a Poisson SLDS using an autoregressive HMM fit."""
        model = init_poisson_via_arhmm(
            key=key,
            data=data,
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
        *,
        inputs: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        r"""Sample states $z_{0:T-1}$, latents $x_{0:T-1}$, and observations $y_{0:T-1}$."""
        return self._model.sample(
            key,
            num_steps,
            inputs=_sampling_inputs(self._model, num_steps, inputs),
        )

    def infer(
        self,
        data: Dataset,
        *,
        num_iters: int,
        params: OptimParams = DEFAULT_OPTIM_PARAMS,
        initial_latents: jax.Array | None = None,
    ) -> Inferred[PoissonSLDS]:
        """Infer structured posteriors with a visibility-respecting initial trajectory."""
        initial_latents = _initial_latents(self._model.emissions, data, initial_latents)

        inferred = _infer_laplace_jit(
            self._model,
            data,
            num_iters=num_iters,
            initial_latents=initial_latents,
            params=params,
        )

        return Inferred(
            model=self,
            posterior=inferred.posterior,
            data=data,
        )

    def fit(
        self,
        data: Dataset,
        *,
        num_iters: int,
        num_inference_iters: int,
        initial_latents: jax.Array | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
        progress: bool | str = 'Laplace EM',
    ) -> Fit[PoissonSLDS]:
        """Fit model parameters with Laplace EM."""

        initial_latents = _initial_latents(self._model.emissions, data, initial_latents)

        fit = _fit_laplace_em_jit(
            self._model,
            data,
            num_iters=num_iters,
            num_inference_iters=num_inference_iters,
            laplace_params=laplace_params,
            initial_latents=initial_latents,
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


SLDSFacadeT = typing.TypeVar('SLDSFacadeT', GaussianSLDS, PoissonSLDS)


@dataclasses.dataclass(frozen=True, eq=False)
class Inferred(typing.Generic[SLDSFacadeT]):
    """Singular SLDS, structured posterior, and the exact source Dataset."""

    model: SLDSFacadeT
    posterior: Posterior
    data: Dataset

    def __post_init__(self) -> None:
        if self.posterior.batch_shape != self.data.batch_shape:
            raise ValueError('posterior batch must match the source dataset')
        if self.posterior.continuous.max_steps != self.data.max_steps:
            raise ValueError('posterior must match the padded time dimension')
        if self.posterior.continuous.variable_dim != self.model.latent_dim:
            raise ValueError('posterior must match the model latent dimension')
        if self.posterior.discrete.num_states != self.model.num_states:
            raise ValueError('posterior must match the model number of states')

    def latent_mean(self) -> Sequences:
        """Return posterior continuous latent means."""
        return Sequences.from_padded(
            self.posterior.continuous.means,
            self.data.lengths,
        )

    def state_probs(self) -> Sequences:
        """Return posterior discrete-state probabilities."""
        return Sequences.from_padded(
            self.posterior.discrete.state_probs,
            self.data.lengths,
        )

    def observation_mean(self) -> Sequences:
        """Return posterior mean observations."""
        traces = self.model.model.emissions.broadcast(
            self.posterior.batch_shape,
        ).observation_mean(
            self.posterior.continuous,
        )

        return Sequences.from_padded(
            traces,
            self.data.lengths,
        )

    def most_likely_states(self) -> Sequences:
        """Return the marginally most likely discrete state at each time point."""
        states = jnp.argmax(
            self.posterior.discrete.state_probs,
            axis=-1,
            keepdims=True,
        )

        return Sequences.from_padded(
            states,
            self.data.lengths,
        )

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> Inferred[SLDSFacadeT]:
        """Relabel discrete states in both the model and posterior."""
        return self.__class__(
            model=self.model.permute_states(permutation),
            posterior=self.posterior.permute_states(permutation),
            data=self.data,
        )

    def align(
        self,
        alignment: Affine,
    ) -> Inferred[SLDSFacadeT]:
        """Transform continuous latent coordinates without reinference."""
        posterior_alignment = alignment

        if self.data.batch_shape:
            posterior_alignment = alignment.broadcast(
                self.data.batch_shape,
                axis=0,
            )

        return self.__class__(
            model=self.model.align(alignment),
            posterior=self.posterior._replace(
                continuous=self.posterior.continuous.affine(
                    posterior_alignment,
                ),
            ),
            data=self.data,
        )


class Fit(typing.NamedTuple, typing.Generic[SLDSFacadeT]):
    """Final inference result and objective trace including the initial objective."""

    inferred: Inferred[SLDSFacadeT]
    objective_trace: jax.Array
