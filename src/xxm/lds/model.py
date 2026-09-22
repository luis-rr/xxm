"""Facade classes for Gaussian and Poisson linear dynamical systems."""

from __future__ import annotations

import dataclasses
import typing

import jax
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.data import Sequences
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions
from xxm.core.inference import Fitted, InferenceState
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model
from .inference import infer_exact, infer_laplace
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


def _observation_mean(
    model: Model,
    posterior: Posterior,
) -> jax.Array:
    batch_shape = model.emissions.batch_shape
    posterior_batch = posterior.batch_shape

    if posterior_batch[: len(batch_shape)] != batch_shape:
        raise ValueError('posterior batch must begin with the model batch shape')

    replicate_shape = posterior_batch[len(batch_shape) :]
    emissions = model.emissions

    if replicate_shape:
        emissions = emissions.broadcast(
            replicate_shape,
            axis=len(batch_shape),
        )

    return emissions.observation_mean(posterior)


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
        observations: jax.Array,
        latent_dim: int,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Initialize a Gaussian LDS from one observation sequence."""
        return cls.via_pca_sequences(
            Sequences.from_sequence(
                observations,
                inputs=inputs,
                mask=mask,
            ),
            latent_dim=latent_dim,
            ridge=ridge,
            covariance_floor=covariance_floor,
        )

    @classmethod
    def via_pca_sequences(
        cls,
        data: Sequences,
        latent_dim: int,
        *,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Initialize one shared Gaussian LDS from independent sequences."""
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
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> InferenceState[typing.Self, Posterior]:
        """Compute the exact posterior for one sequence."""
        inferred = self.infer_sequences(
            Sequences.from_sequence(
                observations,
                inputs=inputs,
                mask=mask,
            )
        )

        return InferenceState(
            model=inferred.model,
            posterior=inferred.posterior.squeeze(axis=-1),
            objective=inferred.objective,
        )

    def infer_sequences(
        self,
        data: Sequences,
    ) -> InferenceState[typing.Self, Posterior]:
        """Compute exact posteriors for independent sequences."""
        inferred = _infer_exact_jit(
            self._model,
            data,
        )

        return InferenceState(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            objective=inferred.objective,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fitted[typing.Self, Posterior]:
        """Fit one sequence by expectation-maximization."""
        fit = self.fit_sequences(
            Sequences.from_sequence(
                observations,
                inputs=inputs,
                mask=mask,
            ),
            num_iters=num_iters,
            progress=progress,
        )

        return Fitted(
            model=fit.model,
            posterior=fit.posterior.squeeze(axis=-1),
            objective_trace=fit.objective_trace,
        )

    def fit_sequences(
        self,
        data: Sequences,
        *,
        num_iters: int,
        progress: bool | str = 'EM',
    ) -> Fitted[typing.Self, Posterior]:
        """Fit shared parameters across independent sequences."""
        fit = _fit_em_jit(
            self._model,
            data,
            num_iters=num_iters,
            progress=progress,
        )

        return Fitted(
            model=self.__class__(fit.state.model),
            posterior=fit.state.posterior,
            objective_trace=fit.objective_trace,
        )

    def latent_mean(self, posterior: Posterior) -> jax.Array:
        """Return posterior mean latent trajectories."""
        return posterior.means

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return posterior mean observations."""
        return _observation_mean(self._model, posterior)

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
        observations: jax.Array,
        latent_dim: int,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
    ) -> typing.Self:
        """Initialize a Poisson LDS from one observation sequence."""
        return cls.via_pca_sequences(
            Sequences.from_sequence(
                observations,
                inputs=inputs,
                mask=mask,
            ),
            latent_dim=latent_dim,
            ridge=ridge,
            covariance_floor=covariance_floor,
            count_floor=count_floor,
        )

    @classmethod
    def via_pca_sequences(
        cls,
        data: Sequences,
        latent_dim: int,
        *,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
        count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
    ) -> typing.Self:
        """Initialize one shared Poisson LDS from independent sequences."""
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
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
        initial_latents: jax.Array | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> InferenceState[typing.Self, Posterior]:
        """Compute the Laplace posterior for one sequence."""
        if initial_latents is not None:
            initial_latents = jnp.expand_dims(
                initial_latents,
                axis=len(self._model.initial.batch_shape),
            )

        inferred = self.infer_sequences(
            Sequences.from_sequence(
                observations,
                inputs=inputs,
                mask=mask,
            ),
            initial_latents=initial_latents,
            laplace_params=laplace_params,
        )

        return InferenceState(
            model=inferred.model,
            posterior=inferred.posterior.squeeze(axis=-1),
            objective=inferred.objective,
        )

    def infer_sequences(
        self,
        data: Sequences,
        *,
        initial_latents: jax.Array | None = None,
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> InferenceState[typing.Self, Posterior]:
        """Compute Laplace posteriors for independent sequences."""
        inferred = _infer_laplace_jit(
            self._model,
            data,
            initial_latents=initial_latents,
            params=laplace_params,
        )

        return InferenceState(
            model=self.__class__(inferred.model),
            posterior=inferred.posterior,
            objective=inferred.objective,
        )

    def fit(
        self,
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
        num_iters: int,
        progress: bool | str = 'Laplace EM',
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Fitted[typing.Self, Posterior]:
        """Fit one sequence with Laplace-approximated EM."""
        fit = self.fit_sequences(
            Sequences.from_sequence(
                observations,
                inputs=inputs,
                mask=mask,
            ),
            num_iters=num_iters,
            progress=progress,
            laplace_params=laplace_params,
        )

        return Fitted(
            model=fit.model,
            posterior=fit.posterior.squeeze(axis=-1),
            objective_trace=fit.objective_trace,
        )

    def fit_sequences(
        self,
        data: Sequences,
        *,
        num_iters: int,
        progress: bool | str = 'Laplace EM',
        laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    ) -> Fitted[typing.Self, Posterior]:
        """Fit shared parameters across independent sequences."""
        fit = _fit_laplace_em_jit(
            self._model,
            data,
            num_iters=num_iters,
            progress=progress,
            laplace_params=laplace_params,
        )

        return Fitted(
            model=self.__class__(fit.state.model),
            posterior=fit.state.posterior,
            objective_trace=fit.objective_trace,
        )

    def latent_mean(self, posterior: Posterior) -> jax.Array:
        """Return posterior mean latent trajectories."""
        return posterior.means

    def observation_mean(self, posterior: Posterior) -> jax.Array:
        """Return posterior mean observations."""
        return _observation_mean(self._model, posterior)

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        return self.__class__(
            _model=self._model.align(alignment),
        )
