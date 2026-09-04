import jax
import jax.numpy as jnp

from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.emissions.continuous import (
    EmissionsT,
    GaussianEmissions,
)
from xxm.core.emissions.discrete_ar import AREmissions
from xxm.core.models.discrete import CategoricalInitial
from xxm.core.models.gaussian import StateConditionedGaussian
from xxm.core.optim import gaussian as gaussian_fit
from xxm.hmm.core import Model as HMMModel
from xxm.hmm.inference import infer_exact as infer_hmm
from xxm.hmm.init import init_gaussian_ar
from xxm.hmm.learning import fit_em as fit_hmm
from xxm.lds.init import pca_latents

from .core import GaussianLinearSwitchingDynamics, Model


def _validate_initialization(
    observations: jax.Array,
    num_states: int,
    latent_dim: int,
) -> None:
    if observations.ndim != 2:
        raise ValueError('observations must have shape (T, N)')

    num_steps, observation_dim = observations.shape

    if num_states < 1:
        raise ValueError('num_states must be at least 1')

    if latent_dim < 1 or latent_dim > observation_dim:
        raise ValueError('latent_dim must be between 1 and the observation dimension')

    if num_steps < latent_dim + 2:
        raise ValueError(
            'At least latent_dim + 2 time steps are required to initialize the SLDS'
        )

    if num_steps - 1 < num_states:
        raise ValueError(
            'At least num_states + 1 time steps are required '
            'to initialize the switching dynamics'
        )


def _add_covariance_floor(
    covariance: jax.Array,
    covariance_floor: float,
    reference: jax.Array,
) -> jax.Array:
    """Add an isotropic floor relative to the typical latent variance."""
    scale = jnp.mean(
        jnp.var(
            reference,
            axis=0,
        )
    )

    identity = jnp.eye(
        covariance.shape[-1],
        dtype=covariance.dtype,
    )

    return covariance + covariance_floor * scale * identity


def _state_conditioned_initial_from_latents(
    latents: jax.Array,  # (T, D)
    state_probs: jax.Array,  # (T, K)
    covariance_floor: float,
) -> StateConditionedGaussian:
    """Estimate state-conditioned latent Gaussians from weighted samples."""
    gaussian = gaussian_fit.from_samples_weighted(
        values=latents,
        weights=state_probs,
    )

    covariance = _add_covariance_floor(
        gaussian.covariance,
        covariance_floor,
        reference=latents,
    )

    return StateConditionedGaussian(
        model=gaussian._replace(
            covariance=covariance,
        )
    )


def _from_arhmm(
    latents: jax.Array,
    emissions: EmissionsT,
    arhmm: HMMModel[AREmissions[LinearGaussian]],
    covariance_floor: float,
) -> Model[EmissionsT]:
    """
    Construct an SLDS from a lag-1 Gaussian AR-HMM over latent values.

    AR-HMM states index the regressions producing ``latents[1:]`` and therefore
    correspond to SLDS states ``z[1:]`` under the incoming-state convention.
    Trajectory-wide state occupancies are used as proxies for the boundary
    distributions of ``z[0]`` and ``x[0] | z[0]``.
    """
    posterior, _ = infer_hmm(
        arhmm,
        latents,
    )

    state_probs = posterior.state_probs  # (T-1, K), aligned with latents[1:]

    state_initial = CategoricalInitial(
        model=Categorical.from_counts(
            jnp.sum(
                state_probs,
                axis=0,
            )
        )
    )

    latent_initial = _state_conditioned_initial_from_latents(
        latents=latents[1:],
        state_probs=state_probs,
        covariance_floor=covariance_floor,
    )

    dynamics = GaussianLinearSwitchingDynamics(
        model=arhmm.emissions.model.reshape_input((latents.shape[-1],))
    )

    return Model(
        state_initial=state_initial,
        transitions=arhmm.transitions,
        latent_initial=latent_initial,
        dynamics=dynamics,
        emissions=emissions,
    )


def _initialize_pca_arhmm(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
    latent_dim: int,
    self_transition_prob: float,
) -> tuple[jax.Array, HMMModel[AREmissions[LinearGaussian]]]:
    """Initialize PCA latents and a lag-1 AR-HMM."""
    _validate_initialization(
        observations,
        num_states,
        latent_dim,
    )

    latents = pca_latents(observations, latent_dim)

    arhmm = init_gaussian_ar(
        key=key,
        observations=latents,
        num_states=num_states,
        num_lags=1,
        self_transition_prob=self_transition_prob,
    )

    return latents, arhmm


def init_pca_gaussian(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
    latent_dim: int,
    *,
    self_transition_prob: float = 0.9,
    covariance_floor: float = 1e-2,
) -> Model[GaussianEmissions]:
    """
    Initialize a Gaussian SLDS from a PCA latent representation.

    PCA initializes the continuous latent trajectory and observation model.
    The latent transitions are then clustered by the Gaussian AR-HMM
    initializer to obtain state-specific linear dynamics, without running
    AR-HMM EM.
    """
    latents, arhmm = _initialize_pca_arhmm(
        key=key,
        observations=observations,
        num_states=num_states,
        latent_dim=latent_dim,
        self_transition_prob=self_transition_prob,
    )

    emissions = GaussianEmissions.from_latents(
        latents=latents,
        observations=observations,
        covariance_floor=covariance_floor,
    )

    return _from_arhmm(
        latents=latents,
        emissions=emissions,
        arhmm=arhmm,
        covariance_floor=covariance_floor,
    )


def init_arhmm_gaussian(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
    latent_dim: int,
    *,
    num_arhmm_iters: int = 10,
    self_transition_prob: float = 0.9,
    covariance_floor: float = 1e-2,
    progress: bool | str = 'AR-HMM',
) -> Model[GaussianEmissions]:
    """
    Initialize a Gaussian SLDS by fitting an AR-HMM to PCA latents.

    PCA initializes the continuous latent trajectory and observation model.
    A lag-1 Gaussian AR-HMM is initialized on that trajectory and refined with
    EM before its transition and dynamics parameters are transferred to the
    SLDS. State occupancies across the fitted latent trajectory initialize the
    boundary distributions of ``z[0]`` and ``x[0] | z[0]``.
    """
    latents, arhmm = _initialize_pca_arhmm(
        key=key,
        observations=observations,
        num_states=num_states,
        latent_dim=latent_dim,
        self_transition_prob=self_transition_prob,
    )

    emissions = GaussianEmissions.from_latents(
        latents=latents,
        observations=observations,
        covariance_floor=covariance_floor,
    )

    arhmm = fit_hmm(
        arhmm,
        latents,
        num_iters=num_arhmm_iters,
        progress=progress,
    ).model

    return _from_arhmm(
        latents=latents,
        emissions=emissions,
        arhmm=arhmm,
        covariance_floor=covariance_floor,
    )
