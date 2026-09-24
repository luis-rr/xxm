"""Initialize switching dynamics from PCA latents and autoregressive HMMs."""

import jax
import jax.numpy as jnp

from xxm.arhmm.core import Model as ARHMMModel
from xxm.arhmm.data import ARDataset
from xxm.arhmm.inference import infer_exact as infer_arhmm
from xxm.arhmm.init import (
    DEFAULT_SELF_TRANSITION_PROB,
    init_gaussian_via_kmeans,
)
from xxm.arhmm.learning import fit_em as fit_arhmm
from xxm.core.data import Dataset
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.emissions.continuous import (
    EmissionsT,
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.latents.discrete import CategoricalInitial
from xxm.core.latents.gaussian import StateConditionedGaussian
from xxm.core.latents.switching import GaussianLinearSwitchingDynamics
from xxm.core.optim import categorical as categorical_fit
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.batch import filter_valid_batches
from xxm.lds.init import (
    _masked_pca_latents,
)
from xxm.lds.init import (
    _validate_initialization as _validate_pca_initialization,
)

from .core import Model


def _validate_initialization(
    data: Dataset,
    num_states: int,
    latent_dim: int,
) -> None:
    """Validate requirements specific to AR-HMM-based SLDS initialization."""
    if num_states < 1:
        raise ValueError('num_states must be at least 1')

    _validate_pca_initialization(
        data,
        latent_dim,
    )

    if bool(jnp.any(data.lengths <= 1)):
        raise ValueError(
            'every sequence must contain at least 2 valid steps '
            'for lag-1 switching-dynamics initialization'
        )

    if data.input_dim != 0:
        raise NotImplementedError(
            'SLDS initialization with known dynamics inputs is not yet supported'
        )


def _gaussian_latents(
    data: Dataset,
    latent_dim: int,
) -> jax.Array:
    """Fit one PCA space from visible rows and project the padded dataset."""
    return _masked_pca_latents(
        data.observations.values,
        data,
        latent_dim,
    )


def _poisson_latents(
    data: Dataset,
    latent_dim: int,
    count_floor: float,
) -> jax.Array:
    """Fit PCA latents from stabilized visible log-count observations."""
    transformed = poisson_fit.inverse_exp_link(
        data.weighted_observations().safe_values(),
        count_floor=count_floor,
    )

    return _masked_pca_latents(
        transformed,
        data,
        latent_dim,
    )


def _latent_dataset(
    latents: jax.Array,
    data: Dataset,
) -> ARDataset:
    """
    Wrap PCA latents as autonomous pseudo-observations for the auxiliary AR-HMM.

    Source lengths and visibility are preserved. Source external inputs are not
    forwarded: the auxiliary lag-1 AR-HMM is used only to discover switching
    structure in latent dynamics.
    """
    data = Dataset.from_padded(
        observations=latents,
        lengths=data.lengths,
        visible=data.visible,
    )

    return ARDataset.from_dataset(
        data,
        num_lags=1,
    )


def _init_gaussian_emissions(
    latents: jax.Array,
    data: Dataset,
    covariance_floor: float,
) -> GaussianEmissions:
    """Fit Gaussian emissions from visible valid PCA rows."""
    observed = data.observation_weights()

    return GaussianEmissions.from_latents(
        latents=latents[observed],
        observations=data.observations.values[observed],
        covariance_floor=covariance_floor,
    )


def _init_poisson_emissions(
    latents: jax.Array,
    data: Dataset,
) -> PoissonEmissions:
    """Fit Poisson emissions from visible valid PCA rows."""
    observed = data.observation_weights()

    return PoissonEmissions.from_latents(
        latents=latents[observed],
        observations=data.observations.values[observed],
    )


# TODO: ARDataset currently exposes complete-window visibility, so this uses
# only targets whose full AR history is also visible. Revisit whether
# ARDataset should retain target-only visibility separately, allowing this
# marginal latent-initial fit to use visible x_t even when x_{t-1:t-L} is hidden.
def _init_state_conditioned_initial_from_latents(
    latents: jax.Array,  # (*B, R, D)
    state_probs: jax.Array,  # (*B, R, K)
    weights: jax.Array,  # (*B, R)
    covariance_floor: float,
) -> StateConditionedGaussian:
    """
    Estimate state-conditioned latent Gaussians from accessible latent samples.

    Batch and time axes are pooled. States with insufficient effective weight
    fall back to the global latent Gaussian.
    """
    latent_dim = latents.shape[-1]
    num_states = state_probs.shape[-1]

    if latents.shape[:-1] != state_probs.shape[:-1]:
        raise ValueError(
            'latents and state probabilities must share batch and time dimensions'
        )

    if weights.shape != latents.shape[:-1]:
        raise ValueError('weights must match latent batch and time dimensions')

    values = latents.reshape((-1, latent_dim))

    flat_weights = weights.reshape(-1)

    state_weights = (state_probs * weights[..., None]).reshape((-1, num_states))
    state_weights = jnp.moveaxis(
        state_weights,
        -1,
        0,
    )  # (K, S)

    fitted = gaussian_fit.from_samples_weighted(
        values=values,
        weights=state_weights,
        covariance_floor=covariance_floor,
    )

    fallback = gaussian_fit.from_samples_weighted(
        values=values,
        weights=flat_weights,
        covariance_floor=covariance_floor,
    ).broadcast(
        (num_states,),
        axis=0,
    )

    fitted = filter_valid_batches(
        fitted,
        fallback,
        state_weights,
        min_expected_count=1.0,
    )

    return StateConditionedGaussian(
        dist=fitted,
    )


def _init_from_arhmm(
    latent_data: ARDataset,
    emissions: EmissionsT,
    arhmm: ARHMMModel[LinearGaussian],
    covariance_floor: float,
    pseudocount: float,
) -> Model[EmissionsT]:
    """
    Construct an SLDS from a lag-1 Gaussian AR-HMM over latent values.

    ``latent_data`` is already the prepared lag-1 AR representation.
    Its compact observation row ``r`` is latent ``x[..., r + 1, :]``,
    and AR-HMM posterior state ``r`` therefore corresponds to SLDS
    switching state ``z[..., r + 1]`` under the incoming-state convention.

    Valid AR-state occupancies initialize the proxy boundary distribution
    over ``z[0]``. Accessible prepared latent targets initialize the proxy
    state-conditioned boundary distribution ``p(x[0] | z[0])``.
    """
    if arhmm.num_lags != 1:
        raise ValueError('SLDS initialization requires a lag-1 Gaussian AR-HMM')

    inferred = infer_arhmm(
        arhmm,
        latent_data,
    )

    state_probs = inferred.posterior.state_probs
    # (*B, R, K)

    # One validity value per AR state / target row.
    valid = latent_data.valid().materialize(latent_data.num_steps)
    # (*B, R)

    if state_probs.shape[:-1] != valid.shape:
        raise ValueError(
            'AR-HMM posterior must align with the prepared lag-1 dataset. '
            f'Got {state_probs.shape[:-1]} vs {valid.shape}'
        )

    # Proxy p(z_0) from occupancy across all structurally valid AR states.
    #
    # Visibility does not remove a latent state, so use structural validity,
    # not observation weights.
    state_counts = jnp.sum(
        state_probs * valid[..., None],
        axis=tuple(range(state_probs.ndim - 1)),
    )

    state_initial = CategoricalInitial(
        dist=categorical_fit.from_counts(
            state_counts,
            pseudocount=pseudocount,
        )
    )

    # ARDataset observations are ALREADY the lag-1 targets x[..., 1:, :].
    latent_targets = latent_data.observations.values
    # (*B, R, D_x)

    # Conservative initialization: use rows whose complete AR window is
    # available. No additional [..., 1:] shift belongs here.
    target_weights = latent_data.observation_weights()
    # (*B, R)

    latent_initial = _init_state_conditioned_initial_from_latents(
        latents=latent_targets,
        state_probs=state_probs,
        weights=target_weights,
        covariance_floor=covariance_floor,
    )

    dynamics = GaussianLinearSwitchingDynamics(
        dist=arhmm.emissions.dist.reshape_input(
            (latent_data.observation_dim,),
        )
    )

    return Model(
        state_initial=state_initial,
        transitions=arhmm.transitions,
        latent_initial=latent_initial,
        dynamics=dynamics,
        emissions=emissions,
    )


def init_gaussian_via_pca(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    latent_dim: int,
    *,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount=categorical_fit.DEFAULT_PSEUDOCOUNT,
) -> Model[GaussianEmissions]:
    """
    Initialize a Gaussian SLDS from PCA latents and a lag-1 Gaussian AR-HMM.

    One PCA space is fit from all visible valid observation rows. The auxiliary
    AR-HMM clusters complete visible latent transition windows across all
    independent sequences.
    """
    _validate_initialization(
        data,
        num_states,
        latent_dim,
    )

    latents = _gaussian_latents(
        data,
        latent_dim,
    )

    latent_data = _latent_dataset(
        latents,
        data,
    )

    arhmm = init_gaussian_via_kmeans(
        key=key,
        data=latent_data,
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        covariance_floor=covariance_floor,
    )

    emissions = _init_gaussian_emissions(
        latents,
        data,
        covariance_floor,
    )

    return _init_from_arhmm(
        latent_data=latent_data,
        emissions=emissions,
        arhmm=arhmm,
        covariance_floor=covariance_floor,
        pseudocount=pseudocount,
    )


def init_gaussian_via_arhmm(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    latent_dim: int,
    *,
    num_arhmm_iters: int = 10,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount=categorical_fit.DEFAULT_PSEUDOCOUNT,
    progress: bool | str = 'AR-HMM',
) -> Model[GaussianEmissions]:
    """
    Initialize a Gaussian SLDS by fitting a lag-1 AR-HMM to PCA latents.
    """
    _validate_initialization(
        data,
        num_states,
        latent_dim,
    )

    latents = _gaussian_latents(
        data,
        latent_dim,
    )

    latent_data = _latent_dataset(
        latents,
        data,
    )

    arhmm = init_gaussian_via_kmeans(
        key=key,
        data=latent_data,
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        covariance_floor=covariance_floor,
    )

    arhmm = fit_arhmm(
        arhmm,
        latent_data,
        num_iters=num_arhmm_iters,
        progress=progress,
    ).state.model

    emissions = _init_gaussian_emissions(
        latents,
        data,
        covariance_floor,
    )

    return _init_from_arhmm(
        latent_data=latent_data,
        emissions=emissions,
        arhmm=arhmm,
        covariance_floor=covariance_floor,
        pseudocount=pseudocount,
    )


def init_poisson_via_pca(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    latent_dim: int,
    *,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount=categorical_fit.DEFAULT_PSEUDOCOUNT,
    count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
) -> Model[PoissonEmissions]:
    """Initialize a Poisson SLDS from inverse-link PCA latents."""
    _validate_initialization(
        data,
        num_states,
        latent_dim,
    )

    latents = _poisson_latents(
        data,
        latent_dim,
        count_floor,
    )

    latent_data = _latent_dataset(
        latents,
        data,
    )

    arhmm = init_gaussian_via_kmeans(
        key=key,
        data=latent_data,
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        covariance_floor=covariance_floor,
    )

    emissions = _init_poisson_emissions(
        latents,
        data,
    )

    return _init_from_arhmm(
        latent_data=latent_data,
        emissions=emissions,
        arhmm=arhmm,
        covariance_floor=covariance_floor,
        pseudocount=pseudocount,
    )


def init_poisson_via_arhmm(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    latent_dim: int,
    *,
    num_arhmm_iters: int = 10,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount=categorical_fit.DEFAULT_PSEUDOCOUNT,
    count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
    progress: bool | str = 'AR-HMM',
) -> Model[PoissonEmissions]:
    """Initialize a Poisson SLDS by fitting an AR-HMM to inverse-link PCA latents."""
    _validate_initialization(
        data,
        num_states,
        latent_dim,
    )

    latents = _poisson_latents(
        data,
        latent_dim,
        count_floor,
    )

    latent_data = _latent_dataset(
        latents,
        data,
    )

    arhmm = init_gaussian_via_kmeans(
        key=key,
        data=latent_data,
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        covariance_floor=covariance_floor,
    )

    arhmm = fit_arhmm(
        arhmm,
        latent_data,
        num_iters=num_arhmm_iters,
        progress=progress,
    ).state.model

    emissions = _init_poisson_emissions(
        latents,
        data,
    )

    return _init_from_arhmm(
        latent_data=latent_data,
        emissions=emissions,
        arhmm=arhmm,
        covariance_floor=covariance_floor,
        pseudocount=pseudocount,
    )
