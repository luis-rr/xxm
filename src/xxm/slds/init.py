"""Initialize switching dynamics from PCA latents and autoregressive HMMs."""

import jax
import jax.numpy as jnp

from xxm.arhmm.core import Model as ARHMMModel
from xxm.arhmm.data import ARDataset, ARDatasetCollection
from xxm.arhmm.inference import infer_exact as infer_arhmm
from xxm.arhmm.init import (
    DEFAULT_SELF_TRANSITION_PROB,
    init_gaussian_via_kmeans,
)
from xxm.arhmm.learning import fit_em as fit_arhmm
from xxm.core.data import Dataset
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.emissions.collection import ContinuousEmissionsCollection
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
    observed_rows,
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

    if bool(jnp.any(data.lengths.values <= 1)):
        raise ValueError(
            'every sequence must contain at least 2 valid steps '
            'for lag-1 switching-dynamics initialization'
        )

    if data.input_dim != 0:
        raise NotImplementedError(
            'SLDS initialization with known dynamics inputs is not yet supported'
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
    emissions: ContinuousEmissionsCollection[EmissionsT],
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
        ARDatasetCollection((latent_data,)),
    )

    state_probs = inferred.posterior.state_probs
    # (*B, R, K)

    # One validity value per AR state / target row.
    valid = latent_data.lengths.to_mask(latent_data.max_steps).values
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
        dist=arhmm.emissions.single().dist.reshape_input(
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


def masked_pca_latents(
    values: jax.Array,  # (*B, T, D_y)
    data: Dataset,
    num_states: int,
    latent_dim: int,
) -> jax.Array:  # (*B, T, D_x)
    """Validate switching initialization, then project `values` onto PCA latents.

    `values` are the observations in the space PCA should operate on, e.g. the
    raw observations or a link-transformed version of them.
    """
    _validate_initialization(data, num_states, latent_dim)
    return _masked_pca_latents(values, data, latent_dim)


def init_via_pca(
    key: jax.Array,
    latents: jax.Array,  # (*B, T, D_x)
    data: Dataset,
    emissions: ContinuousEmissionsCollection[EmissionsT],
    num_states: int,
    *,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount: float = categorical_fit.DEFAULT_PSEUDOCOUNT,
) -> Model[EmissionsT]:
    """Initialize switching structure from a K-means lag-1 AR-HMM over latents.

    One PCA space is fit from all visible valid observation rows. The auxiliary
    AR-HMM clusters complete visible latent transition windows across all
    independent sequences.
    """
    latent_data = _latent_dataset(latents, data)

    arhmm = init_gaussian_via_kmeans(
        key,
        latent_data,
        num_states,
        self_transition_prob=self_transition_prob,
        covariance_floor=covariance_floor,
    )

    return _init_from_arhmm(
        latent_data=latent_data,
        emissions=emissions,
        arhmm=arhmm,
        covariance_floor=covariance_floor,
        pseudocount=pseudocount,
    )


def init_via_arhmm(
    key: jax.Array,
    latents: jax.Array,  # (*B, T, D_x)
    data: Dataset,
    emissions: ContinuousEmissionsCollection[EmissionsT],
    num_states: int,
    *,
    num_arhmm_iters: int = 10,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount: float = categorical_fit.DEFAULT_PSEUDOCOUNT,
    progress: bool | str = 'AR-HMM',
) -> Model[EmissionsT]:
    """Initialize switching structure by fitting a lag-1 AR-HMM to latents."""
    latent_data = _latent_dataset(latents, data)

    arhmm = init_gaussian_via_kmeans(
        key,
        latent_data,
        num_states,
        self_transition_prob=self_transition_prob,
        covariance_floor=covariance_floor,
    )

    arhmm = fit_arhmm(
        arhmm,
        ARDatasetCollection((latent_data,)),
        num_iters=num_arhmm_iters,
        progress=progress,
    ).state.model

    return _init_from_arhmm(
        latent_data=latent_data,
        emissions=emissions,
        arhmm=arhmm,
        covariance_floor=covariance_floor,
        pseudocount=pseudocount,
    )


def _gaussian_latents(
    data: Dataset,
    num_states: int,
    latent_dim: int,
    covariance_floor: float,
) -> tuple[jax.Array, ContinuousEmissionsCollection[GaussianEmissions]]:
    """PCA latents of the observations and Gaussian emissions fitted to them."""
    latents = masked_pca_latents(data.observations.values, data, num_states, latent_dim)
    latent_rows, observation_rows = observed_rows(latents, data)
    emissions = GaussianEmissions.from_latents(
        latent_rows,
        observation_rows,
        covariance_floor=covariance_floor,
    )
    return latents, ContinuousEmissionsCollection.from_single(emissions)


def _poisson_latents(
    data: Dataset,
    num_states: int,
    latent_dim: int,
    count_floor: float,
) -> tuple[jax.Array, ContinuousEmissionsCollection[PoissonEmissions]]:
    """PCA latents of stabilized log counts and Poisson emissions fitted to them."""
    log_counts = poisson_fit.inverse_exp_link(
        data.weighted_observations().safe_values(),
        count_floor=count_floor,
    )
    latents = masked_pca_latents(log_counts, data, num_states, latent_dim)
    latent_rows, observation_rows = observed_rows(latents, data)
    emissions = PoissonEmissions.from_latents(latent_rows, observation_rows)
    return latents, ContinuousEmissionsCollection.from_single(emissions)


def init_gaussian_via_pca(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    latent_dim: int,
    *,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount: float = categorical_fit.DEFAULT_PSEUDOCOUNT,
) -> Model[GaussianEmissions]:
    """Initialize a Gaussian SLDS from PCA latents and a K-means AR-HMM."""
    latents, emissions = _gaussian_latents(
        data, num_states, latent_dim, covariance_floor
    )
    return init_via_pca(
        key,
        latents,
        data,
        emissions,
        num_states,
        self_transition_prob=self_transition_prob,
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
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount: float = categorical_fit.DEFAULT_PSEUDOCOUNT,
    progress: bool | str = 'AR-HMM',
) -> Model[GaussianEmissions]:
    """Initialize a Gaussian SLDS by fitting a lag-1 AR-HMM to PCA latents."""
    latents, emissions = _gaussian_latents(
        data, num_states, latent_dim, covariance_floor
    )
    return init_via_arhmm(
        key,
        latents,
        data,
        emissions,
        num_states,
        num_arhmm_iters=num_arhmm_iters,
        self_transition_prob=self_transition_prob,
        covariance_floor=covariance_floor,
        pseudocount=pseudocount,
        progress=progress,
    )


def init_poisson_via_pca(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    latent_dim: int,
    *,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount: float = categorical_fit.DEFAULT_PSEUDOCOUNT,
    count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
) -> Model[PoissonEmissions]:
    """Initialize a Poisson SLDS from inverse-link PCA latents and a K-means AR-HMM."""
    latents, emissions = _poisson_latents(data, num_states, latent_dim, count_floor)
    return init_via_pca(
        key,
        latents,
        data,
        emissions,
        num_states,
        self_transition_prob=self_transition_prob,
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
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount: float = categorical_fit.DEFAULT_PSEUDOCOUNT,
    count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
    progress: bool | str = 'AR-HMM',
) -> Model[PoissonEmissions]:
    """Initialize a Poisson SLDS by fitting an AR-HMM to inverse-link PCA latents."""
    latents, emissions = _poisson_latents(data, num_states, latent_dim, count_floor)
    return init_via_arhmm(
        key,
        latents,
        data,
        emissions,
        num_states,
        num_arhmm_iters=num_arhmm_iters,
        self_transition_prob=self_transition_prob,
        covariance_floor=covariance_floor,
        pseudocount=pseudocount,
        progress=progress,
    )
