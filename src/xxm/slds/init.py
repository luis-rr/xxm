"""Initialize switching dynamics from PCA pseudo-latents and autoregressive HMMs."""

import typing

import jax
import jax.numpy as jnp

from xxm.arhmm.core import Model as ARHMMModel
from xxm.arhmm.core import Posterior as ARHMMPosterior
from xxm.arhmm.data import ARDatasetCollection
from xxm.core.data import Dataset, DatasetCollection
from xxm.core.dists.gaussian import (
    FullLinearGaussian,
    LinearGaussian,
)
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
from xxm.core.optim import gaussian_full as gaussian_full_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.batch import filter_valid_batches
from xxm.core.optim.gaussian import CovarianceType
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
    latent_dim: int,
) -> None:
    """Validate requirements specific to PCA pseudo-latent SLDS initialization."""
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


def _pca_pseudo_latents(
    values: jax.Array,  # (*B, T, D_y)
    data: Dataset,
    latent_dim: int,
) -> Dataset:
    """
    Project `values` onto PCA pseudo-latents, wrapped as pseudo-observations.

    One PCA space is fit from all visible valid rows. Source lengths and
    visibility are preserved; inputs are not forwarded, since the auxiliary
    AR-HMM only discovers switching structure in latent dynamics.
    """
    _validate_initialization(data, latent_dim)
    return Dataset.from_padded(
        observations=_masked_pca_latents(values, data, latent_dim),
        lengths=data.lengths,
        visible=data.visible,
    )


# TODO: ARDataset currently exposes complete-window visibility, so this uses
# only targets whose full AR history is also visible. Revisit whether
# ARDataset should retain target-only visibility separately, allowing this
# marginal latent-initial fit to use visible x_t even when x_{t-1:t-L} is hidden.
def _init_state_conditioned_initial_from_pseudo_latents(
    pseudo_latents: jax.Array,  # (*B, R, D)
    state_probs: jax.Array,  # (*B, R, K)
    weights: jax.Array,  # (*B, R)
    covariance_floor: float,
) -> StateConditionedGaussian:
    """
    Estimate state-conditioned latent Gaussians from accessible pseudo-latents.

    Batch and time axes are pooled. States with insufficient effective weight
    fall back to the global pseudo-latent Gaussian.
    """
    latent_dim = pseudo_latents.shape[-1]
    num_states = state_probs.shape[-1]

    if pseudo_latents.shape[:-1] != state_probs.shape[:-1]:
        raise ValueError(
            'pseudo-latents and state probabilities must share batch and time dimensions'
        )

    if weights.shape != pseudo_latents.shape[:-1]:
        raise ValueError('weights must match pseudo-latent batch and time dimensions')

    values = pseudo_latents.reshape((-1, latent_dim))

    flat_weights = weights.reshape(-1)

    state_weights = (state_probs * weights[..., None]).reshape((-1, num_states))
    state_weights = jnp.moveaxis(
        state_weights,
        -1,
        0,
    )  # (K, S)

    fitted = gaussian_full_fit.from_samples_weighted(
        values=values,
        weights=state_weights,
        covariance_floor=covariance_floor,
    )

    fallback = gaussian_full_fit.from_samples_weighted(
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


def _full_dynamics(dist: LinearGaussian) -> FullLinearGaussian:
    """Promote diagonal AR-HMM noise to the full noise of SLDS dynamics."""
    if isinstance(dist, FullLinearGaussian):
        return dist

    return FullLinearGaussian.from_components(
        dist.affine.coefficients,
        dist.affine.bias,
        dist.variances[..., None] * jnp.eye(dist.output_dim, dtype=dist.dtype),
    )


def init_from_arhmm(
    arhmm: ARHMMModel[LinearGaussian],
    pseudo_latents: DatasetCollection,
    posterior: ARHMMPosterior,
    emissions: ContinuousEmissionsCollection[EmissionsT],
    *,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    pseudocount: float = categorical_fit.DEFAULT_PSEUDOCOUNT,
) -> Model[EmissionsT]:
    """
    Construct an SLDS from a lag-1 Gaussian AR-HMM over pseudo-latents.

    `posterior` is the AR-HMM posterior over `pseudo_latents`, and `emissions`
    map each session's pseudo-latents to its observations. AR-HMM transitions
    and regression become the SLDS transitions and dynamics; diagonal AR-HMM
    noise is promoted to full.

    In the lag-1 AR representation, compact row ``r`` is pseudo-latent
    ``x[..., r + 1, :]``, and AR-HMM posterior state ``r`` therefore
    corresponds to SLDS switching state ``z[..., r + 1]`` under the
    incoming-state convention.

    Valid AR-state occupancies initialize the proxy boundary distribution
    over ``z[0]``. Accessible pseudo-latent targets initialize the proxy
    state-conditioned boundary distribution ``p(x[0] | z[0])``. Both pool all
    sessions; SLDS dynamics are shared, so the AR-HMM must have one regression.
    """
    if arhmm.num_lags != 1:
        raise ValueError('SLDS initialization requires a lag-1 Gaussian AR-HMM')

    dist = arhmm.emissions.single().dist

    if len(emissions.emissions) != len(pseudo_latents.datasets):
        raise ValueError(
            'expected emissions for every pseudo-latent session; got '
            f'{len(emissions.emissions)} for {len(pseudo_latents.datasets)}'
        )

    if any(dim != dist.output_dim for dim in pseudo_latents.observation_dims):
        raise ValueError(
            f'AR-HMM observation dimension {dist.output_dim} must match '
            f'pseudo-latent dimensions {pseudo_latents.observation_dims}'
        )

    ar_data = ARDatasetCollection.from_datasets(pseudo_latents, num_lags=1)

    state_probs = posterior.state_probs
    # (N, R, K), sequences pooled across sessions

    # One validity value per AR state / target row.
    valid = jnp.concatenate(
        [
            dataset.lengths.to_mask(dataset.max_steps).values
            for dataset in ar_data.datasets
        ]
    )
    # (N, R)

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
    # Conservative initialization: use rows whose complete AR window is
    # available. No additional [..., 1:] shift belongs here.
    latent_initial = _init_state_conditioned_initial_from_pseudo_latents(
        pseudo_latents=jnp.concatenate(
            [dataset.observations.values for dataset in ar_data.datasets]
        ),  # (N, R, D_x)
        state_probs=state_probs,
        weights=jnp.concatenate(
            [dataset.observation_weights() for dataset in ar_data.datasets]
        ),  # (N, R)
        covariance_floor=covariance_floor,
    )

    dynamics = GaussianLinearSwitchingDynamics(
        dist=_full_dynamics(dist).reshape_input((dist.output_dim,))
    )

    return Model(
        state_initial=state_initial,
        transitions=arhmm.transitions,
        latent_initial=latent_initial,
        dynamics=dynamics,
        emissions=emissions,
    )


def _pseudo_latents_via_pca(
    data: DatasetCollection,
    latent_dim: int,
    transform: typing.Callable[[Dataset], jax.Array],
) -> DatasetCollection:
    """Project each session's transformed observations onto its own PCA space."""
    return DatasetCollection(
        tuple(
            _pca_pseudo_latents(transform(dataset), dataset, latent_dim)
            for dataset in data.datasets
        )
    )


def pseudo_latents_via_pca(
    data: DatasetCollection,
    latent_dim: int,
) -> DatasetCollection:
    """
    PCA pseudo-latents of the observations, estimated independently per session.

    Sessions do not share a PCA space, so their pseudo-latent coordinates are
    unrelated.
    """
    return _pseudo_latents_via_pca(
        data,
        latent_dim,
        lambda dataset: dataset.observations.values,
    )


def pseudo_latents_via_log_pca(
    data: DatasetCollection,
    latent_dim: int,
    *,
    count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
) -> DatasetCollection:
    """
    PCA pseudo-latents of stabilized log counts, estimated independently per session.

    Sessions do not share a PCA space, so their pseudo-latent coordinates are
    unrelated.
    """
    return _pseudo_latents_via_pca(
        data,
        latent_dim,
        lambda dataset: poisson_fit.inverse_exp_link(
            dataset.weighted_observations().safe_values(),
            count_floor=count_floor,
        ),
    )


def _observed_session_rows(
    pseudo_latents: DatasetCollection,
    data: DatasetCollection,
) -> tuple[tuple[jax.Array, jax.Array], ...]:
    """Pair each session's observed pseudo-latent and observation rows."""
    if len(pseudo_latents.datasets) != len(data.datasets):
        raise ValueError(
            'pseudo-latents and data must have the same number of sessions; got '
            f'{len(pseudo_latents.datasets)} and {len(data.datasets)}'
        )

    if data.input_dim != 0:
        raise NotImplementedError(
            'SLDS initialization with known dynamics inputs is not yet supported'
        )

    for session, (latents, dataset) in enumerate(
        zip(pseudo_latents.datasets, data.datasets, strict=True)
    ):
        if not (
            latents.batch_shape == dataset.batch_shape
            and latents.max_steps == dataset.max_steps
            and bool(jnp.array_equal(latents.lengths.values, dataset.lengths.values))
            and bool(jnp.array_equal(latents.visible.values, dataset.visible.values))
        ):
            raise ValueError(
                f'session {session}: pseudo-latents and data must share '
                'batch shape, lengths, and visibility'
            )

    return tuple(
        observed_rows(latents.observations.values, dataset)
        for latents, dataset in zip(pseudo_latents.datasets, data.datasets, strict=True)
    )


def gaussian_emissions_from_pseudo_latents(
    pseudo_latents: DatasetCollection,
    data: DatasetCollection,
    *,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    noise: CovarianceType,
) -> ContinuousEmissionsCollection[GaussianEmissions]:
    """Regress each session's observations on its pseudo-latents."""
    return ContinuousEmissionsCollection(
        tuple(
            GaussianEmissions.from_latents(
                latent_rows,
                observation_rows,
                covariance_floor=covariance_floor,
                noise=noise,
            )
            for latent_rows, observation_rows in _observed_session_rows(
                pseudo_latents, data
            )
        )
    )


def poisson_emissions_from_pseudo_latents(
    pseudo_latents: DatasetCollection,
    data: DatasetCollection,
) -> ContinuousEmissionsCollection[PoissonEmissions]:
    """Poisson-regress each session's counts on its pseudo-latents."""
    return ContinuousEmissionsCollection(
        tuple(
            PoissonEmissions.from_latents(latent_rows, observation_rows)
            for latent_rows, observation_rows in _observed_session_rows(
                pseudo_latents, data
            )
        )
    )
