"""Temporary observation completion for emission-based latent initialization."""

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions

_Emissions = GaussianEmissions | PoissonEmissions


def _time_axis(values: jax.Array) -> int:
    """Return the nonnegative time-axis index for `(*B, T, D)` values."""
    return values.ndim - 2


def _validate_inputs(
    emissions: _Emissions,
    observations: jax.Array,
    visible: jax.Array | None,
    num_refinement_iters: int,
) -> jax.Array | None:
    """Validate initializer inputs and broadcast visibility to observations."""
    if observations.ndim < 2 or observations.shape[-2] < 1:
        raise ValueError('observations must have shape (*B, T, D_y) with T >= 1')

    batch.require_same_shape(
        emissions.batch_shape,
        observations.shape[:-2],
    )

    if observations.shape[-1] != emissions.dist.output_dim:
        raise ValueError('observations must match the emission output dimension')

    if num_refinement_iters < 0:
        raise ValueError('num_refinement_iters must be nonnegative')

    if visible is None:
        return None

    visible = jnp.asarray(visible)

    if visible.dtype != jnp.bool_:
        raise ValueError('visibility must have boolean dtype')

    return jnp.broadcast_to(
        visible,
        observations.shape,
    )


def _nearest_visible_indices(
    visible: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return nearest visible time indices on the left and right."""
    num_steps = visible.shape[-2]
    time_axis = _time_axis(visible)

    times = jnp.arange(num_steps)[:, None]

    previous = jax.lax.associative_scan(
        jnp.maximum,
        jnp.where(
            visible,
            times,
            -1,
        ),
        axis=time_axis,
    )

    following = jax.lax.associative_scan(
        jnp.minimum,
        jnp.where(
            visible,
            times,
            num_steps,
        ),
        axis=time_axis,
        reverse=True,
    )

    left = jnp.where(
        previous >= 0,
        previous,
        following,
    )

    right = jnp.where(
        following < num_steps,
        following,
        previous,
    )

    return (
        jnp.clip(
            left,
            0,
            num_steps - 1,
        ),
        jnp.clip(
            right,
            0,
            num_steps - 1,
        ),
    )


def _gather_time(
    values: jax.Array,
    indices: jax.Array,
) -> jax.Array:
    """Gather values at per-element time indices."""
    return jnp.take_along_axis(
        values,
        indices,
        axis=_time_axis(values),
    )


def _interpolation_fraction(
    left: jax.Array,
    right: jax.Array,
    *,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    """Return linear interpolation weights between neighboring times."""
    num_steps = left.shape[-2]

    times = jnp.arange(num_steps)[:, None]
    span = jnp.maximum(
        right - left,
        1,
    )

    return jnp.where(
        right > left,
        (times - left) / span,
        0,
    ).astype(dtype)


def _interpolate_observations(
    observations: jax.Array,
    visible: jax.Array,
    baseline: jax.Array,
) -> jax.Array:
    """Complete hidden observations by temporal interpolation per channel.

    Interior gaps are linearly interpolated. Boundary gaps use the nearest
    visible value. Channels with no visible samples use `baseline`. Hidden
    numerical values are discarded before gathering or interpolation.
    """
    left, right = _nearest_visible_indices(
        visible,
    )

    safe_observations = jnp.where(
        visible,
        observations,
        0,
    )

    left_values = _gather_time(
        safe_observations,
        left,
    )

    right_values = _gather_time(
        safe_observations,
        right,
    )

    fraction = _interpolation_fraction(
        left,
        right,
        dtype=baseline.dtype,
    )

    interpolated = (1 - fraction) * left_values + fraction * right_values

    has_visible = jnp.any(
        visible,
        axis=_time_axis(visible),
        keepdims=True,
    )

    completed_hidden = jnp.where(
        has_visible,
        interpolated,
        baseline,
    )

    return jnp.where(
        visible,
        observations,
        completed_hidden,
    )


def _baseline_observations(
    emissions: _Emissions,
    observations: jax.Array,
) -> jax.Array:
    """Return the emission mean at zero latent for each model batch."""
    dtype = jnp.result_type(
        observations,
        emissions.dist.affine.dtype,
        jnp.float32,
    )

    zero_latents = jnp.zeros(
        (
            *emissions.batch_shape,
            1,
            *emissions.dist.input_shape,
        ),
        dtype=dtype,
    )

    return emissions.conditional_mean(
        zero_latents,
    )


def _refine_completed_observations(
    emissions: _Emissions,
    observations: jax.Array,
    visible: jax.Array,
    completed: jax.Array,
    *,
    num_iters: int,
) -> jax.Array:
    """Refine hidden entries by inverse-then-forward emission prediction."""

    def refine(
        _,
        values: jax.Array,
    ) -> jax.Array:
        latent = emissions.invert(
            values,
        )

        predicted = emissions.conditional_mean(
            latent,
        )

        return jnp.where(
            visible,
            observations,
            predicted,
        )

    return jax.lax.fori_loop(
        0,
        num_iters,
        refine,
        completed,
    )


def _complete_and_invert(
    emissions: _Emissions,
    observations: jax.Array,
    visible: jax.Array,
    *,
    num_refinement_iters: int,
) -> jax.Array:
    """Complete hidden observations, refine them, then invert."""
    baseline = _baseline_observations(
        emissions,
        observations,
    )

    completed = _interpolate_observations(
        observations,
        visible,
        baseline,
    )

    completed = _refine_completed_observations(
        emissions,
        observations,
        visible,
        completed,
        num_iters=num_refinement_iters,
    )

    return emissions.invert(
        completed,
    )


def initial_latent_trajectory(
    emissions: GaussianEmissions | PoissonEmissions,
    observations: jax.Array,
    visible: jax.Array | None = None,
    *,
    num_refinement_iters: int = 3,
) -> jax.Array:
    """Construct an emission-based initial latent trajectory.

    `observations` has shape `(*B, T, D_y)` with batches aligned to the
    emissions. `visible`, when provided, is boolean and broadcastable to that
    shape; whole-timestep masks should use a trailing singleton channel axis.

    Fully visible observations are inverted directly. Otherwise hidden values
    are completed independently per sequence and observation channel before
    inversion. Interior gaps are interpolated, edge gaps are extrapolated from
    the nearest visible sample, and entirely hidden channels start from the
    emission mean at zero latent.

    Optional inverse/prediction refinements update hidden entries only.

    Completed observations are temporary initialization state and must never
    replace the observations used by inference or parameter fitting.
    """
    visible = _validate_inputs(
        emissions,
        observations,
        visible,
        num_refinement_iters,
    )

    if visible is None:
        return emissions.invert(
            observations,
        )

    return jax.lax.cond(
        jnp.all(visible),
        lambda: emissions.invert(
            observations,
        ),
        lambda: _complete_and_invert(
            emissions,
            observations,
            visible,
            num_refinement_iters=num_refinement_iters,
        ),
    )
