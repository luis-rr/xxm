import jax
from jax import numpy as jnp


def expected_log_likelihood(
    observations: jax.Array,
    means: jax.Array,
    covariances: jax.Array,
    readout: jax.Array,
    bias: jax.Array,
) -> jax.Array:
    """Expected Poisson log likelihood, up to parameter-independent constants."""
    linear_predictors = means @ readout.T + bias

    variance_correction = 0.5 * jnp.einsum(
        'ni,tij,nj->tn',
        readout,
        covariances,
        readout,
    )

    expected_rates = jnp.exp(linear_predictors + variance_correction)

    return jnp.sum(observations * linear_predictors - expected_rates)


def newton_direction(
    observations: jax.Array,
    means: jax.Array,
    covariances: jax.Array,
    readout: jax.Array,
    bias: jax.Array,
) -> jax.Array:
    """Compute the Newton direction for all neurons in parallel."""
    expected_rates = jnp.exp(
        means @ readout.T
        + bias
        + 0.5
        * jnp.einsum(
            'ni,tij,nj->tn',
            readout,
            covariances,
            readout,
        )
    )

    # Derivative of
    #
    #   c.T m_t + 1/2 c.T V_t c
    #
    # with respect to c.
    shifted_means = means[:, None, :] + jnp.einsum(
        'tij,nj->tni',
        covariances,
        readout,
    )

    gradient_readout = observations.T @ means - jnp.einsum(
        'tn,tni->ni',
        expected_rates,
        shifted_means,
    )

    gradient_bias = jnp.sum(
        observations - expected_rates,
        axis=0,
    )

    # Negative Hessian (= positive Newton precision).
    precision_readout = jnp.einsum(
        'tn,tni,tnj->nij',
        expected_rates,
        shifted_means,
        shifted_means,
    ) + jnp.einsum(
        'tn,tij->nij',
        expected_rates,
        covariances,
    )

    precision_cross = jnp.einsum(
        'tn,tni->ni',
        expected_rates,
        shifted_means,
    )

    precision_bias = jnp.sum(
        expected_rates,
        axis=0,
    )

    # Assemble one (D+1)x(D+1) system per neuron.
    top = jnp.concatenate(
        [
            precision_readout,
            precision_cross[..., None],
        ],
        axis=-1,
    )

    bottom = jnp.concatenate(
        [
            precision_cross,
            precision_bias[:, None],
        ],
        axis=-1,
    )[:, None, :]

    precision = jnp.concatenate(
        [top, bottom],
        axis=-2,
    )

    # Tiny numerical jitter for nearly singular regressions.
    parameter_dim = precision.shape[-1]
    precision = (
        precision
        + 1e-8
        * jnp.eye(
            parameter_dim,
            dtype=precision.dtype,
        )[None]
    )

    gradient = jnp.concatenate(
        [
            gradient_readout,
            gradient_bias[:, None],
        ],
        axis=-1,
    )

    return jnp.linalg.solve(
        precision,
        gradient[..., None],
    )[..., 0]


def fit_from_moments(
    observations: jax.Array,
    means: jax.Array,
    covariances: jax.Array,
    readout: jax.Array,
    bias: jax.Array,
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
) -> tuple[jax.Array, jax.Array]:
    """Fit Poisson readout parameters by damped Newton optimization."""
    latent_dim = readout.shape[1]

    params = jnp.concatenate(
        [readout, bias[:, None]],
        axis=1,
    )

    def unpack(
        params: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return params[:, :latent_dim], params[:, latent_dim]

    def objective(params: jax.Array) -> jax.Array:
        readout, bias = unpack(params)

        return expected_log_likelihood(
            observations,
            means,
            covariances,
            readout,
            bias,
        )

    initial_objective = objective(params)

    def should_continue(carry) -> jax.Array:
        iteration, _, _, done = carry
        return (iteration < max_iter) & ~done

    def step(carry):
        iteration, params, current_objective, _ = carry

        readout, bias = unpack(params)

        direction = newton_direction(
            observations,
            means,
            covariances,
            readout,
            bias,
        )

        step_size = jnp.asarray(
            1.0,
            dtype=params.dtype,
        )

        candidate = params + direction
        candidate_objective = objective(candidate)

        def needs_backtracking(search) -> jax.Array:
            (
                line_iteration,
                _,
                _,
                candidate_objective,
            ) = search

            accepted = jnp.isfinite(candidate_objective) & (
                candidate_objective >= current_objective
            )

            return ~accepted & (line_iteration < max_line_search_iters)

        def backtrack(search):
            line_iteration, step_size, _, _ = search

            step_size = 0.5 * step_size
            candidate = params + step_size * direction

            return (
                line_iteration + 1,
                step_size,
                candidate,
                objective(candidate),
            )

        _, _, candidate, candidate_objective = jax.lax.while_loop(
            needs_backtracking,
            backtrack,
            (
                jnp.asarray(0),
                step_size,
                candidate,
                candidate_objective,
            ),
        )

        accepted = jnp.isfinite(candidate_objective) & (candidate_objective >= current_objective)

        next_params = jnp.where(
            accepted,
            candidate,
            params,
        )

        next_objective = jnp.where(
            accepted,
            candidate_objective,
            current_objective,
        )

        relative_change = jnp.linalg.norm(next_params - params) / (1.0 + jnp.linalg.norm(params))

        done = (relative_change <= tol) | ~accepted

        return (
            iteration + 1,
            next_params,
            next_objective,
            done,
        )

    _, params, _, _ = jax.lax.while_loop(
        should_continue,
        step,
        (
            jnp.asarray(0),
            params,
            initial_objective,
            jnp.asarray(False),
        ),
    )

    return unpack(params)
