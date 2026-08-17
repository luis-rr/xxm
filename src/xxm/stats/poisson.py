import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from ..newton import NewtonSearch


class LinearPoissonFit(typing.NamedTuple):
    """Parameters of a fitted linear Poisson model."""

    coefficients: jax.Array
    bias: jax.Array


class _PredictorMoments(typing.NamedTuple):
    """Mean and variance of Gaussian linear predictors."""

    mean: jax.Array
    variance: jax.Array | float


def _linear_predictor_moments(
    means: jax.Array,
    covariances: jax.Array | None,
    coefficients: jax.Array,
    bias: jax.Array,
) -> _PredictorMoments:
    """Compute moments of linear predictors under Gaussian inputs."""
    mean = means @ coefficients.T + bias

    if covariances is None:
        variance = 0.0
    else:
        variance = jnp.einsum(
            'ni,tij,nj->tn',
            coefficients,
            covariances,
            coefficients,
        )

    return _PredictorMoments(mean=mean, variance=variance)


def _expected_poisson_log_prob(
    observations: jax.Array,
    mean_log_rates: jax.Array,
    variance_log_rates: jax.Array | float = 0.0,
) -> jax.Array:
    """Expected elementwise Poisson log probabilities."""
    return (
        observations * mean_log_rates
        - jnp.exp(mean_log_rates + 0.5 * variance_log_rates)
        - jsp.special.gammaln(observations + 1)
    )


def log_likelihoods(
    observations: jax.Array,  # (T, N)
    log_rates: jax.Array,  # (1 or T, K, N)
) -> jax.Array:  # (T, K)
    """Poisson log likelihood for each time and state."""
    return jnp.sum(
        _expected_poisson_log_prob(
            observations=observations[:, None, :],
            mean_log_rates=log_rates,
        ),
        axis=-1,
    )


def expected_log_likelihood_per_output(
    observations: jax.Array,
    means: jax.Array,
    covariances: jax.Array | None,
    coefficients: jax.Array,
    bias: jax.Array,
    sample_weights: jax.Array | None = None,
) -> jax.Array:
    """Expected Poisson log likelihood for each output dimension."""
    predictor = _linear_predictor_moments(
        means=means,
        covariances=covariances,
        coefficients=coefficients,
        bias=bias,
    )

    log_probs = _expected_poisson_log_prob(
        observations,
        predictor.mean,
        predictor.variance,
    )

    if sample_weights is not None:
        log_probs = sample_weights[:, None] * log_probs

    return jnp.sum(log_probs, axis=0)


def expected_log_likelihood(
    observations: jax.Array,
    means: jax.Array,
    covariances: jax.Array | None,
    coefficients: jax.Array,
    bias: jax.Array,
    sample_weights: jax.Array | None = None,
) -> jax.Array:
    """Expected Poisson log likelihood under Gaussian input marginals."""
    return jnp.sum(
        expected_log_likelihood_per_output(
            observations=observations,
            means=means,
            covariances=covariances,
            coefficients=coefficients,
            bias=bias,
            sample_weights=sample_weights,
        )
    )


def fit_weighted(
    observations: jax.Array,  # (T, N)
    weights: jax.Array,  # (T, K)
    eps: float = 1e-8,
) -> jax.Array:
    """Fit weighted state-specific Poisson log rates."""
    counts = jnp.maximum(weights.sum(axis=0), eps)  # (K,)
    rates = weights.T @ observations / counts[:, None]  # (K, N)
    return jnp.log(jnp.maximum(rates, eps))


class _NewtonSearchParams(typing.NamedTuple):
    """Poisson regression parameters, with one independent block per output."""

    coefficients: jax.Array
    bias: jax.Array

    def take_step(
        self,
        direction: typing.Self,
        step_size: jax.Array,
    ) -> typing.Self:
        """Take a possibly different step for each output."""
        return self.__class__(
            coefficients=self.coefficients + step_size[..., None] * direction.coefficients,
            bias=self.bias + step_size * direction.bias,
        )

    def norm(self) -> jax.Array:
        """Return the parameter norm for each output."""
        return jnp.sqrt(jnp.sum(self.coefficients**2, axis=-1) + self.bias**2)

    def relative_change_from(self, other: typing.Self) -> jax.Array:
        """Return the relative parameter change from ``other`` for each output."""
        distance = self.__class__(
            coefficients=self.coefficients - other.coefficients,
            bias=self.bias - other.bias,
        ).norm()

        return distance / (1.0 + other.norm())

    def where(
        self,
        mask: jax.Array,
        other: typing.Self,
    ) -> typing.Self:
        """Select one parameter block or the other independently per output."""
        return self.__class__(
            coefficients=jnp.where(mask[..., None], self.coefficients, other.coefficients),
            bias=jnp.where(mask, self.bias, other.bias),
        )


class _NewtonSearchModel(typing.NamedTuple):
    """Quantities held fixed while fitting a Poisson linear model."""

    observations: jax.Array
    means: jax.Array
    covariances: jax.Array | None
    sample_weights: jax.Array  # (T,)
    ridge: float

    def objective(self, params: _NewtonSearchParams) -> jax.Array:
        """Expected Poisson log likelihood for each output."""
        log_likelihood = expected_log_likelihood_per_output(
            observations=self.observations,
            means=self.means,
            covariances=self.covariances,
            coefficients=params.coefficients,
            bias=params.bias,
            sample_weights=self.sample_weights,
        )
        weight_sum = jnp.sum(self.sample_weights)
        penalty = 0.5 * self.ridge * weight_sum * jnp.sum(params.coefficients**2, axis=-1)
        return log_likelihood - penalty

    def newton_direction(self, params: _NewtonSearchParams) -> _NewtonSearchParams:
        """Compute the Newton direction independently for all outputs."""
        predictor = _linear_predictor_moments(
            means=self.means,
            covariances=self.covariances,
            coefficients=params.coefficients,
            bias=params.bias,
        )
        expected_rates = jnp.exp(predictor.mean + 0.5 * predictor.variance)

        weighted_observations = self.sample_weights[:, None] * self.observations
        weighted_rates = self.sample_weights[:, None] * expected_rates

        gradient_bias = jnp.sum(weighted_observations - weighted_rates, axis=0)

        if self.covariances is None:
            residual_weights = weighted_observations - weighted_rates
            gradient_coefficients = residual_weights.T @ self.means

            precision_coefficients = jnp.einsum(
                'tn,ti,tj->nij',
                weighted_rates,
                self.means,
                self.means,
            )
            precision_cross = weighted_rates.T @ self.means
        else:
            shifted_means = self.means[:, None, :] + jnp.einsum(
                'tij,nj->tni',
                self.covariances,
                params.coefficients,
            )

            gradient_coefficients = weighted_observations.T @ self.means - jnp.einsum(
                'tn,tni->ni',
                weighted_rates,
                shifted_means,
            )

            precision_coefficients = jnp.einsum(
                'tn,tni,tnj->nij',
                weighted_rates,
                shifted_means,
                shifted_means,
            ) + jnp.einsum(
                'tn,tij->nij',
                weighted_rates,
                self.covariances,
            )

            precision_cross = jnp.einsum(
                'tn,tni->ni',
                weighted_rates,
                shifted_means,
            )
        weight_sum = jnp.sum(self.sample_weights)
        ridge_precision = self.ridge * weight_sum
        coefficient_dim = params.coefficients.shape[-1]

        gradient_coefficients = gradient_coefficients - ridge_precision * params.coefficients
        precision_coefficients = (
            precision_coefficients
            + ridge_precision * jnp.eye(coefficient_dim, dtype=precision_coefficients.dtype)[None]
        )

        precision_bias = jnp.sum(weighted_rates, axis=0)

        top = jnp.concatenate(
            [precision_coefficients, precision_cross[..., None]],
            axis=-1,
        )
        bottom = jnp.concatenate(
            [precision_cross, precision_bias[:, None]],
            axis=-1,
        )[:, None, :]
        precision = jnp.concatenate([top, bottom], axis=-2)

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
            [gradient_coefficients, gradient_bias[:, None]],
            axis=-1,
        )

        direction = jnp.linalg.solve(
            precision,
            gradient[..., None],
        )[..., 0]

        return _NewtonSearchParams(
            coefficients=direction[..., :-1],
            bias=direction[..., -1],
        )


def fit_linear_from_marginals(
    observations: jax.Array,
    means: jax.Array,
    covariances: jax.Array | None,
    coefficients: jax.Array,
    bias: jax.Array,
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
    sample_weights: jax.Array | None = None,
    ridge: float = 0.0,
) -> LinearPoissonFit:
    """Fit a Poisson linear model from Gaussian input marginals.

    Predictors are centered internally for numerical conditioning.
    ``ridge`` penalizes coefficients relative to the average weighted
    log likelihood; the bias is not penalized.
    """
    if sample_weights is None:
        sample_weights = jnp.ones(
            observations.shape[0],
            dtype=observations.dtype,
        )

    weight_sum = jnp.sum(sample_weights)

    center = jnp.sum(sample_weights[:, None] * means, axis=0) / jnp.maximum(weight_sum, 1e-8)
    centered_means = means - center

    # b + C x = (b + C center) + C (x - center)
    centered_bias = bias + coefficients @ center

    model = _NewtonSearchModel(
        observations=observations,
        means=centered_means,
        covariances=covariances,
        sample_weights=sample_weights,
        ridge=ridge,
    )

    initial_params = _NewtonSearchParams(
        coefficients=coefficients,
        bias=centered_bias,
    )

    search = NewtonSearch[_NewtonSearchParams](
        model=model,
        max_line_search_iters=max_line_search_iters,
        tol=tol,
    )

    final = search.optimize(
        params=initial_params,
        max_iter=max_iter,
    )

    # Convert the intercept back to the original predictor coordinates.
    bias = final.params.bias - final.params.coefficients @ center

    return LinearPoissonFit(
        coefficients=final.params.coefficients,
        bias=bias,
    )


def fit_linear(
    inputs: jax.Array,  # (T, P)
    outputs: jax.Array,  # (T, N)
    coefficients: jax.Array,  # (N, P)
    bias: jax.Array,  # (N,)
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
    ridge: float = 0.0,
) -> LinearPoissonFit:
    """Fit a Poisson linear model from paired samples."""
    dtype = jnp.result_type(inputs, coefficients, bias, jnp.float32)

    return fit_linear_from_marginals(
        observations=outputs,
        means=inputs.astype(dtype),
        covariances=None,
        coefficients=coefficients.astype(dtype),
        bias=bias.astype(dtype),
        max_iter=max_iter,
        tol=tol,
        max_line_search_iters=max_line_search_iters,
        ridge=ridge,
    )


def fit_weighted_linear(
    inputs: jax.Array,  # (T, P)
    outputs: jax.Array,  # (T, N)
    weights: jax.Array,  # (T, K)
    coefficients: jax.Array,  # (K, N, P)
    bias: jax.Array,  # (K, N)
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
    ridge: float = 0.0,
) -> LinearPoissonFit:
    """Fit state-specific weighted Poisson linear models."""
    dtype = jnp.result_type(inputs, coefficients, bias, jnp.float32)

    inputs = inputs.astype(dtype)
    coefficients = coefficients.astype(dtype)
    bias = bias.astype(dtype)

    def fit_state(
        state_weights,
        state_coefficients,
        state_bias,
    ):
        return fit_linear_from_marginals(
            observations=outputs,
            means=inputs,
            covariances=None,
            coefficients=state_coefficients,
            bias=state_bias,
            sample_weights=state_weights,
            max_iter=max_iter,
            tol=tol,
            max_line_search_iters=max_line_search_iters,
            ridge=ridge,
        )

    return jax.vmap(
        fit_state,
        in_axes=(1, 0, 0),
    )(
        weights,
        coefficients,
        bias,
    )
