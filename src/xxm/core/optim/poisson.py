import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.optim.newton import NewtonSearch

EPS: float = 1e-8


class _NewtonSearchParams(typing.NamedTuple):
    """Poisson regression parameters, with one independent block per output."""

    affine: Affine

    def take_step(
        self,
        direction: typing.Self,
        step_size: jax.Array,
    ) -> typing.Self:
        return self.__class__(
            affine=self.affine._replace(
                coefficients=self.affine.coefficients
                + step_size[..., None] * direction.affine.coefficients,
                bias=self.affine.bias + step_size * direction.affine.bias,
            )
        )

    def relative_change_from(self, other: typing.Self) -> jax.Array:

        change = self.affine._replace(
            coefficients=self.affine.coefficients - other.affine.coefficients,
            bias=self.affine.bias - other.affine.bias,
        )

        distance = change.norm()

        return distance / (1.0 + other.affine.norm())

    def where(
        self,
        mask: jax.Array,
        other: typing.Self,
    ) -> typing.Self:
        return self.__class__(
            affine=self.affine._replace(
                coefficients=jnp.where(
                    mask[..., None],
                    self.affine.coefficients,
                    other.affine.coefficients,
                ),
                bias=jnp.where(mask, self.affine.bias, other.affine.bias),
            )
        )

    def to_linear_model(self) -> LinearPoisson:
        return LinearPoisson(self.affine)


class _NewtonSearchModel(typing.NamedTuple):
    """Quantities held fixed while fitting a Poisson linear model."""

    values: jax.Array  # (T, O)
    input_means: jax.Array  # (T, I)
    input_covariances: jax.Array | None  # (T, I, I)
    weights: jax.Array  # (T,)
    ridge: float

    def _expected_rates(self, params: _NewtonSearchParams) -> jax.Array:
        linear_model = params.to_linear_model()

        if self.input_covariances is None:
            return linear_model.conditional(self.input_means).rates

        return linear_model.expected_rates(
            Gaussian(
                mean=self.input_means,
                covariance=self.input_covariances,
            )
        )

    def objective(self, params: _NewtonSearchParams) -> jax.Array:
        """Expected Poisson log likelihood for each output."""
        linear_model = params.to_linear_model()

        if self.input_covariances is None:
            log_probs = linear_model.conditional(self.input_means).log_prob_each(self.values)

        else:
            log_probs = linear_model.expected_log_prob_each(
                values=self.values,
                inputs=Gaussian(
                    mean=self.input_means,
                    covariance=self.input_covariances,
                ),
            )

        log_likelihood = jnp.sum(
            self.weights[:, None] * log_probs,
            axis=0,
        )

        weight_sum = jnp.sum(self.weights)
        penalty = 0.5 * self.ridge * weight_sum * jnp.sum(params.affine.coefficients**2, axis=-1)

        return log_likelihood - penalty

    def newton_direction(
        self,
        params: _NewtonSearchParams,
    ) -> _NewtonSearchParams:
        """Compute the Newton direction independently for all outputs."""
        expected_rates = self._expected_rates(params)

        weighted_observations = self.weights[:, None] * self.values
        weighted_rates = self.weights[:, None] * expected_rates

        gradient_bias = jnp.sum(
            weighted_observations - weighted_rates,
            axis=0,
        )

        if self.input_covariances is None:
            residual_weights = weighted_observations - weighted_rates
            gradient_coefficients = residual_weights.T @ self.input_means

            precision_coefficients = jnp.einsum(
                'to,ti,tj->oij',
                weighted_rates,
                self.input_means,
                self.input_means,
            )
            precision_cross = weighted_rates.T @ self.input_means

        else:
            shifted_means = self.input_means[:, None, :] + jnp.einsum(
                'tij,oj->toi',
                self.input_covariances,
                params.affine.coefficients,
            )

            gradient_coefficients = weighted_observations.T @ self.input_means - jnp.einsum(
                'to,toi->oi',
                weighted_rates,
                shifted_means,
            )

            precision_coefficients = jnp.einsum(
                'to,toi,toj->oij',
                weighted_rates,
                shifted_means,
                shifted_means,
            ) + jnp.einsum(
                'to,tij->oij',
                weighted_rates,
                self.input_covariances,
            )

            precision_cross = jnp.einsum(
                'to,toi->oi',
                weighted_rates,
                shifted_means,
            )

        weight_sum = jnp.sum(self.weights)
        ridge_precision = self.ridge * weight_sum
        input_dim = params.affine.coefficients.shape[-1]

        gradient_coefficients = gradient_coefficients - ridge_precision * params.affine.coefficients
        precision_coefficients = (
            precision_coefficients
            + ridge_precision
            * jnp.eye(
                input_dim,
                dtype=precision_coefficients.dtype,
            )[None]
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
            affine=params.affine._replace(
                coefficients=direction[..., :-1],
                bias=direction[..., -1],
            )
        )


def from_samples(
    values: jax.Array,  # (T, N)
) -> Poisson:
    rates = jnp.mean(values, axis=0)
    return Poisson(log_rates=jnp.log(jnp.maximum(rates, EPS)))


def from_samples_weighted(
    values: jax.Array,  # (T, N)
    weights: jax.Array,  # (T, ...)
) -> Poisson:
    """Fit one weighted Poisson model per batch entry of ``weights``."""
    total = jnp.sum(weights, axis=0)
    counts = jnp.where(total > 0, total, EPS)
    rates = (
        jnp.einsum(
            't...,tn->...n',
            weights,
            values,
        )
        / counts[..., None]
    )

    return Poisson(
        log_rates=jnp.log(jnp.maximum(rates, EPS)),
    )


def from_samples_grouped(
    values: jax.Array,  # (T, N)
    assignments: jax.Array,  # (T,)
    num_groups: int,
) -> Poisson:  # K-batched
    """Fit one distribution to each group of assigned values."""
    weights = jax.nn.one_hot(
        assignments,
        num_groups,
        dtype=jnp.result_type(values, jnp.float32),
    )  # (T, K)

    return from_samples_weighted(
        values=values,
        weights=weights,
    )


def _initial_affine(
    inputs: jax.Array,  # (T, I)
    outputs: jax.Array,  # (T, O)
    weights: jax.Array | None = None,  # (T, ...)
) -> Affine:
    """Initialize with zero coefficients and empirical output log rates."""
    dtype = jnp.result_type(
        inputs,
        outputs,
        jnp.float32,
    )

    if weights is None:
        bias = from_samples(
            outputs,
        ).log_rates  # (O,)

        batch_shape = ()

    else:
        bias = from_samples_weighted(
            values=outputs,
            weights=weights,
        ).log_rates  # (..., O)

        batch_shape = weights.shape[1:]

    coefficients = jnp.zeros(
        batch_shape
        + (
            outputs.shape[-1],
            inputs.shape[-1],
        ),
        dtype=dtype,
    )  # (..., O, I)

    return Affine(
        coefficients=coefficients,
        bias=bias.astype(dtype),
    )


def _linear_from_moments(
    outputs: jax.Array,
    input_means: jax.Array,
    input_covariances: jax.Array | None,
    weights: jax.Array | None,
    initial_affine: Affine | None,
    max_iter: int,
    tol: float,
    max_line_search_iters: int,
    ridge: float,
) -> LinearPoisson:
    """
    Fit from Gaussian input moments.
    """
    dtype = jnp.result_type(
        outputs,
        input_means,
        jnp.float32,
    )

    outputs = outputs.astype(dtype)
    input_means = input_means.astype(dtype)

    if input_covariances is not None:
        input_covariances = input_covariances.astype(dtype)

    if weights is None:
        weights = jnp.ones(
            outputs.shape[0],
            dtype=dtype,
        )  # (T,)
    else:
        weights = weights.astype(dtype)

    if initial_affine is None:
        initial_affine = _initial_affine(
            inputs=input_means,
            outputs=outputs,
            weights=weights,
        )

    initial_affine = initial_affine.astype(dtype)

    weight_sum = jnp.sum(weights)

    center = jnp.sum(weights[:, None] * input_means, axis=0) / jnp.maximum(weight_sum, 1e-8)  # (I,)

    centered_means = input_means - center  # (T, I)

    # b + A x = (b + A center) + A (x - center)
    centered_affine = initial_affine.shift(center)

    model = _NewtonSearchModel(
        values=outputs,
        input_means=centered_means,
        input_covariances=input_covariances,
        weights=weights,
        ridge=ridge,
    )

    search = NewtonSearch[_NewtonSearchParams](
        model=model,
        max_line_search_iters=max_line_search_iters,
        tol=tol,
    )

    final = search.optimize(
        params=_NewtonSearchParams(centered_affine),
        max_iter=max_iter,
    )

    return LinearPoisson(
        affine=final.params.affine.shift(-center),
    )


def linear_from_marginals(
    inputs: Gaussian,  # T-batched
    outputs: jax.Array,  # (T, O)
    weights: jax.Array | None = None,  # (T,)
    initial_affine: Affine | None = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
    ridge: float = 0.0,
) -> LinearPoisson:
    """Fit from Gaussian input marginals."""
    return _linear_from_moments(
        outputs=outputs,
        input_means=inputs.mean,
        input_covariances=inputs.covariance,
        weights=weights,
        initial_affine=initial_affine,
        max_iter=max_iter,
        tol=tol,
        max_line_search_iters=max_line_search_iters,
        ridge=ridge,
    )


def linear_from_pairs(
    inputs: jax.Array,  # (T, I)
    outputs: jax.Array,  # (T, O)
    initial_affine: Affine | None = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
    ridge: float = 0.0,
) -> LinearPoisson:
    """Fit from paired input-output samples."""
    return _linear_from_moments(
        outputs=outputs,
        input_means=inputs,
        input_covariances=None,
        weights=None,
        initial_affine=initial_affine,
        max_iter=max_iter,
        tol=tol,
        max_line_search_iters=max_line_search_iters,
        ridge=ridge,
    )


def linear_from_pairs_weighted(
    inputs: jax.Array,  # (T, I)
    outputs: jax.Array,  # (T, O)
    weights: jax.Array,  # (T, K)
    initial_affine: Affine | None = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
    ridge: float = 0.0,
) -> LinearPoisson:
    """Fit one weighted model for each weight column."""

    def fit_state(
        state_weights: jax.Array,  # (T,)
        state_affine: Affine | None,
    ) -> LinearPoisson:
        return _linear_from_moments(
            outputs=outputs,
            input_means=inputs,
            input_covariances=None,
            weights=state_weights,
            initial_affine=state_affine,
            max_iter=max_iter,
            tol=tol,
            max_line_search_iters=max_line_search_iters,
            ridge=ridge,
        )

    if initial_affine is None:
        return jax.vmap(
            lambda state_weights: fit_state(state_weights, None),
            in_axes=1,
        )(weights)

    return jax.vmap(
        fit_state,
        in_axes=(1, 0),
    )(
        weights,
        initial_affine,
    )


def linear_from_pairs_grouped(
    inputs: jax.Array,  # (T, I)
    outputs: jax.Array,  # (T, O)
    assignments: jax.Array,  # (T,)
    num_groups: int,
    initial_affine: Affine | None = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
    ridge: float = 0.0,
) -> LinearPoisson:
    """Fit one model to each assigned group."""
    weights = jax.nn.one_hot(
        assignments,
        num_groups,
        dtype=jnp.result_type(inputs, outputs, jnp.float32),
    )  # (T, K)

    return linear_from_pairs_weighted(
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        initial_affine=initial_affine,
        max_iter=max_iter,
        tol=tol,
        max_line_search_iters=max_line_search_iters,
        ridge=ridge,
    )
