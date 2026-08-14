import typing

import jax
from jax import numpy as jnp

from ..newton import NewtonSearch


def log_likelihood_batched(mean, covariances, observations, readout, bias) -> jax.Array:
    """Expected Poisson log likelihood for each neuron."""

    linear_predictors = mean @ readout.T + bias

    variance_correction = 0.5 * jnp.einsum(
        'ni,tij,nj->tn',
        readout,
        covariances,
        readout,
    )

    expected_rates = jnp.exp(linear_predictors + variance_correction)

    return jnp.sum(
        observations * linear_predictors - expected_rates,
        axis=0,
    )


def log_likelihood(
    observations: jax.Array,
    means: jax.Array,
    covariances: jax.Array,
    readout: jax.Array,
    bias: jax.Array,
) -> jax.Array:
    """Expected Poisson log likelihood of linear model with exponential link function."""
    return jnp.sum(
        log_likelihood_batched(
            mean=means,
            covariances=covariances,
            observations=observations,
            readout=readout,
            bias=bias,
        )
    )


class _NewtonSearchParams(typing.NamedTuple):
    """Poisson readout parameters, with one independent block per neuron."""

    readout: jax.Array
    bias: jax.Array

    def take_step(
        self,
        direction: typing.Self,
        step_size: jax.Array,
    ) -> typing.Self:
        """Take a possibly different step for each neuron."""
        return self.__class__(
            readout=self.readout + step_size[..., None] * direction.readout,
            bias=self.bias + step_size * direction.bias,
        )

    def norm(self) -> jax.Array:
        """Return the parameter norm for each neuron."""
        return jnp.sqrt(jnp.sum(self.readout**2, axis=-1) + self.bias**2)

    def relative_change_from(self, other: typing.Self) -> jax.Array:
        """Return the relative parameter change from ``other`` for each neuron."""
        distance = self.__class__(
            readout=self.readout - other.readout,
            bias=self.bias - other.bias,
        ).norm()

        return distance / (1.0 + other.norm())

    def where(
        self,
        mask: jax.Array,
        other: typing.Self,
    ) -> typing.Self:
        """Select one parameter block or the other independently per neuron."""
        return self.__class__(
            readout=jnp.where(mask[..., None], self.readout, other.readout),
            bias=jnp.where(mask, self.bias, other.bias),
        )


class _NewtonSearchModel(typing.NamedTuple):
    """Quantities held fixed while fitting the Poisson readout."""

    observations: jax.Array
    means: jax.Array
    covariances: jax.Array

    def objective(self, params: _NewtonSearchParams) -> jax.Array:
        """Expected Poisson log likelihood for each neuron."""

        return log_likelihood_batched(
            mean=self.means,
            covariances=self.covariances,
            observations=self.observations,
            readout=params.readout,
            bias=params.bias,
        )

    def newton_direction(self, params: _NewtonSearchParams) -> _NewtonSearchParams:
        """Compute the Newton direction independently for all neurons."""

        expected_rates = jnp.exp(
            self.means @ params.readout.T
            + params.bias
            + 0.5
            * jnp.einsum(
                'ni,tij,nj->tn',
                params.readout,
                self.covariances,
                params.readout,
            )
        )

        # Derivative of
        #
        #   c.T m_t + 1/2 c.T V_t c
        #
        # with respect to c.
        shifted_means = self.means[:, None, :] + jnp.einsum(
            'tij,nj->tni', self.covariances, params.readout
        )

        gradient_readout = self.observations.T @ self.means - jnp.einsum(
            'tn,tni->ni',
            expected_rates,
            shifted_means,
        )

        gradient_bias = jnp.sum(
            self.observations - expected_rates,
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
            self.covariances,
        )

        precision_cross = jnp.einsum('tn,tni->ni', expected_rates, shifted_means)

        precision_bias = jnp.sum(expected_rates, axis=0)

        # Assemble one (D+1)x(D+1) Newton system per neuron.
        top = jnp.concatenate([precision_readout, precision_cross[..., None]], axis=-1)
        bottom = jnp.concatenate([precision_cross, precision_bias[:, None]], axis=-1)[:, None, :]
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
            [
                gradient_readout,
                gradient_bias[:, None],
            ],
            axis=-1,
        )

        direction = jnp.linalg.solve(
            precision,
            gradient[..., None],
        )[..., 0]

        return _NewtonSearchParams(
            readout=direction[..., :-1],
            bias=direction[..., -1],
        )


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

    model = _NewtonSearchModel(
        observations=observations,
        means=means,
        covariances=covariances,
    )

    initial_params = _NewtonSearchParams(
        readout=readout,
        bias=bias,
    )

    search = NewtonSearch[_NewtonSearchParams](
        model=model,
        max_line_search_iters=max_line_search_iters,
        tol=tol,
    )

    final = search.optimize(params=initial_params, max_iter=max_iter)

    return final.params.readout, final.params.bias
