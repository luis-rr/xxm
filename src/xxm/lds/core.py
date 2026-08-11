r"""Linear Dynamical Systems."""

from __future__ import annotations

import typing

import jax
from jax import numpy as jnp
from jax.scipy import linalg as jsp_linalg

from ..gaussian_chain import GaussianChain, GaussianPairPotential, GaussianPotential
from ..gaussian_chain import GaussianChainMarginals as Posterior


class GaussianEmissions(typing.NamedTuple):
    readout: jax.Array  # C, shape (N, D)
    bias: jax.Array  # d, shape (N,)
    noise_covariance: jax.Array  # R, shape (N, N)

    def get_potential(
        self,
        observations: jax.Array,
    ) -> GaussianPotential:
        # observations: (T, N)
        t = observations.shape[0]
        n = observations.shape[1]

        cholesky = jnp.linalg.cholesky(self.noise_covariance)

        precision = jsp_linalg.cho_solve(
            (cholesky, True),
            jnp.eye(n, dtype=self.noise_covariance.dtype),
        )

        centered_observations = observations - self.bias

        precision_matrix = precision @ self.readout

        precision_blocks = jnp.broadcast_to(
            self.readout.T @ precision_matrix,
            (t, self.readout.shape[1], self.readout.shape[1]),
        )

        information_vectors = centered_observations @ precision_matrix

        log_det_covariance = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))

        quadratic_terms = jnp.einsum(
            'ti,ij,tj->t',
            centered_observations,
            precision,
            centered_observations,
        )

        log_constant = jnp.sum(
            -0.5 * quadratic_terms - 0.5 * log_det_covariance - 0.5 * n * jnp.log(2.0 * jnp.pi)
        )

        return GaussianPotential(
            precision_blocks=precision_blocks,
            information_vectors=information_vectors,
            log_constant=log_constant,
        )

    def sample(
        self,
        key: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        """Sample an observation conditional on a state."""
        mean = self.readout @ state + self.bias

        return jax.random.multivariate_normal(
            key,
            mean=mean,
            cov=self.noise_covariance,
        )

    # @classmethod
    # def initialize(
    #     cls,
    #     observations: jax.Array,
    #     num_states: int,
    #     key: jax.Array,
    # ) -> typing.Self: ...

    # def m_step(
    #     self,
    #     observations: jax.Array,
    #     posterior: Posterior,
    # ) -> typing.Self: ...


class Model(typing.NamedTuple):
    r"""
    Container for LDS model parameters.
    """

    initial_mean: jax.Array
    initial_covariance: jax.Array

    dynamics_matrix: jax.Array
    dynamics_bias: jax.Array
    dynamics_noise_covariance: jax.Array

    emissions: GaussianEmissions

    # @classmethod
    # def initialize(
    #     cls,
    # ) -> Model: ...  # TODO

    def _to_chain(self, observations: jax.Array) -> GaussianChain:
        t = observations.shape[0]

        observation_potential = self.emissions.get_potential(observations)

        initial_potential = GaussianPotential.from_moments(
            self.initial_mean, self.initial_covariance
        )

        dynamics_potential = GaussianPairPotential.from_linear_conditional(
            self.dynamics_matrix, self.dynamics_bias, self.dynamics_noise_covariance
        )

        diagonal = observation_potential.precision_blocks
        diagonal = diagonal.at[0].add(initial_potential.precision_blocks)
        diagonal = diagonal.at[:-1].add(dynamics_potential.left_precision)
        diagonal = diagonal.at[1:].add(dynamics_potential.right_precision)

        information_vectors = observation_potential.information_vectors
        information_vectors = information_vectors.at[0].add(initial_potential.information_vectors)
        information_vectors = information_vectors.at[:-1].add(dynamics_potential.left_information)
        information_vectors = information_vectors.at[1:].add(dynamics_potential.right_information)

        lower_precision_blocks = jnp.broadcast_to(
            dynamics_potential.lower_precision,
            (t - 1,) + dynamics_potential.lower_precision.shape,
        )

        log_constant = (
            observation_potential.log_constant
            + initial_potential.log_constant
            + (t - 1) * dynamics_potential.log_constant
        )

        return GaussianChain(
            diagonal_precision_blocks=diagonal,
            lower_precision_blocks=lower_precision_blocks,
            information_vectors=information_vectors,
            log_constant=log_constant,
        )

    def inference(self, observations: jax.Array) -> Posterior:
        chain = self._to_chain(observations)
        return chain.forward_backward()

    def sample(
        self,
        num_steps: int,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample states and observations from the LDS."""
        key, initial_state_key, initial_observation_key = jax.random.split(key, 3)

        initial_state = jax.random.multivariate_normal(
            initial_state_key,
            mean=self.initial_mean,
            cov=self.initial_covariance,
        )

        initial_observation = self.emissions.sample(
            initial_observation_key,
            initial_state,
        )

        def sample_step(
            carry: tuple[jax.Array, jax.Array],
            _: None,
        ) -> tuple[
            tuple[jax.Array, jax.Array],
            tuple[jax.Array, jax.Array],
        ]:
            previous_state, key = carry

            key, state_key, observation_key = jax.random.split(key, 3)

            state_mean = self.dynamics_matrix @ previous_state + self.dynamics_bias

            state = jax.random.multivariate_normal(
                state_key,
                mean=state_mean,
                cov=self.dynamics_noise_covariance,
            )

            observation = self.emissions.sample(
                observation_key,
                state,
            )

            return (
                (state, key),
                (state, observation),
            )

        _, (remaining_states, remaining_observations) = jax.lax.scan(
            sample_step,
            (initial_state, key),
            None,
            length=num_steps - 1,
        )

        states = jnp.concatenate(
            [initial_state[None], remaining_states],
            axis=0,
        )
        observations = jnp.concatenate(
            [initial_observation[None], remaining_observations],
            axis=0,
        )

        return states, observations
