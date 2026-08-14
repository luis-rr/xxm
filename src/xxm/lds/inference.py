import typing

import jax
from jax import numpy as jnp

from ..gaussian_chain import GaussianChain
from ..gaussian_chain import GaussianChainMarginals as Posterior
from .core import Model
from .emissions import LaplaceEmissions, LaplaceEmissionsT, QuadraticEmissionsT


def inference_exact(model: Model[QuadraticEmissionsT], observations: jax.Array) -> Posterior:
    """Compute the exact posterior over latents for quadratic emissions."""

    latent_chain = model.to_gaussian_chain(num_time_steps=observations.shape[0])

    observation_potential = model.emissions.get_potential(observations)

    posterior_chain = latent_chain.add_local_potential(observation_potential)

    return posterior_chain.forward_backward()


def _laplace_objective(
    latent_chain: GaussianChain,
    emissions: LaplaceEmissions,
    observations: jax.Array,
    latents: jax.Array,
) -> jax.Array:

    return latent_chain.log_potential(latents) + emissions.log_likelihood(observations, latents)


class NewtonModeSearch(typing.NamedTuple, typing.Generic[LaplaceEmissionsT]):
    latent_chain: GaussianChain
    emissions: LaplaceEmissionsT
    observations: jax.Array
    max_line_search_iters: int
    tol: float

    class State(typing.NamedTuple):
        """
        State for a damped Newton iteration to find the posterior mode.
        """

        iteration: jax.Array
        latents: jax.Array
        objective: jax.Array
        done: jax.Array

    def initial_state(self, latents: jax.Array, current_objective: jax.Array) -> State:
        return NewtonModeSearch.State(
            iteration=jnp.asarray(0),
            latents=latents,
            objective=current_objective,
            done=jnp.asarray(False),
        )

    def _newton_candidate(self, latents: jax.Array) -> jax.Array:
        """Compute the full Newton proposal for the latent trajectory."""

        observation_potential = self.emissions.get_local_potential(self.observations, latents)
        local_posterior_chain = self.latent_chain.add_local_potential(observation_potential)

        # The mean is also the mode of this local Gaussian approximation.
        # TODO: Use a more efficient solver for the mode, rather than computing the full posterior.
        return local_posterior_chain.forward_backward().means

    def _newton_step(self, search_state: State) -> State:
        """Perform one damped Newton iteration."""

        newton_latents = self._newton_candidate(search_state.latents)

        direction = newton_latents - search_state.latents

        candidate_latents, candidate_objective, accepted = self._backtracking_line_search(
            latents=search_state.latents,
            direction=direction,
            current_objective=search_state.objective,
            max_iter=self.max_line_search_iters,
        )

        next_latents = jnp.where(accepted, candidate_latents, search_state.latents)
        next_objective = jnp.where(accepted, candidate_objective, search_state.objective)

        relative_change = jnp.linalg.norm(next_latents - search_state.latents) / (
            1.0 + jnp.linalg.norm(search_state.latents)
        )

        return NewtonModeSearch.State(
            iteration=search_state.iteration + 1,
            latents=next_latents,
            objective=next_objective,
            done=(relative_change <= self.tol) | ~accepted,
        )

    def iterate(self, initial: State, max_iter: int) -> State:
        """Iteratively refine the posterior mode using damped Newton iterations."""

        def should_continue(search_state: NewtonModeSearch.State) -> jax.Array:
            return (search_state.iteration < max_iter) & ~search_state.done

        result = jax.lax.while_loop(
            should_continue,
            self._newton_step,
            initial,
        )

        return result

    def _backtracking_line_search(
        self,
        latents: jax.Array,
        direction: jax.Array,
        current_objective: jax.Array,
        max_iter: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Find an improving step along a Newton direction."""

        search = LineSearch(
            latent_chain=self.latent_chain,
            emissions=self.emissions,
            observations=self.observations,
            current_objective=current_objective,
            direction=direction,
            initial_latents=latents,
        )

        initial = search.initial_state()

        final = search.iterate(initial, max_iter=max_iter)

        return (
            final.candidate_latents,
            final.candidate_objective,
            final.is_accepted(current_objective),
        )


class LineSearch(typing.NamedTuple, typing.Generic[LaplaceEmissionsT]):
    latent_chain: GaussianChain
    emissions: LaplaceEmissionsT
    observations: jax.Array
    current_objective: jax.Array
    direction: jax.Array
    initial_latents: jax.Array

    class State(typing.NamedTuple):
        """
        State for a backtracking line search along a Newton direction.
        """

        iteration: jax.Array
        step_size: jax.Array
        candidate_latents: jax.Array
        candidate_objective: jax.Array

        def is_accepted(self, current_objective: jax.Array) -> jax.Array:
            return jnp.isfinite(self.candidate_objective) & (
                self.candidate_objective >= current_objective
            )

    def initial_state(self) -> State:
        candidate_latents = self.initial_latents + self.direction

        return LineSearch.State(
            iteration=jnp.asarray(0),
            step_size=jnp.asarray(1.0, dtype=self.initial_latents.dtype),
            candidate_latents=candidate_latents,
            candidate_objective=self.evaluate_candidate(candidate_latents),
        )

    def evaluate_candidate(self, candidate_latents: jax.Array) -> jax.Array:
        """Evaluate the log joint at a candidate latent trajectory."""

        return _laplace_objective(
            self.latent_chain,
            self.emissions,
            self.observations,
            candidate_latents,
        )

    def iterate(self, initial: State, max_iter: int) -> State:

        def needs_backtracking(search: LineSearch.State) -> jax.Array:
            return ~search.is_accepted(self.current_objective) & (search.iteration < max_iter)

        def backtrack(search: LineSearch.State) -> LineSearch.State:
            step_size = 0.5 * search.step_size
            candidate_latents = self.initial_latents + step_size * self.direction
            candidate_objective = self.evaluate_candidate(candidate_latents)

            return LineSearch.State(
                iteration=search.iteration + 1,
                step_size=step_size,
                candidate_latents=candidate_latents,
                candidate_objective=candidate_objective,
            )

        return jax.lax.while_loop(
            needs_backtracking,
            backtrack,
            initial,
        )


def inference_laplace(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    initial_latents: jax.Array | None = None,
    max_iter: int = 20,
    tol: float = 1e-6,
    max_line_search_iters: int = 20,
) -> Posterior:
    """
    Approximate the posterior over latents using Laplace inference.

    The posterior mode is found with damped Newton iterations. At each
    iteration, the emission likelihood is replaced by its local quadratic
    approximation, producing a Gaussian chain whose mean is the Newton
    candidate.

    The final Gaussian chain, evaluated at the converged mode, is returned
    as the Laplace approximation to p(x | y).
    """
    if max_iter < 1:
        raise ValueError('max_iter must be at least 1')

    if tol <= 0.0:
        raise ValueError('tol must be positive')

    if max_line_search_iters < 0:
        raise ValueError('max_line_search_iters must be non-negative')

    num_time_steps = observations.shape[0]

    if initial_latents is None:
        latents = model.get_prior_mean_latents(num_time_steps)

    else:
        latents = initial_latents

    latent_chain = model.to_gaussian_chain(num_time_steps=num_time_steps)

    emissions = model.emissions

    search = NewtonModeSearch(
        emissions=emissions,
        latent_chain=latent_chain,
        observations=observations,
        max_line_search_iters=max_line_search_iters,
        tol=tol,
    )

    current_objective = _laplace_objective(latent_chain, emissions, observations, latents)

    initial = search.initial_state(
        latents=latents,
        current_objective=current_objective,
    )

    final = search.iterate(initial, max_iter=max_iter)

    latents = final.latents

    # Rebuild the approximation at the final mode. This matters:
    # the covariance of the Laplace approximation must come from the
    # Hessian evaluated at the final reference state.
    observation_potential = emissions.get_local_potential(observations, latents=latents)

    posterior_chain = latent_chain.add_local_potential(observation_potential)

    return posterior_chain.forward_backward()
