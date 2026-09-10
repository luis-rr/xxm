"""
Structured mean-field inference for Switching Linear Dynamical Systems.
"""

import typing

import jax
import jax.numpy as jnp

from xxm.core.chains.discrete import (
    DiscreteChain,
    DiscretePotential,
)
from xxm.core.chains.discrete import (
    DiscreteChainMarginals as DiscretePosterior,
)
from xxm.core.chains.gaussian import (
    GaussianChain,
    GaussianPairPotential,
    GaussianPotential,
)
from xxm.core.chains.gaussian import (
    GaussianChainMarginals as ContinuousPosterior,
)
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.emissions.continuous import (
    EmissionsT,
    LaplaceEmissionsT,
    QuadraticEmissionsT,
)
from xxm.core.inference import Inferred
from xxm.core.optim.laplace import laplace_inference, local_gaussian_approximation
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model, Posterior


class QuadraticContinuousFactors(typing.NamedTuple):
    """Non-switching quadratic observation factors of the SLDS."""

    observations: GaussianPotential

    @classmethod
    def from_model(
        cls,
        model: Model[QuadraticEmissionsT],
        observations: jax.Array,
    ) -> typing.Self:
        """Build quadratic observation factors from an SLDS and observations."""
        return cls(
            observations=model.emissions.compute_potential(observations),
        )

    def infer(
        self,
        initial_potential: GaussianPotential,
        dynamics_potential: GaussianPairPotential,
    ) -> tuple[ContinuousPosterior, jax.Array]:
        """Infer q(x) from expected initial and dynamics potentials."""

        chain = GaussianChain.from_pair_potentials(
            initial_potential,
            dynamics_potential,
        )

        chain = chain.add_local_potential(
            self.observations,
        )

        return chain.forward_backward()


class LaplaceContinuousFactors(
    typing.NamedTuple,
    typing.Generic[LaplaceEmissionsT],
):
    """Non-switching non-conjugate observation factors of the SLDS."""

    emissions: LaplaceEmissionsT
    observations: jax.Array
    search_params: OptimParams

    @classmethod
    def from_model(
        cls,
        model: Model[LaplaceEmissionsT],
        observations: jax.Array,
        search_params: OptimParams,
    ) -> typing.Self:
        """Build Laplace observation factors from an SLDS and observations."""

        return cls(
            emissions=model.emissions,
            observations=observations,
            search_params=search_params,
        )

    def approximate(
        self,
        initial_potential: GaussianPotential,
        dynamics_potential: GaussianPairPotential,
        latents: jax.Array,
    ) -> tuple[ContinuousPosterior, jax.Array]:
        """Construct a local Gaussian approximation around `latents`."""

        chain = GaussianChain.from_pair_potentials(
            initial_potential,
            dynamics_potential,
        )

        return local_gaussian_approximation(
            chain=chain,
            emissions=self.emissions,
            observations=self.observations,
            latents=latents,
        )

    def infer(
        self,
        initial_potential: GaussianPotential,
        dynamics_potential: GaussianPairPotential,
        initial_latents: jax.Array,
    ) -> tuple[ContinuousPosterior, jax.Array]:
        """Infer q(x) using a local Laplace approximation."""

        chain = GaussianChain.from_pair_potentials(
            initial_potential,
            dynamics_potential,
        )

        return laplace_inference(
            chain=chain,
            emissions=self.emissions,
            observations=self.observations,
            initial_latents=initial_latents,
            search_params=self.search_params,
        )


class DiscreteFactors(typing.NamedTuple):
    """Non-switching Markov-chain factors of the SLDS."""

    chain: DiscreteChain

    @classmethod
    def from_model(
        cls,
        model: Model[EmissionsT],
        num_steps: int,
    ) -> typing.Self:
        """Build the discrete Markov-chain prior for `num_steps` states."""
        transition_probs = model.transitions.dist.broadcast((num_steps - 1,)).probs

        return cls(
            chain=DiscreteChain.from_markov_prior(
                initial_probs=model.state_initial.dist.probs,
                transition_probs=transition_probs,
            )
        )

    def infer(self, discrete_potential: DiscretePotential) -> DiscretePosterior:
        """Infer q(z) for a given state potential."""
        chain = self.chain.add_local_potential(
            discrete_potential,
        )

        posterior, _ = chain.forward_backward()

        return posterior


class SwitchingFactors(typing.NamedTuple):
    """State-dependent factors coupling discrete and continuous latents."""

    initial_dist: Gaussian  # (K,)
    dynamics_dist: LinearGaussian  # (K,)

    initial_potential: GaussianPotential  # (K,)
    dynamics_potential: GaussianPairPotential  # (K,)

    @classmethod
    def from_model(
        cls,
        model: Model[EmissionsT],
    ) -> typing.Self:
        """Build state-dependent initial and transition factors from an SLDS."""
        return cls(
            initial_dist=model.latent_initial.dist,
            dynamics_dist=model.dynamics.dist,
            initial_potential=model.latent_initial.compute_potentials(),
            dynamics_potential=model.dynamics.compute_pair_potentials(),
        )

    def continuous_potentials(
        self,
        discrete_posterior: DiscretePosterior,
    ) -> tuple[GaussianPotential, GaussianPairPotential]:
        """Compute expected Gaussian factors under q(z)."""
        initial = self.initial_potential.weighted_sum(
            discrete_posterior.state_probs[0],
        )

        dynamics = self.dynamics_potential.weighted_sum(
            discrete_posterior.state_probs[1:],
        )

        return initial, dynamics

    def discrete_potential(
        self,
        continuous_posterior: ContinuousPosterior,
    ) -> DiscretePotential:
        """Compute expected state potentials under q(x)."""

        means = continuous_posterior.means
        covariances = continuous_posterior.covariances
        cross_covariances = continuous_posterior.cross_covariances

        initial_log_values = self.initial_dist.expected_log_prob(
            Gaussian(
                mean=means[0],
                covariance=covariances[0],
            )
        )  # (K,)

        dynamics_log_values = self.dynamics_dist.expected_log_prob_broadcast(
            input=Gaussian(
                mean=means[:-1],
                covariance=covariances[:-1],
            ),
            output=Gaussian(
                mean=means[1:],
                covariance=covariances[1:],
            ),
            input_output_covariance=cross_covariances,
        )  # (T - 1, K)

        return DiscretePotential(
            log_values=jnp.concatenate(
                [
                    initial_log_values[None, :],
                    dynamics_log_values,
                ],
                axis=0,
            )
        )

    def discrete_potential_from_latents(
        self,
        latents: jax.Array,
    ) -> DiscretePotential:
        """Compute state potentials for a deterministic latent trajectory."""

        initial_log_values = self.initial_dist.log_prob_broadcast(
            latents[0],
        )

        dynamics_log_values = self.dynamics_dist.conditional(
            latents[:-1, None, :],
        ).log_prob(
            latents[1:, None, :],
        )

        return DiscretePotential(
            log_values=jnp.concatenate(
                [
                    initial_log_values[None, :],
                    dynamics_log_values,
                ],
                axis=0,
            )
        )


class QuadraticVI(typing.NamedTuple):
    """Structured mean-field factors for conjugate SLDS inference."""

    switching: SwitchingFactors
    continuous: QuadraticContinuousFactors
    discrete: DiscreteFactors

    @classmethod
    def from_model(
        cls,
        model: Model[QuadraticEmissionsT],
        observations: jax.Array,
    ) -> typing.Self:
        """Construct all variational factors for a quadratic-emission SLDS."""
        num_steps = observations.shape[0]

        return cls(
            switching=SwitchingFactors.from_model(model),
            continuous=QuadraticContinuousFactors.from_model(
                model,
                observations,
            ),
            discrete=DiscreteFactors.from_model(model, num_steps),
        )

    def infer_continuous(
        self,
        discrete_posterior: DiscretePosterior,
    ) -> tuple[ContinuousPosterior, jax.Array]:
        """Update $q(x)$ using expected factors under $q(z)$."""

        initial_potential, dynamics_potential = self.switching.continuous_potentials(
            discrete_posterior
        )

        return self.continuous.infer(
            initial_potential,
            dynamics_potential,
        )

    def infer_discrete(
        self,
        continuous_posterior: ContinuousPosterior,
    ) -> DiscretePosterior:
        """Update $q(z)$ using expected factors under $q(x)$."""
        return self.discrete.infer(
            self.switching.discrete_potential(
                continuous_posterior,
            )
        )


def _quadratic_elbo(
    discrete_posterior: DiscretePosterior,
    continuous_log_normalizer: jax.Array,
    discrete_prior: DiscreteChain,
) -> jax.Array:
    """Evidence lower bound for the structured SLDS posterior."""
    return (
        continuous_log_normalizer
        + discrete_posterior.expected_log_potential(discrete_prior)
        + discrete_posterior.entropy()
    )


def infer_variational(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
    num_iters: int,
    initial_latents: jax.Array,
) -> Inferred[Model[QuadraticEmissionsT], Posterior]:
    """
    Run structured mean-field inference with conjugate Gaussian updates for $q(x)$.

    `initial_latents` must be provided to initialize $q(z)$ by scoring that
    deterministic trajectory under the switching model before the first continuous update.
    """

    inference = QuadraticVI.from_model(
        model,
        observations,
    )

    def step(
        _,
        discrete_posterior: DiscretePosterior,
    ) -> DiscretePosterior:
        """Perform one coordinate-ascent update of q(x) and q(z)."""

        continuous_posterior, _ = inference.infer_continuous(
            discrete_posterior,
        )

        return inference.infer_discrete(
            continuous_posterior,
        )

    discrete_posterior = inference.discrete.infer(
        inference.switching.discrete_potential_from_latents(
            initial_latents,
        )
    )

    discrete_posterior = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        discrete_posterior,
    )

    (
        continuous_posterior,
        continuous_log_normalizer,
    ) = inference.infer_continuous(
        discrete_posterior,
    )

    posterior = Posterior(
        discrete=discrete_posterior,
        continuous=continuous_posterior,
    )

    objective = _quadratic_elbo(
        discrete_posterior=discrete_posterior,
        continuous_log_normalizer=continuous_log_normalizer,
        discrete_prior=inference.discrete.chain,
    )

    return Inferred(
        model=model,
        posterior=posterior,
        objective=objective,
    )


class LaplaceState(typing.NamedTuple):
    """Iterative state for Laplace structured mean-field inference."""

    discrete: DiscretePosterior
    continuous: ContinuousPosterior


class LaplaceVI(typing.NamedTuple, typing.Generic[LaplaceEmissionsT]):
    """Structured mean-field factors with local Laplace updates for q(x)."""

    switching: SwitchingFactors
    continuous: LaplaceContinuousFactors[LaplaceEmissionsT]
    discrete: DiscreteFactors

    @classmethod
    def from_model(
        cls,
        model: Model[LaplaceEmissionsT],
        observations: jax.Array,
        search_params: OptimParams,
    ) -> typing.Self:
        """Construct all variational factors for a nonconjugate SLDS."""

        num_steps = observations.shape[0]

        return cls(
            switching=SwitchingFactors.from_model(
                model,
            ),
            continuous=LaplaceContinuousFactors.from_model(
                model,
                observations,
                search_params=search_params,
            ),
            discrete=DiscreteFactors.from_model(
                model,
                num_steps,
            ),
        )

    def approximate_continuous(
        self,
        discrete_posterior: DiscretePosterior,
        latents: jax.Array,
    ) -> ContinuousPosterior:
        """Construct a local q(x) approximation around `latents`."""

        (
            initial_potential,
            dynamics_potential,
        ) = self.switching.continuous_potentials(
            discrete_posterior,
        )

        posterior, _ = self.continuous.approximate(
            initial_potential,
            dynamics_potential,
            latents=latents,
        )

        return posterior

    def infer_continuous(
        self,
        discrete_posterior: DiscretePosterior,
        initial_latents: jax.Array,
    ) -> ContinuousPosterior:
        """Update q(x) under the current q(z)."""

        (initial_potential, dynamics_potential) = self.switching.continuous_potentials(
            discrete_posterior,
        )

        posterior, _ = self.continuous.infer(
            initial_potential,
            dynamics_potential,
            initial_latents=initial_latents,
        )

        return posterior

    def infer_discrete(
        self,
        continuous_posterior: ContinuousPosterior,
    ) -> DiscretePosterior:
        """Update q(z) using expected continuous-state factors."""

        return self.discrete.infer(
            self.switching.discrete_potential(
                continuous_posterior,
            )
        )

    def initial_state(
        self,
        initial_latents: jax.Array,
    ) -> LaplaceState:
        """Construct an observation-informed initial variational state."""

        discrete_posterior = self.discrete.infer(
            self.switching.discrete_potential_from_latents(
                initial_latents,
            )
        )

        continuous_posterior = self.approximate_continuous(
            discrete_posterior,
            initial_latents,
        )

        return LaplaceState(
            discrete=discrete_posterior,
            continuous=continuous_posterior,
        )


def _laplace_elbo(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    continuous_posterior: ContinuousPosterior,
    discrete_posterior: DiscretePosterior,
    switching_factors: SwitchingFactors,
    discrete_prior: DiscreteChain,
) -> jax.Array:
    """Evaluate the structured mean-field ELBO."""

    latent_potential = switching_factors.discrete_potential(
        continuous_posterior,
    )

    expected_latent_log_prob = jnp.sum(
        discrete_posterior.state_probs * latent_potential.log_values,
    )

    expected_observation_log_prob = model.emissions.expected_log_likelihood(
        observations,
        continuous_posterior,
    )

    expected_discrete_log_prob = discrete_posterior.expected_log_potential(
        discrete_prior,
    )

    return (
        expected_observation_log_prob
        + expected_latent_log_prob
        + expected_discrete_log_prob
        + continuous_posterior.entropy()
        + discrete_posterior.entropy()
    )


def infer_laplace(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    num_iters: int,
    initial_latents: jax.Array,
    params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> Inferred[Model[LaplaceEmissionsT], Posterior]:
    """
    Run structured mean-field inference with Laplace updates for q(x).

    The initial continuous posterior is constructed from a local Gaussian
    approximation that includes the observation likelihood. Coordinate updates
    then alternate q(z) followed by q(x).
    """

    params = params or OptimParams()

    inference = LaplaceVI.from_model(
        model,
        observations,
        search_params=params,
    )

    def step(_, state: LaplaceState) -> LaplaceState:
        """Perform one coordinate update of q(z) followed by q(x)."""

        discrete_posterior = inference.infer_discrete(state.continuous)

        continuous_posterior = inference.infer_continuous(
            discrete_posterior,
            initial_latents=state.continuous.means,
        )

        return LaplaceState(
            discrete=discrete_posterior,
            continuous=continuous_posterior,
        )

    state = inference.initial_state(initial_latents)

    state = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        state,
    )

    return Inferred(
        model=model,
        posterior=Posterior(
            discrete=state.discrete,
            continuous=state.continuous,
        ),
        objective=_laplace_elbo(
            model=model,
            observations=observations,
            continuous_posterior=state.continuous,
            discrete_posterior=state.discrete,
            switching_factors=inference.switching,
            discrete_prior=inference.discrete.chain,
        ),
    )
