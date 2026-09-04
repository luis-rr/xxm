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
from xxm.core.emissions.continuous import (
    EmissionsT,
    LaplaceEmissionsT,
    QuadraticEmissionsT,
)
from xxm.core.optim.laplace import laplace_inference
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model, Posterior


class Inferred(typing.NamedTuple):
    posterior: Posterior
    objective: jax.Array


class QuadraticContinuousFactors(typing.NamedTuple):
    """Non-switching quadratic observation factors of the SLDS."""

    observations: GaussianPotential

    @classmethod
    def from_model(
        cls,
        model: Model[QuadraticEmissionsT],
        observations: jax.Array,
    ) -> typing.Self:
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
        return cls(
            emissions=model.emissions,
            observations=observations,
            search_params=search_params,
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
        transition_probs = model.transitions.model.broadcast((num_steps - 1,)).probs

        return cls(
            chain=DiscreteChain.from_markov_prior(
                initial_probs=model.state_initial.model.probs,
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

    def prior(self) -> DiscretePosterior:
        """Return q(z) under the discrete Markov prior."""
        posterior, _ = self.chain.forward_backward()

        return posterior


class SwitchingFactors(typing.NamedTuple):
    """State-dependent factors coupling discrete and continuous latents."""

    initial_potential: GaussianPotential  # (K,)
    dynamics_potential: GaussianPairPotential  # (K,)

    @classmethod
    def from_model(cls, model: Model[EmissionsT]) -> typing.Self:
        return cls(
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
        second = continuous_posterior.raw_second_moments()

        initial_log_values = self.initial_potential.expected_log_potentials(
            mean=continuous_posterior.means[0],
            second_moment=second[0],
        )  # (K,)

        dynamics_log_values = self.dynamics_potential.expected_log_potentials(
            continuous_posterior,
        )  # (T-1, K)

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
    switching: SwitchingFactors
    continuous: QuadraticContinuousFactors
    discrete: DiscreteFactors

    @classmethod
    def from_model(
        cls,
        model: Model[QuadraticEmissionsT],
        observations: jax.Array,
    ) -> typing.Self:
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
) -> Inferred:
    """Run structured mean-field inference for the SLDS."""
    inference = QuadraticVI.from_model(model, observations)

    def step(_, discrete_posterior):
        """Perform one coordinate-ascent update of q(x) and q(z)."""
        continuous_posterior, _ = inference.infer_continuous(discrete_posterior)
        discrete_posterior = inference.infer_discrete(continuous_posterior)

        return discrete_posterior

    discrete_posterior = inference.discrete.prior()

    discrete_posterior = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        discrete_posterior,
    )

    continuous_posterior, continuous_log_normalizer = inference.infer_continuous(
        discrete_posterior,
    )

    posterior = Posterior(
        discrete=discrete_posterior,
        continuous=continuous_posterior,
    )

    elbo_value = _quadratic_elbo(
        discrete_posterior=discrete_posterior,
        continuous_log_normalizer=continuous_log_normalizer,
        discrete_prior=inference.discrete.chain,
    )

    return Inferred(posterior, elbo_value)


class LaplaceState(typing.NamedTuple):
    discrete: DiscretePosterior
    latents: jax.Array


class LaplaceVI(typing.NamedTuple):
    switching: SwitchingFactors
    continuous: LaplaceContinuousFactors
    discrete: DiscreteFactors

    @classmethod
    def from_model(
        cls,
        model: Model[LaplaceEmissionsT],
        observations: jax.Array,
        search_params: OptimParams,
    ) -> typing.Self:
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

    def infer_continuous(
        self, state: LaplaceState
    ) -> tuple[ContinuousPosterior, jax.Array]:

        initial_potential, dynamics_potential = self.switching.continuous_potentials(
            state.discrete,
        )

        return self.continuous.infer(
            initial_potential,
            dynamics_potential,
            initial_latents=state.latents,
        )

    def infer_discrete(
        self,
        continuous_posterior: ContinuousPosterior,
    ) -> DiscretePosterior:

        return self.discrete.infer(
            self.switching.discrete_potential(
                continuous_posterior,
            )
        )

    def initial_latents(self, discrete_posterior: DiscretePosterior) -> jax.Array:
        initial_potential, dynamics_potential = self.switching.continuous_potentials(
            discrete_posterior,
        )

        prior_chain = GaussianChain.from_pair_potentials(
            initial_potential,
            dynamics_potential,
        )

        prior_posterior, _ = prior_chain.forward_backward()

        return prior_posterior.means

    def initial_state(self, initial_latents: jax.Array | None = None) -> LaplaceState:
        discrete_posterior = self.discrete.prior()

        if initial_latents is None:
            initial_latents = self.initial_latents(discrete_posterior)

        return LaplaceState(
            discrete=discrete_posterior,
            latents=initial_latents,
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
        discrete_posterior.state_probs * latent_potential.log_values
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
    initial_latents: jax.Array | None = None,
    params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> Inferred:
    """Run structured mean-field inference with Laplace continuous updates."""

    params = params or OptimParams()

    inference = LaplaceVI.from_model(
        model,
        observations,
        search_params=params,
    )

    def step(_, state: LaplaceState) -> LaplaceState:
        """Perform one coordinate update of q(x) and q(z)."""

        continuous_posterior, _ = inference.infer_continuous(state)

        discrete_posterior = inference.infer_discrete(continuous_posterior)

        return LaplaceState(
            discrete=discrete_posterior,
            latents=continuous_posterior.means,
        )

    state = inference.initial_state(initial_latents)

    state: LaplaceState = jax.lax.fori_loop(0, num_iters, step, state)

    continuous_posterior, _ = inference.infer_continuous(state)

    return Inferred(
        posterior=Posterior(
            discrete=state.discrete,
            continuous=continuous_posterior,
        ),
        objective=_laplace_elbo(
            model=model,
            observations=observations,
            continuous_posterior=continuous_posterior,
            discrete_posterior=state.discrete,
            switching_factors=inference.switching,
            discrete_prior=inference.discrete.chain,
        ),
    )
