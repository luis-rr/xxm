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
from xxm.core.emissions.continuous import QuadraticEmissionsT

from .core import Model, Posterior


class ContinuousFactors(typing.NamedTuple):
    """Non-switching observation factors of the SLDS."""

    observations: GaussianPotential

    @classmethod
    def from_model(
        cls,
        model: Model[QuadraticEmissionsT],
        observations: jax.Array,
    ) -> typing.Self:
        return cls(observations=model.emissions.compute_potential(observations))

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

        chain = chain.add_local_potential(self.observations)

        return chain.forward_backward()


class DiscreteFactors(typing.NamedTuple):
    """Non-switching Markov-chain factors of the SLDS."""

    chain: DiscreteChain

    @classmethod
    def from_model(
        cls,
        model: Model[QuadraticEmissionsT],
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
    def from_model(cls, model: Model[QuadraticEmissionsT]) -> typing.Self:
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


def infer_variational(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
    num_iters: int,
) -> tuple[Posterior, jax.Array]:
    """Run structured mean-field inference for the SLDS."""
    num_steps = observations.shape[0]

    switching_factors = SwitchingFactors.from_model(
        model,
    )

    continuous_factors = ContinuousFactors.from_model(
        model,
        observations,
    )

    discrete_factors = DiscreteFactors.from_model(
        model,
        num_steps,
    )

    def infer_continuous(
        discrete_posterior: DiscretePosterior,
    ) -> tuple[ContinuousPosterior, jax.Array]:
        initial_potential, dynamics_potential = switching_factors.continuous_potentials(
            discrete_posterior
        )

        return continuous_factors.infer(
            initial_potential,
            dynamics_potential,
        )

    def step(_, discrete_posterior):
        """Perform one coordinate-ascent update of q(x) and q(z)."""
        continuous_posterior, _ = infer_continuous(
            discrete_posterior,
        )

        discrete_posterior = discrete_factors.infer(
            switching_factors.discrete_potential(
                continuous_posterior,
            )
        )

        return discrete_posterior

    discrete_posterior = discrete_factors.prior()

    discrete_posterior = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        discrete_posterior,
    )

    continuous_posterior, continuous_log_normalizer = infer_continuous(
        discrete_posterior,
    )

    posterior = Posterior(
        discrete=discrete_posterior,
        continuous=continuous_posterior,
    )

    elbo_value = posterior.elbo_from_chain(
        continuous_log_normalizer=continuous_log_normalizer,
        discrete_prior=discrete_factors.chain,
    )

    return posterior, elbo_value
