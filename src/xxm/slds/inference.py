import typing

import jax

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
    """Non-switching Gaussian factors of the SLDS."""

    observations: GaussianPotential
    initial: GaussianPotential

    @classmethod
    def from_model(
        cls,
        model: Model[QuadraticEmissionsT],
        observations: jax.Array,
    ) -> typing.Self:
        return cls(
            observations=model.emissions.compute_potential(observations),
            initial=GaussianPotential.from_moments(
                model.latent_initial.model,
            ),
        )

    def infer(
        self,
        continuous_potential: GaussianPairPotential,
    ) -> tuple[ContinuousPosterior, jax.Array]:
        """Infer the continuous posterior q(x) for a given dynamics potential."""

        chain = GaussianChain.from_pair_potentials(
            self.initial,
            continuous_potential,
        )
        chain = chain.add_local_potential(
            self.observations,
        )
        return chain.forward_backward()


class DiscreteFactors(typing.NamedTuple):
    """Non-switching discrete factors of the SLDS."""

    chain: DiscreteChain

    @classmethod
    def from_model(
        cls,
        model: Model[QuadraticEmissionsT],
        num_switch_steps: int,
    ) -> typing.Self:

        transition_probs = model.transitions.model.broadcast(
            (num_switch_steps - 1,)
        ).probs

        return cls(
            chain=DiscreteChain.from_markov_prior(
                initial_probs=model.state_initial.model.probs,
                transition_probs=transition_probs,
            )
        )

    def infer(
        self,
        discrete_potential: DiscretePotential,
    ) -> DiscretePosterior:
        """Infer the discrete posterior q(z) for a given state potential."""

        chain = self.chain.add_local_potential(
            discrete_potential,
        )
        posterior, _ = chain.forward_backward()
        return posterior

    def prior(self) -> DiscretePosterior:
        """Return the discrete posterior under the Markov-chain prior."""

        posterior, _ = self.chain.forward_backward()

        return posterior


class SwitchingFactors(typing.NamedTuple):
    """State-dependent dynamics factors coupling the discrete and continuous latents."""

    dynamics_potential: GaussianPairPotential

    @classmethod
    def from_model(
        cls,
        model: Model[QuadraticEmissionsT],
    ) -> typing.Self:

        dynamics = model.dynamics.compute_pair_potentials()

        return cls(
            dynamics_potential=dynamics,
        )

    def continuous_potential(
        self,
        discrete_posterior: DiscretePosterior,
    ) -> GaussianPairPotential:
        """Compute the expected Gaussian dynamics potential under q(z)."""

        return self.dynamics_potential.weighted_sum(
            discrete_posterior.state_probs,
        )

    def discrete_potential(
        self,
        continuous_posterior: ContinuousPosterior,
    ) -> DiscretePotential:
        """Compute the expected discrete state potential under q(x)."""

        return DiscretePotential(
            log_values=self.dynamics_potential.expected_log_potentials(
                continuous_posterior,
            )
        )


def infer_variational(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
    num_iters: int,
) -> tuple[Posterior, jax.Array]:
    """Run structured mean-field inference for the SLDS."""

    num_cont_steps = observations.shape[0]
    num_switch_steps = num_cont_steps - 1

    switch_factors = SwitchingFactors.from_model(model)
    cont_factors = ContinuousFactors.from_model(model, observations)
    disc_factors = DiscreteFactors.from_model(model, num_switch_steps)

    def step(_, disc_posterior):
        """Perform one coordinate-ascent update of q(x) and q(z)."""

        cont_potential = switch_factors.continuous_potential(disc_posterior)
        cont_posterior, _ = cont_factors.infer(cont_potential)

        disc_potential = switch_factors.discrete_potential(cont_posterior)
        disc_posterior = disc_factors.infer(disc_potential)

        return disc_posterior

    disc_posterior = disc_factors.prior()

    disc_posterior = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        disc_posterior,
    )

    cont_posterior, cont_log_normalizer = cont_factors.infer(
        switch_factors.continuous_potential(disc_posterior),
    )

    posterior = Posterior(
        discrete=disc_posterior,
        continuous=cont_posterior,
    )

    elbo_value = posterior.elbo_from_chain(
        continuous_log_normalizer=cont_log_normalizer,
        discrete_prior=disc_factors.chain,
    )

    return posterior, elbo_value
