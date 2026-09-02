import typing

import jax
import jax.numpy as jnp

from xxm.core.chains.discrete import DiscreteChain, DiscreteChainMarginals
from xxm.core.chains.gaussian import GaussianChainMarginals, GaussianPairPotential
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.models.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.models.gaussian import GaussianInitial
from xxm.core.optim import gaussian as gaussian_fit
from xxm.lds.core import EmissionsT


class Posterior(typing.NamedTuple):
    discrete: DiscreteChainMarginals  # T-1 discrete states
    continuous: GaussianChainMarginals  # T latents

    def elbo_from_chain(
        self,
        continuous_log_normalizer: jax.Array,
        discrete_prior: DiscreteChain,
    ) -> jax.Array:
        """Evidence lower bound for the structured SLDS posterior."""

        discrete = self.discrete

        return (
            continuous_log_normalizer
            + discrete.expected_log_potential(discrete_prior)
            + discrete.entropy()
        )


class GaussianLinearSwitchingDynamics(typing.NamedTuple):
    model: LinearGaussian  # K-batched, input dimension D, output dimension D

    @property
    def num_states(self) -> int:
        return self.model.covariance.shape[0]

    def compute_pair_potentials(self) -> GaussianPairPotential:
        """Return one Gaussian pair potential for each discrete state."""
        return GaussianPairPotential.from_linear_conditional(self.model)

    def fit_params(
        self,
        posterior: Posterior,
    ) -> typing.Self:
        """Fit one linear-Gaussian dynamics model per discrete state."""

        weights = posterior.discrete.state_probs  # (T-1, K)

        means = posterior.continuous.means
        second = posterior.continuous.raw_second_moments()
        cross = posterior.continuous.raw_cross_moments()

        def fit_state(weights_k):
            total = jnp.sum(weights_k)

            input_mean = jnp.einsum('t,ti->i', weights_k, means[:-1]) / total
            output_mean = jnp.einsum('t,ti->i', weights_k, means[1:]) / total
            input_second = jnp.einsum('t,tij->ij', weights_k, second[:-1]) / total
            output_second = jnp.einsum('t,tij->ij', weights_k, second[1:]) / total

            # raw_cross_moments[t] = E[x_t x_{t+1}^T],
            # while the fitter expects E[x_{t+1} x_t^T].
            output_input = (
                jnp.einsum(
                    't,tij->ij',
                    weights_k,
                    jnp.swapaxes(cross, -1, -2),
                )
                / total
            )

            return gaussian_fit.linear_from_moments(
                input_mean=input_mean,
                output_mean=output_mean,
                input_second_moment=input_second,
                output_second_moment=output_second,
                output_input_moment=output_input,
            )

        linear_gaussian = jax.vmap(
            fit_state,
            in_axes=1,
        )(weights)

        return self._replace(
            model=linear_gaussian,
        )


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    state_initial: CategoricalInitial
    transitions: CategoricalTransitions
    latent_initial: GaussianInitial
    dynamics: GaussianLinearSwitchingDynamics
    emissions: EmissionsT

    @property
    def num_states(self) -> int:
        return self.dynamics.num_states

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self:

        return self.__class__(
            state_initial=self.state_initial.fit_params(posterior.discrete),
            transitions=self.transitions.fit_params(posterior.discrete),
            latent_initial=self.latent_initial.fit_params(posterior.continuous),
            dynamics=self.dynamics.fit_params(posterior),
            emissions=self.emissions.fit_params(observations, posterior.continuous),
        )
