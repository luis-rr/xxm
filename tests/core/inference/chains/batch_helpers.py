import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.affine import Affine
from xxm.core.dists.discrete import DiscreteChainMarginals
from xxm.core.dists.gaussian import Gaussian, GaussianChainMarginals, LinearGaussian
from xxm.core.inference.chains.discrete import DiscreteChain
from xxm.core.inference.chains.gaussian import GaussianChain
from xxm.core.lengths import Lengths
from xxm.core.potentials.discrete import DiscreteChainPotentials, DiscretePotential
from xxm.core.potentials.gaussian import (
    GaussianChainPotentials,
    GaussianPairPotential,
    GaussianPotential,
)


def discrete_chain(batch, steps):
    rng = np.random.default_rng(13)
    k = 3
    initial = jax.nn.softmax(jnp.asarray(rng.normal(size=batch + (k,))), axis=-1)
    transitions = jax.nn.softmax(
        jnp.asarray(rng.normal(size=batch + (steps - 1, k, k))), axis=-1
    )
    local = DiscretePotential(jnp.asarray(rng.normal(size=batch + (steps, k))))
    return DiscreteChain.from_markov_prior(
        initial, transitions, Lengths(jnp.full(batch, steps))
    ).add_local_potential(local)


def gaussian_chain(batch, steps):
    d = 2
    rng = np.random.default_rng(32)
    initial = Gaussian(
        jnp.asarray(rng.normal(size=batch + (d,))),
        jnp.broadcast_to(jnp.eye(d), batch + (d, d)),
    )
    pairs = LinearGaussian(
        Affine(
            jnp.asarray(rng.normal(scale=0.1, size=batch + (steps - 1, d, d))),
            jnp.asarray(rng.normal(size=batch + (steps - 1, d))),
        ),
        jnp.broadcast_to(jnp.eye(d), batch + (steps - 1, d, d)),
    )
    chain = GaussianChain.from_pair_potentials(
        GaussianPotential.from_moments(initial),
        GaussianPairPotential.from_linear_conditional(pairs),
        Lengths(jnp.full(batch, steps)),
    )
    local = GaussianPotential.from_moments(
        Gaussian(
            jnp.asarray(rng.normal(size=batch + (steps, d))),
            jnp.broadcast_to(jnp.eye(d), batch + (steps, d, d)),
        )
    )
    return chain.add_local_potential(local)


def gaussian_chain_structure(batch, steps):
    d = 2
    return GaussianChain(
        lengths=Lengths(jnp.full(batch, steps)),
        diagonal_precision_blocks=jnp.broadcast_to(jnp.eye(d), batch + (steps, d, d)),
        lower_precision_blocks=jnp.zeros(batch + (steps - 1, d, d)),
        information_vectors=jnp.zeros(batch + (steps, d)),
        log_constant=jnp.zeros(batch),
    )


def discrete_chain_structure(batch, steps):
    k = 3
    return DiscreteChain(
        lengths=Lengths(jnp.full(batch, steps)),
        initial_probs=jnp.full(batch + (k,), 1.0 / k),
        transition_probs=jnp.full(batch + (steps - 1, k, k), 1.0 / k),
        state_log_potentials=jnp.zeros(batch + (steps, k)),
    )


def gaussian_marginals(batch, steps):
    d = 2
    return GaussianChainMarginals(
        lengths=Lengths(jnp.full(batch, steps)),
        means=jnp.zeros(batch + (steps, d)),
        covariances=jnp.broadcast_to(jnp.eye(d), batch + (steps, d, d)),
        cross_covariances=jnp.zeros(batch + (steps - 1, d, d)),
    )


def discrete_marginals(batch, steps):
    k = 3
    state_probs = jnp.full(batch + (steps, k), 1.0 / k)
    pair_probs = state_probs[..., :-1, :, None] * state_probs[..., 1:, None, :]
    return DiscreteChainMarginals(
        state_probs, pair_probs, Lengths(jnp.full(batch, steps))
    )


def expected_log_potential(chain, posterior):
    """Evaluate the inference chain's factors through their potential APIs."""
    if isinstance(chain, DiscreteChain):
        prior = DiscreteChainPotentials(
            chain.initial_probs, chain.transition_probs, chain.lengths
        )
        local = DiscretePotential(chain.state_log_potentials)
        return prior.expected_log_potential(posterior) + local.expected_log_potential(
            posterior
        )

    # Allocate the first unary block and global constant to the initial factor;
    # each pair owns its right unary block and the intervening cross term.
    # Leave padding intact to exercise masking in the potential expectation API.
    diagonal = chain.diagonal_precision_blocks
    information = chain.information_vectors
    lower = chain.lower_precision_blocks
    initial = GaussianPotential(
        diagonal[..., 0, :, :], information[..., 0, :], chain.log_constant
    )
    pairs = GaussianPairPotential(
        left_precision=jnp.zeros_like(diagonal[..., 1:, :, :]),
        right_precision=diagonal[..., 1:, :, :],
        lower_precision=lower,
        left_information=jnp.zeros_like(information[..., 1:, :]),
        right_information=information[..., 1:, :],
        log_constant=jnp.zeros_like(information[..., 1:, 0]),
    )
    return GaussianChainPotentials(
        initial, pairs, chain.lengths
    ).expected_log_potential(posterior)
