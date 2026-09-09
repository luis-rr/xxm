"""Posterior marginal distribution protocols."""

import typing

import jax

from xxm.core.dists.gaussian import PairedGaussian


class DiscretePosterior(typing.Protocol):
    r"""Protocol for discrete posterior marginals.

    Represents marginal probabilities $\gamma_t(k)$ of discrete latent states
    and pair marginals $\xi_t(i,j)$ under a represented distribution $q(z)$,
    which may be an exact or approximate posterior given observations.
    """

    @property
    def state_probs(self) -> jax.Array:  # (T, K)
        r"""State marginals $\gamma_t(k)=q(z_t=k)$."""
        ...

    @property
    def pair_probs(self) -> jax.Array:  # (T-1, K, K)
        r"""Adjacent-state marginals $\xi_t(i,j)=q(z_t=i,z_{t+1}=j)$."""
        ...


class ContinuousPosterior(typing.Protocol):
    r"""Protocol for continuous Gaussian posterior marginals.

    Represents marginal means, covariances, and raw moments across time under
    a Gaussian distribution $q(x)$, which may be an exact or approximate
    posterior given observations.
    """

    @property
    def means(self) -> jax.Array:  # (T, D)
        r"""Marginal means $\mathbb{E}_q[x_t]$."""
        ...

    @property
    def covariances(self) -> jax.Array:  # (T, D, D)
        r"""Marginal covariances $\operatorname{Cov}_q(x_t)$."""
        ...

    @property
    def cross_covariances(self) -> jax.Array:  # (T-1, D, D)
        r"""Cross-covariances $\operatorname{Cov}_q(x_t,x_{t+1})$."""
        ...

    def raw_second_moments(self) -> jax.Array:  # (T, D, D)
        r"""Raw second moments $\mathbb{E}_q[x_t x_t^\top]$."""
        ...

    def raw_cross_moments(self) -> jax.Array:  # (T-1, D, D)
        r"""Raw cross-moments $\mathbb{E}_q[x_t x_{t+1}^\top]$."""
        ...

    def paired_marginals(self) -> PairedGaussian:
        r"""Return adjacent Gaussian marginals $q(x_t, x_{t+1})$."""
        ...
