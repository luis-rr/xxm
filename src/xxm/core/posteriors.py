"""Posterior marginal distribution protocols."""

import typing

import jax


class DiscretePosterior(typing.Protocol):
    r"""Protocol for discrete posterior marginals.

    Represents marginal probabilities $\gamma_t(k)$ of discrete latent states
    and pair marginals $\xi_t(i,j)$.
    """

    @property
    def state_probs(self) -> jax.Array:  # (T, K)
        r"""State marginals $\gamma_t(k)=p(z_t=k\mid y)$."""
        ...

    @property
    def pair_probs(self) -> jax.Array:  # (T-1, K, K)
        r"""Adjacent-state marginals $\xi_t(i,j)=p(z_t=i,z_{t+1}=j\mid y)$."""
        ...


class ContinuousPosterior(typing.Protocol):
    r"""Protocol for continuous Gaussian posterior marginals.

    Represents marginal means and covariances, and cross-moments across time.
    """

    @property
    def means(self) -> jax.Array:  # (T, D)
        r"""Posterior marginal means $\mathbb{E}[x_t\mid y]$."""
        ...

    @property
    def covariances(self) -> jax.Array:  # (T, D, D)
        r"""Posterior marginal covariances $\operatorname{Cov}(x_t\mid y)$."""
        ...

    @property
    def cross_covariances(self) -> jax.Array:  # (T-1, D, D)
        r"""Cross-covariances $\operatorname{Cov}(x_t,x_{t+1}\mid y)$."""
        ...

    def raw_second_moments(self) -> jax.Array:  # (T, D, D)
        r"""Raw second moments $\mathbb{E}[x_t x_t^\top\mid y]$."""
        ...

    def raw_cross_moments(self) -> jax.Array:  # (T-1, D, D)
        r"""Raw cross-moments $\mathbb{E}[x_t x_{t+1}^\top\mid y]$."""
        ...
