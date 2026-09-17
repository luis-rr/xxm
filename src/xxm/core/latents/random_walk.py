"""Gaussian random-walk latent processes."""

import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian


def _zero_mean_from_covariance(
    covariance: jax.Array,
) -> jax.Array:
    """Construct a zero mean matching a Gaussian covariance."""
    if covariance.ndim < 2:
        raise ValueError('covariance must have shape (*B, D, D)')

    if covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError('covariance must be square in its final two dimensions')

    # covariance: (*B, D, D) -> mean: (*B, D)
    return jnp.zeros(
        covariance.shape[:-1],
        dtype=covariance.dtype,
    )


def _random_walk_transition(
    innovation: Gaussian,
) -> LinearGaussian:
    """Construct x[t] | x[t-1] from an additive Gaussian innovation."""
    variable_dim = innovation.variable_dim

    coefficients = jnp.broadcast_to(
        jnp.eye(
            variable_dim,
            dtype=innovation.dtype,
        ),
        innovation.batch_shape + (variable_dim, variable_dim),
    )

    return LinearGaussian(
        affine=Affine(
            coefficients=coefficients,
            bias=innovation.mean,
        ),
        covariance=innovation.covariance,
    )


class GaussianRandomWalk(typing.NamedTuple):
    r"""Gaussian random walk, independently replicated over arbitrary batch `*B`.

    `initial` and `innovation` have batch shape `*B` and event dimension `D`.
    Samples have shape `(*B, T, D)`.

    $$
    x_0 \sim \mathcal{N}(\mu_0, Q_0),
    $$

    $$
    x_t
    =
    x_{t-1}
    +
    \epsilon_t,
    \qquad
    \epsilon_t
    \sim
    \mathcal{N}(b, Q).
    $$
    """

    initial: Gaussian
    innovation: Gaussian

    @classmethod
    def from_covariances(
        cls,
        *,
        initial_covariance: jax.Array,
        innovation_covariance: jax.Array,
    ) -> typing.Self:
        """Construct a zero-mean random walk from `(*B, D, D)` covariances."""
        return cls(
            initial=Gaussian(
                mean=_zero_mean_from_covariance(initial_covariance),
                covariance=initial_covariance,
            ),
            innovation=Gaussian(
                mean=_zero_mean_from_covariance(innovation_covariance),
                covariance=innovation_covariance,
            ),
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Independent structural batch dimensions `*B`."""
        if self.initial.batch_shape != self.innovation.batch_shape:
            raise ValueError('initial and innovation batch shapes must match')

        return self.initial.batch_shape

    @property
    def variable_dim(self) -> int:
        """Dimension `D` of each random-walk variable."""
        if self.initial.variable_dim != self.innovation.variable_dim:
            raise ValueError('initial and innovation variable dimensions must match')

        return self.initial.variable_dim

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        """Common parameter dtype."""
        return jnp.result_type(
            self.initial.dtype,
            self.innovation.dtype,
        )

    def transition_dist(self) -> LinearGaussian:
        """Construct the additive Gaussian transition distribution."""
        _ = self.batch_shape
        _ = self.variable_dim

        return _random_walk_transition(self.innovation)

    def sample(self, key: jax.Array, num_steps: int) -> jax.Array:
        """Sample a trajectory with shape `(*B, T, D)`."""
        if num_steps < 0:
            raise ValueError('num_steps must be non-negative')
        batch_shape = self.batch_shape
        variable_dim = self.variable_dim
        if num_steps == 0:
            return jnp.zeros(batch_shape + (0, variable_dim), dtype=self.dtype)
        key_initial, key_innovation = jax.random.split(key)
        initial = self.initial.sample(key_initial)[..., None, :]
        if num_steps == 1:
            return initial
        innovations = self.innovation.sample(
            key_innovation, sample_shape=(num_steps - 1,)
        )
        subsequent = initial + jnp.cumsum(innovations, axis=-2)
        return jnp.concatenate((initial, subsequent), axis=-2)

    def reorient_variables(
        self,
        signs: jax.Array,
    ) -> typing.Self:
        """Reorient random-walk variable dimensions."""
        return self._replace(
            initial=self.initial.reorient_variables(signs),
            innovation=self.innovation.reorient_variables(signs),
        )

    def permute_variables(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel random-walk variable dimensions."""
        return self._replace(
            initial=self.initial.permute_variables(permutation),
            innovation=self.innovation.permute_variables(permutation),
        )

    def shift_variables(
        self,
        offset: jax.Array,
    ) -> typing.Self:
        r"""Shift random-walk coordinates by a constant offset.

        Defines

        $$
        x'_t = x_t - c.
        $$

        The initial mean is shifted by $-c$ while the innovation
        distribution is unchanged.
        """
        offset = jnp.asarray(offset)

        expected_shape = self.batch_shape + (self.variable_dim,)

        if offset.shape != expected_shape:
            raise ValueError(
                f'offset must have shape {expected_shape}, got {offset.shape}'
            )

        return self._replace(
            initial=self.initial._replace(
                mean=self.initial.mean - offset,
            ),
        )

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return self._replace(
            initial=self.initial.select(index),
            innovation=self.innovation.select(index),
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at axis."""
        return self._replace(
            initial=self.initial.broadcast(shape, axis=axis),
            innovation=self.innovation.broadcast(shape, axis=axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return self._replace(
            initial=self.initial.squeeze(axis=axis),
            innovation=self.innovation.squeeze(axis=axis),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return self._replace(
            initial=self.initial.permute(permutation, axis=axis),
            innovation=self.innovation.permute(permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return self._replace(
            initial=self.initial.move_axis(source, destination),
            innovation=self.innovation.move_axis(source, destination),
        )


class GatedGaussianRandomWalk(typing.NamedTuple):
    r"""Gate-controlled Gaussian random walks with additional batch dimensions `*B`.

    Each Gaussian has batch shape `(K, *B)` and event dimension `D`. The first
    batch axis indexes the `K` gates; `*B` independently replicates the
    complete gated process. Samples have shape `(K, *B, T, D)`.

    For each gate $k$,

    $$
    x_{0,k}
    \sim
    \mathcal{N}(\mu_{0,k}, Q_{0,k}),
    $$

    and each gate $g_t$ controls the increment

    $$
    x_{t+1,k}
    =
    x_{t,k}
    +
    \epsilon_{t,k},
    $$

    with

    $$
    \epsilon_{t,k}
    \sim
    \begin{cases}
    \mathcal{N}(b_{\mathrm{on},k}, Q_{\mathrm{on},k}),
        & g_t = k, \\
    \mathcal{N}(b_{\mathrm{off},k}, Q_{\mathrm{off},k}),
        & g_t \neq k.
    \end{cases}
    $$
    """

    # Gaussian batch shape: (K, *B), event dimension: D.
    initial: Gaussian
    active_innovation: Gaussian
    inactive_innovation: Gaussian

    @classmethod
    def from_covariances(
        cls,
        *,
        initial_covariance: jax.Array,
        active_covariance: jax.Array,
        inactive_covariance: jax.Array,
    ) -> typing.Self:
        """Construct a zero-mean gated walk from `(K, *B, D, D)` covariances."""
        return cls(
            initial=Gaussian(
                mean=_zero_mean_from_covariance(initial_covariance),
                covariance=initial_covariance,
            ),
            active_innovation=Gaussian(
                mean=_zero_mean_from_covariance(active_covariance),
                covariance=active_covariance,
            ),
            inactive_innovation=Gaussian(
                mean=_zero_mean_from_covariance(inactive_covariance),
                covariance=inactive_covariance,
            ),
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Complete batch shape, including the gate-indexed axis when present."""
        shape = self.initial.batch_shape
        if (
            self.active_innovation.batch_shape != shape
            or self.inactive_innovation.batch_shape != shape
        ):
            raise ValueError('initial and innovation batch shapes must match')
        return shape

    @property
    def variable_dim(self) -> int:
        """Dimension `D` of each random-walk variable."""
        variable_dim = self.initial.variable_dim

        if self.active_innovation.variable_dim != variable_dim:
            raise ValueError('active innovation variable dimension must match initial')

        if self.inactive_innovation.variable_dim != variable_dim:
            raise ValueError(
                'inactive innovation variable dimension must match initial'
            )

        return variable_dim

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        """Common parameter dtype."""
        return jnp.result_type(
            self.initial.dtype,
            self.active_innovation.dtype,
            self.inactive_innovation.dtype,
        )

    def active_transition_dist(
        self,
    ) -> LinearGaussian:
        """Construct active transition distributions with batch `(K, *B)`."""
        _ = self.batch_shape
        _ = self.variable_dim

        return _random_walk_transition(self.active_innovation)

    def inactive_transition_dist(
        self,
    ) -> LinearGaussian:
        """Construct inactive transition distributions with batch `(K, *B)`."""
        _ = self.batch_shape
        _ = self.variable_dim

        return _random_walk_transition(self.inactive_innovation)

    def sample(self, key: jax.Array, num_steps: int, gates: jax.Array) -> jax.Array:
        """Sample `(K, *B, T, D)` values with gates shaped `(*B, T - 1)`.

        The first batch axis indexes gates; the remaining batch axes replicate
        the gated process. Gate labels refer to the current first-axis ordering.
        """
        if num_steps < 0:
            raise ValueError('num_steps must be non-negative')
        num_gates = self.num_gates
        variable_dim = self.variable_dim
        expected_gate_shape = self.batch_shape[1:] + (max(num_steps - 1, 0),)
        if gates.shape != expected_gate_shape:
            raise ValueError(
                f'gates must have shape {expected_gate_shape}, got {gates.shape}'
            )
        if num_steps == 0:
            return jnp.zeros(self.batch_shape + (0, variable_dim), dtype=self.dtype)
        key_initial, key_innovation = jax.random.split(key)
        initial = self.initial.sample(key_initial)[..., None, :]
        if num_steps == 1:
            return initial
        gate_indices = jnp.arange(num_gates).reshape(
            (num_gates,) + (1,) * len(self.batch_shape)
        )
        active = gates[None, ...] == gate_indices
        innovation = Gaussian(
            mean=jnp.where(
                active[..., None],
                self.active_innovation.mean[..., None, :],
                self.inactive_innovation.mean[..., None, :],
            ),
            covariance=jnp.where(
                active[..., None, None],
                self.active_innovation.covariance[..., None, :, :],
                self.inactive_innovation.covariance[..., None, :, :],
            ),
        )
        increments = innovation.sample(key_innovation)
        subsequent = initial + jnp.cumsum(increments, axis=-2)
        return jnp.concatenate((initial, subsequent), axis=-2)

    def reorient_variables(
        self,
        signs: jax.Array,
    ) -> typing.Self:
        """Reorient random-walk variable dimensions."""
        return self._replace(
            initial=self.initial.reorient_variables(signs),
            active_innovation=(self.active_innovation.reorient_variables(signs)),
            inactive_innovation=(self.inactive_innovation.reorient_variables(signs)),
        )

    def permute_variables(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel random-walk variable dimensions."""
        return self._replace(
            initial=self.initial.permute_variables(permutation),
            active_innovation=(self.active_innovation.permute_variables(permutation)),
            inactive_innovation=(
                self.inactive_innovation.permute_variables(permutation)
            ),
        )

    def shift_variables(
        self,
        offset: jax.Array,
    ) -> typing.Self:
        r"""Shift gated random-walk coordinates by a constant offset.

        Defines

        $$
        x'_{t,k} = x_{t,k} - c_k.
        $$

        Initial means are shifted by $-c_k$ while active and inactive
        innovation distributions are unchanged.
        """
        offset = jnp.asarray(offset)

        if offset.shape != self.initial.mean.shape:
            raise ValueError(
                f'offset must have shape {self.initial.mean.shape}, got {offset.shape}'
            )

        return self._replace(
            initial=self.initial._replace(
                mean=self.initial.mean - offset,
            ),
        )

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return self._replace(
            initial=self.initial.select(index),
            active_innovation=self.active_innovation.select(index),
            inactive_innovation=self.inactive_innovation.select(index),
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at axis."""
        return self._replace(
            initial=self.initial.broadcast(shape, axis=axis),
            active_innovation=self.active_innovation.broadcast(shape, axis=axis),
            inactive_innovation=self.inactive_innovation.broadcast(shape, axis=axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return self._replace(
            initial=self.initial.squeeze(axis=axis),
            active_innovation=self.active_innovation.squeeze(axis=axis),
            inactive_innovation=self.inactive_innovation.squeeze(axis=axis),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return self._replace(
            initial=self.initial.permute(permutation, axis=axis),
            active_innovation=self.active_innovation.permute(permutation, axis=axis),
            inactive_innovation=self.inactive_innovation.permute(
                permutation, axis=axis
            ),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return self._replace(
            initial=self.initial.move_axis(source, destination),
            active_innovation=self.active_innovation.move_axis(source, destination),
            inactive_innovation=self.inactive_innovation.move_axis(source, destination),
        )

    @property
    def num_gates(self) -> int:
        """Number of gates, interpreted along the first batch axis for sampling."""
        if not self.batch_shape or self.batch_shape[0] < 1:
            raise ValueError('gated sampling requires a nonempty first batch axis')
        return self.batch_shape[0]

    def permute_gates(self, permutation: jax.Array) -> typing.Self:
        """Relabel gates along the first batch axis; gate inputs use the new labels."""
        _ = self.num_gates
        return self.permute(permutation, axis=0)
