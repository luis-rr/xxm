"""Continuous model components: initial distribution and linear dynamics."""

import typing

import jax
from jax import numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianPotential
from xxm.core.dists.gaussian import Gaussian, LinearGaussian, PairedGaussian
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim.batch import filter_valid_batches
from xxm.core.posteriors import ContinuousPosterior


class GaussianInitial(typing.NamedTuple):
    r"""Initial distribution $p(x_0)$ for continuous latent state."""

    dist: Gaussian  # structural batch *B

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.dist.batch_shape

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    def fit_params(
        self,
        posterior: ContinuousPosterior,
        weights: jax.Array,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        r"""
        Fit initial-state parameters while preserving the receiver batch prefix.

        ``self.batch_shape`` is preserved. Any additional posterior batch axes
        are independent replicates and are pooled when fitting the initial
        distribution. ``weights`` contains one validity weight per latent state.
        """
        posterior_batch = posterior.batch_shape
        replicate_shape = batch.split_prefix(
            posterior_batch, self.batch_shape, name='posterior batch shape'
        )

        if weights.shape != posterior.means.shape[:-1]:
            raise ValueError('weights must match posterior batch and time dimensions')

        initial_marginals, initial_weights = batch.pool_samples(
            (
                Gaussian(
                    mean=posterior.means[..., 0, :],
                    covariance=posterior.covariances[..., 0, :, :],
                ),
                weights[..., 0],
            ),
            self.batch_shape,
            replicate_shape,
        )
        initial = gaussian_fit.from_moment_match(
            initial_marginals, weights=initial_weights
        )

        reference_marginals, reference_weights = batch.pool_samples(
            (
                Gaussian(
                    mean=posterior.means,
                    covariance=posterior.covariances,
                ),
                weights,
            ),
            self.batch_shape,
            (*replicate_shape, posterior.num_steps),
        )
        reference = gaussian_fit.from_moment_match(
            reference_marginals, weights=reference_weights
        )

        covariance = gaussian_fit.add_covariance_floor(
            initial.covariance,
            reference_covariance=reference.covariance,
            covariance_floor=covariance_floor,
        )

        fitted = initial._replace(covariance=covariance)
        return self._replace(
            dist=filter_valid_batches(fitted, self.dist, initial_weights)
        )

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample initial latent state."""
        return self.dist.sample(key)

    def align(self, alignment: Affine) -> typing.Self:
        r"""
        Express the initial distribution in coordinates $x' = f(x)$ defined by `alignment`.
        """
        batch.require_same_shape(
            alignment.batch_shape,
            self.batch_shape,
        )

        return self._replace(
            dist=self.dist.affine(alignment),
        )

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Estimate an initial Gaussian from a known latent trajectory."""

        batch_shape = latents.shape[:-2]
        reference = batch.vmap_batch(
            lambda values: gaussian_fit.from_samples(
                values, covariance_floor=covariance_floor
            ),
            latents,
            batch_shape=batch_shape,
        )

        return cls(
            Gaussian(
                mean=latents[..., 0, :],
                covariance=reference.covariance,
            )
        )


class GaussianLinearDynamics(typing.NamedTuple):
    r"""
    Linear-Gaussian latent dynamics.

    $$x_t \mid x_{t-1}, u_t
    \sim
    \mathcal{N}(A x_{t-1} + B u_t + b, Q).$$

    The underlying `LinearGaussian` acts on the concatenated predictor
    `[x_{t-1}, u_t]`, with coefficients `[A, B]`.
    """

    dist: LinearGaussian

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.dist.batch_shape

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    @property
    def latent_dim(self) -> int:
        """Dimension $D_x$ of the latent state."""
        return self.dist.output_dim

    @property
    def input_dim(self) -> int:
        """Dimension $D_u$ of the known dynamics input."""
        return self.dist.input_size - self.latent_dim

    @property
    def latent_coefficients(self) -> jax.Array:
        """Latent transition coefficients $A$."""
        return self.dist.affine.coefficients_flat[..., :, : self.latent_dim]

    @property
    def input_coefficients(self) -> jax.Array:
        """Known-input coefficients $B$."""
        return self.dist.affine.coefficients_flat[..., :, self.latent_dim :]

    def conditional(
        self,
        inputs: jax.Array,
    ) -> LinearGaussian:
        r"""
        Condition the dynamics on known inputs.

        Returns

        $$x_t \mid x_{t-1}
        \sim
        \mathcal{N}(A x_{t-1} + B u_t + b, Q).$$

        Query dimensions in `inputs` become batch dimensions of the returned
        conditional distribution.
        """
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f'expected input dimension {self.input_dim}, got {inputs.shape[-1]}'
            )

        input_affine = Affine(
            coefficients=self.input_coefficients,
            bias=self.dist.affine.bias,
        )
        bias = input_affine.apply(inputs)

        coefficients = batch.align_array(
            self.latent_coefficients, self.batch_shape, inputs.shape[:-1]
        )
        covariance = batch.align_array(
            self.dist.covariance, self.batch_shape, inputs.shape[:-1]
        )

        return LinearGaussian(
            affine=Affine(
                coefficients=coefficients,
                bias=bias,
            ),
            covariance=covariance,
        )

    def fit_params(
        self,
        posterior: ContinuousPosterior,
        inputs: jax.Array,
        weights: jax.Array,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        """
        Fit dynamics while preserving the receiver batch prefix.

        ``self.batch_shape`` is preserved. Any additional posterior batch axes
        are pooled together with the transition axis. ``inputs`` and ``weights``
        contain one entry per transition.
        """
        posterior_batch = posterior.batch_shape
        replicate_shape = batch.split_prefix(
            posterior_batch, self.batch_shape, name='posterior batch shape'
        )

        num_transitions = posterior.num_steps - 1
        expected_inputs_shape = (
            *posterior_batch,
            num_transitions,
            self.input_dim,
        )
        expected_weights_shape = (
            *posterior_batch,
            num_transitions,
        )

        if inputs.shape != expected_inputs_shape:
            raise ValueError(
                f'inputs must have shape {expected_inputs_shape}; got {inputs.shape}'
            )

        if weights.shape != expected_weights_shape:
            raise ValueError(
                f'weights must have shape {expected_weights_shape}; got {weights.shape}'
            )

        inputs = jnp.where(
            weights[..., None] != 0,
            inputs,
            jnp.zeros(
                (),
                dtype=inputs.dtype,
            ),
        )

        sample_shape = (*replicate_shape, num_transitions)
        if 0 in sample_shape:
            return self

        paired, inputs, weights = batch.pool_samples(
            (posterior.paired_marginals(), inputs, weights),
            self.batch_shape,
            sample_shape,
        )

        def fit_one(
            paired_i: PairedGaussian,
            inputs_i: jax.Array,
            weights_i: jax.Array,
        ) -> LinearGaussian:
            return gaussian_fit.linear_from_paired_covariates(
                paired_i,
                inputs_i,
                weights=weights_i,
                ridge=ridge,
                covariance_floor=covariance_floor,
            )

        fitted = batch.vmap_batch(
            fit_one,
            paired,
            inputs,
            weights,
            batch_shape=self.batch_shape,
        )

        fitted = filter_valid_batches(fitted, self.dist, weights)

        return type(self)(fitted)

    def validate_trajectory_inputs(self, inputs: jax.Array, num_steps: int) -> None:
        """Require full trajectory inputs `(*B, T, D_u)` with at least one step.

        The boundary input at time zero is included but unused by dynamics.
        """
        if num_steps < 1:
            raise ValueError('trajectories require at least one time step')

        expected_shape = (*self.batch_shape, num_steps, self.input_dim)
        if inputs.shape != expected_shape:
            raise ValueError(
                f'inputs must have shape {expected_shape}; got {inputs.shape}'
            )

    def mean_trajectory(
        self,
        initial_mean: jax.Array,
        num_steps: int,
        *,
        inputs: jax.Array,
    ) -> jax.Array:
        """Roll out transition means, including the initial mean at time zero.

        Inputs have shape `(*B, T, D_u)`; the input at time zero is unused.
        """
        self.validate_trajectory_inputs(inputs, num_steps)
        batch_shape = self.batch_shape

        if initial_mean.shape != (*batch_shape, self.latent_dim):
            raise ValueError(
                'initial_mean must match dynamics batch and latent dimensions'
            )

        time_axis = len(batch_shape)
        transition_inputs = jnp.moveaxis(
            inputs[..., 1:, :],
            time_axis,
            0,
        )

        def step(
            previous: jax.Array,
            current_inputs: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            conditional = self.conditional(current_inputs)
            current = conditional.conditional_mean(previous)
            return current, current

        _, following = jax.lax.scan(
            step,
            initial_mean,
            transition_inputs,
        )

        following = jnp.moveaxis(
            following,
            0,
            time_axis,
        )

        return jnp.concatenate(
            [
                jnp.expand_dims(
                    initial_mean,
                    axis=time_axis,
                ),
                following,
            ],
            axis=time_axis,
        )

    def sample_next(
        self,
        key: jax.Array,
        previous: jax.Array,
        inputs: jax.Array,
    ) -> jax.Array:
        """Sample the next latent state conditional on the previous state and input."""
        predictor = jnp.concatenate(
            [
                previous,
                inputs,
            ],
            axis=-1,
        )

        return self.dist.sample(
            key,
            predictor,
        )

    def sample(
        self,
        key: jax.Array,
        initial_latent: jax.Array,
        num_steps: int,
        *,
        inputs: jax.Array,
    ) -> jax.Array:
        """
        Sample latent trajectories with shape `(*B, T, D_x)`.

        `inputs[..., t, :]` affects the transition into latent state `t`;
        `inputs[..., 0, :]` is unused.
        """
        self.validate_trajectory_inputs(inputs, num_steps)

        time_axis = len(self.batch_shape)

        transition_inputs = jnp.moveaxis(
            inputs[..., 1:, :],
            time_axis,
            0,
        )

        keys = jax.random.split(
            key,
            num_steps - 1,
        )

        def step(previous, args):
            step_key, step_inputs = args

            current = self.sample_next(
                step_key,
                previous,
                step_inputs,
            )

            return current, current

        _, following = jax.lax.scan(
            step,
            initial_latent,
            (
                keys,
                transition_inputs,
            ),
        )

        following = jnp.moveaxis(
            following,
            0,
            time_axis,
        )

        return jnp.concatenate(
            [
                jnp.expand_dims(
                    initial_latent,
                    axis=time_axis,
                ),
                following,
            ],
            axis=time_axis,
        )

    def align(self, alignment: Affine) -> typing.Self:
        """
        Express dynamics in coordinates $x' = P x + c$.

        External input coordinates are unchanged.
        """
        batch.require_same_shape(
            alignment.batch_shape,
            self.batch_shape,
        )

        latent_affine = Affine(
            coefficients=self.latent_coefficients,
            bias=self.dist.affine.bias,
        )

        aligned_latent = alignment.compose(latent_affine.compose(alignment.inverse()))

        input_coefficients = alignment.coefficients @ self.input_coefficients

        covariance = (
            alignment.coefficients
            @ self.dist.covariance
            @ jnp.swapaxes(
                alignment.coefficients,
                -1,
                -2,
            )
        )

        coefficients = jnp.concatenate(
            [
                aligned_latent.coefficients,
                input_coefficients,
            ],
            axis=-1,
        )

        return type(self)(
            LinearGaussian(
                affine=Affine(
                    coefficients=coefficients,
                    bias=aligned_latent.bias,
                ),
                covariance=covariance,
            )
        )

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        *,
        inputs: jax.Array,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """
        Fit linear dynamics to known latent trajectories.

        `inputs[..., t, :]` is paired with the transition from
        `latents[..., t - 1, :]` to `latents[..., t, :]`.
        """
        if inputs.shape[:-1] != latents.shape[:-1]:
            raise ValueError('inputs must align with latent batch and time dimensions')

        batch_shape = latents.shape[:-2]

        def fit_one(
            latents_i: jax.Array,
            inputs_i: jax.Array,
        ) -> LinearGaussian:
            predictors = jnp.concatenate(
                [
                    latents_i[:-1],
                    inputs_i[1:],
                ],
                axis=-1,
            )

            return gaussian_fit.linear_from_samples(
                predictors,
                latents_i[1:],
                ridge=ridge,
                covariance_floor=covariance_floor,
            )

        fitted = batch.vmap_batch(
            fit_one,
            latents,
            inputs,
            batch_shape=batch_shape,
        )

        return cls(fitted)


class StateConditionedGaussian(typing.NamedTuple):
    r"""
    Gaussian distribution conditioned on a discrete state.

    $$x \mid z=k \sim \mathcal{N}(\mu_k,\Sigma_k).$$
    """

    dist: Gaussian  # underlying batch (*B, K); K is the intrinsic state axis

    @property
    def batch_shape(self) -> tuple[int, ...]:
        assert self.dist.batch_shape
        return self.dist.batch_shape[:-1]

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        assert self.dist.batch_shape
        return self.dist.batch_shape[-1]

    def conditional(
        self,
        state: jax.Array,
    ) -> Gaussian:
        """Gaussian distribution conditional on the given state."""
        return batch.take_along_last_batch(self.dist, self.dist.batch_shape, state)

    def compute_potentials(self) -> GaussianPotential:
        """Return one Gaussian potential per discrete state."""
        return GaussianPotential.from_moments(
            self.dist,
        )

    def sample(
        self,
        key: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        """Sample conditional on the given discrete state."""
        return self.conditional(state).sample(key)

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """Relabel state-conditioned Gaussian distributions."""
        return self._replace(dist=self.dist.permute(permutation, axis=-1))

    def align(self, alignment: Affine) -> typing.Self:
        """Express the conditional distributions in coordinates $x' = f(x)$."""
        batch.require_same_shape(
            alignment.batch_shape,
            self.batch_shape,
        )

        aligned = alignment.broadcast(
            (self.num_states,),
            axis=len(self.batch_shape),
        )

        return self._replace(dist=self.dist.affine(aligned))
