import jax.numpy as jnp
import matplotlib.colors
import matplotlib.patches
import numpy as np
from matplotlib import pyplot as plt

from xxm.core.dists.gaussian import Gaussian, LinearGaussian

STATES_CMAP = matplotlib.colors.ListedColormap(['tab:blue', 'tab:orange', 'tab:green'])


def plot_latent_2d(ax, latents, color='k', **kwargs):
    ax.plot(latents[:, 0], latents[:, 1], color=color, **kwargs)
    ax.scatter(latents[0, 0], latents[0, 1], color='g', zorder=10)
    ax.scatter(latents[-1, 0], latents[-1, 1], color='r', zorder=10)
    ax.set(title='latent states', aspect='equal')


def plot_sequence_2d(
    observations: jnp.ndarray,
    states: jnp.ndarray,
    ax=None,
) -> None:
    observations = jnp.asarray(observations)
    states = jnp.asarray(states)

    if ax is None:
        _f, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    ax.plot(
        observations[:, 0],
        observations[:, 1],
        linewidth=0.5,
        alpha=0.3,
    )

    ax.scatter(
        observations[:, 0],
        observations[:, 1],
        c=states,
        s=12,
        cmap=STATES_CMAP,
    )

    ax.set_xlabel('Observation 1')
    ax.set_ylabel('Observation 2')
    ax.set_aspect('equal')


def plot_linear_dynamics(
    ax,
    affine,
    xlim=(-3, 3),
    ylim=(-3, 3),
    num_points=15,
    **kwargs,
):
    """Plot the expected displacement field of 2D linear dynamics."""

    # there's only one lag, so squeeze to get the 2D dynamics.
    if affine.input_ndim == 2 and affine.input_shape[0] == 1:
        affine = affine.input_squeeze()

    if affine.input_shape != (2,) or affine.output_dim != 2:
        raise ValueError(
            '2D dynamics require an affine map (2,) -> (2,), '
            f'got {affine.input_shape} -> ({affine.output_dim},)'
        )

    affine = affine.input_squeeze()

    matrix = affine.coefficients
    bias = affine.bias

    x = np.linspace(*xlim, num_points)  # type: ignore
    y = np.linspace(*ylim, num_points)  # type: ignore
    xx, yy = np.meshgrid(x, y)

    points = np.stack([xx, yy], axis=-1)  # (..., 2)

    next_points = points @ matrix.T + bias
    displacement = next_points - points

    ax.quiver(
        xx,
        yy,
        displacement[..., 0],
        displacement[..., 1],
        **kwargs,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')


def plot_states(ax, states, **kwargs):
    ax.imshow(
        states.reshape(-1, 1).T,
        aspect='auto',
        cmap=STATES_CMAP,
        vmin=0,
        vmax=2,
        alpha=0.25,
        extent=(0, len(states), 0, 1),
        transform=ax.get_xaxis_transform(),
        **kwargs,
    )


def plot_observations_im(ax, observations, ylabel, **kwargs):
    im = ax.imshow(
        observations.T,
        aspect='auto',
        interpolation='none',
        **kwargs,
    )

    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xlabel='time',
        ylabel=ylabel,
    )
    ax.set(title='observations')


def plot_sequence_1d(
    ax,
    states: jnp.ndarray,
    observations: jnp.ndarray,
) -> None:
    observations = jnp.asarray(observations)
    states = jnp.asarray(states)

    plot_states(ax, states)

    ax.plot(observations.T[0], color='k')
    ax.plot(observations.T[1], color='xkcd:magenta')

    ax.set(xlabel='Time step', ylabel='Observations')


def plot_fit_progress(objective: jnp.ndarray, name='Log Likelihood', ax=None) -> None:

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    ax.plot(np.asarray(objective))

    ax.set_xlabel('iteration')
    ax.set_ylabel(name)


def plot_fit_progress_many(fits, highlight_idx=None, name='Log Likelihood', ax=None):

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(fits.objective_traces.T)

    if highlight_idx is not None:
        _, log_likelihoods = fits.get(highlight_idx)
        ax.plot(log_likelihoods, color='k')

    ax.set(
        xlabel='iteration',
        ylabel=name,
    )


def plot_inferred_states_comparison(
    true_states, observations, inferred_states, reconstruction
):
    _f, axs = plt.subplots(nrows=2, constrained_layout=True, figsize=(8, 6))

    ax = axs[0]
    plot_sequence_1d(ax, true_states, observations)

    ax.set(
        title='True states',
    )

    ax = axs[1]
    plot_sequence_1d(ax, inferred_states, reconstruction)
    ax.set(
        title='Inferred states',
    )


def plot_sequence_2d_comparison(
    true_states,
    observations,
    inferred_states,
    reconstruction,
):

    _, axs = plt.subplots(ncols=2, sharex='all', sharey='all', figsize=(8, 4))
    plot_sequence_2d(observations, true_states, ax=axs[0])
    plot_sequence_2d(reconstruction, inferred_states, ax=axs[1])


def plot_conditional_linear_dynamics(
    dyn: LinearGaussian,
):
    """Plot the expected displacement field of a 2D linear dynamics model."""

    _, axs = plt.subplots(
        figsize=(8, 6),
        ncols=dyn.batch_shape[0],
        sharex='all',
        sharey='all',
        constrained_layout=True,
    )

    for state, ax in enumerate(axs):
        plot_linear_dynamics(
            ax,
            dyn.select(state).affine,
            color='k',
        )

        ax.set_title(f'State {state}')


def plot_conditional_linear_dynamics_comparison(
    d0: LinearGaussian,
    d1: LinearGaussian,
) -> None:
    """Plot the expected displacement field of multiple 2D linear dynamics model."""

    assert d0.batch_shape == d1.batch_shape, (
        'Models must have the same number of states.'
    )

    _, axs = plt.subplots(
        figsize=(8, 6),
        ncols=d0.batch_shape[0],
        sharex='all',
        sharey='all',
        constrained_layout=True,
    )

    for state, ax in enumerate(axs):
        plot_linear_dynamics(
            ax,
            d0.select(state).affine,
            color='k',
        )

        plot_linear_dynamics(
            ax,
            d1.select(state).affine,
            color='xkcd:magenta',
        )

        ax.set_title(f'State {state}')


def plot_gaussian_2d_ellipse(
    ax,
    gaussian: Gaussian,
    *,
    linestyle: str,
    label: str | None = None,
    color='k',
    **kwargs,
) -> None:

    assert gaussian.variable_dim == 2, 'Only 2D Gaussians are supported.'

    eigenvalues, eigenvectors = np.linalg.eigh(gaussian.covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(
        np.arctan2(
            eigenvectors[1, 0],
            eigenvectors[0, 0],
        )
    )

    ellipse = matplotlib.patches.Ellipse(
        tuple(gaussian.mean),
        width=2 * np.sqrt(eigenvalues[0]),
        height=2 * np.sqrt(eigenvalues[1]),
        angle=angle,
        fill=False,
        linestyle=linestyle,
        linewidth=2,
        label=label,
        color=color,
        **kwargs,
    )

    ax.add_patch(ellipse)

    ax.scatter(*gaussian.mean, marker='o', s=40, color=color)
