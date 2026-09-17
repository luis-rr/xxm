"""Structural plotting checks; skipped when the plotting extra is absent."""

# Optional-dependency skipping must precede imports of the plotting modules.

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip('matplotlib')

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.quiver import Quiver
from matplotlib.streamplot import StreamplotSet

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian
from xxm.plot import core, models


@pytest.fixture
def ax():
    figure = Figure()
    FigureCanvasAgg(figure)
    return figure.subplots()


@pytest.fixture
def figure_backend():
    with plt.rc_context({'backend': 'Agg'}):
        yield
        plt.close('all')


def test_sequences_and_image_return(ax):
    traces = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]])
    lines = core.plot_traces_1d(ax, traces)
    assert len(lines) == 2
    np.testing.assert_array_equal(lines[1].get_ydata(), traces[:, 1])
    limits = ax.get_ylim()
    core.plot_state_1d(ax, [0, 1, 0])
    assert ax.get_ylim() == limits
    image = core.plot_traces_image(ax, traces)
    assert isinstance(image, AxesImage)
    np.testing.assert_array_equal(image.get_array(), traces.T)
    assert len(ax.figure.axes) == 1


@pytest.mark.parametrize('states', [None, [0, 1, 0]])
def test_sequence_2d(ax, states):
    core.plot_seq_2d(ax, states, np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]))
    assert len(ax.lines) == 1


def test_shape_validation(ax):
    with pytest.raises(ValueError, match='shape'):
        core.plot_traces_2d(ax, np.zeros((3, 4)))
    with pytest.raises(ValueError, match='shape'):
        core.plot_seq_1d(ax, [0, 1], np.zeros((3, 2)))
    with pytest.raises(ValueError, match='nonnegative integer'):
        core.plot_state_1d(ax, [0, -1])
    with pytest.raises(ValueError, match='unbatched affine'):
        core.plot_dyn_linear(ax, Affine(jnp.eye(3), jnp.zeros(3)))


def test_gaussian_geometry(ax):
    gaussian = Gaussian(jnp.array([1.0, 2.0]), jnp.diag(jnp.array([4.0, 1.0])))
    ellipse = core.plot_gaussian_2d(ax, gaussian, scale=2)
    np.testing.assert_allclose(ellipse.center, [1, 2])
    assert ellipse.width == 8
    assert ellipse.height == 4


def test_dynamics_displacement_and_artists(ax):
    identity = Affine(jnp.eye(2), jnp.zeros(2))
    translated = Affine(jnp.eye(2), jnp.array([3.0, 4.0]))
    quiver = core.plot_dyn_linear_quiver(ax, translated, num_points=3)
    assert isinstance(quiver, Quiver)
    np.testing.assert_allclose(np.asarray(quiver.U), 3)
    np.testing.assert_allclose(np.asarray(quiver.V), 4)
    stream = core.plot_dyn_linear_stream(ax, translated, num_points=3)
    assert isinstance(stream, StreamplotSet)
    mesh = core.plot_dyn_mismatch(ax, identity, translated, num_points=3)
    magnitudes = mesh.get_array()
    assert magnitudes is not None
    np.testing.assert_allclose(magnitudes, 5)


def test_comparison_shared_normalization(figure_backend):
    norm = Normalize()
    axs = models.plot_traces_image_comparison(
        np.zeros((3, 2)), np.full((3, 2), 10), norm=norm
    )
    assert axs.shape == (2,)
    assert axs[0].images[0].norm is axs[1].images[0].norm
    assert axs[0].images[0].get_clim() == (0, 10)
    assert norm.vmin is None and norm.vmax is None
    assert len(axs[0].figure.axes) == 3
    comparison = models.plot_seq_1d_comparison(
        [0, 0], np.zeros((2, 1)), [0, 2], np.ones((2, 1)), state_kwargs=None
    )
    assert comparison.shape == (2,)
    np.testing.assert_array_equal(
        comparison[0].images[0].norm.boundaries,
        comparison[1].images[0].norm.boundaries,
    )


def test_dynamics_grids(figure_backend):
    identity = Affine(jnp.eye(2), jnp.zeros(2)).broadcast((1, 2))
    translated = identity._replace(bias=jnp.array([[[3.0, 4.0], [0.0, 10.0]]]))
    assert models.plot_dyn_grid(identity, num_points=3).shape == (1, 2)
    norm = Normalize()
    axs = models.plot_dyn_mismatch_grid(
        identity, translated, num_points=3, mesh_kwargs={'norm': norm}
    )
    assert axs.shape == (1, 2)
    left, right = [ax.collections[0] for ax in axs[0]]
    assert left.norm is right.norm
    assert left.get_clim() == (5, 10)
    assert norm.vmin is None and norm.vmax is None
    assert len(axs[0, 0].figure.axes) == 3


def test_fit_progress(figure_backend):
    assert models.plot_fit_progress([1.0]).shape == (2,)
    axs = models.plot_fit_progress_many([[1.0, 3.0, 4.0]], highlight_idx=0)
    assert axs.shape == (2,)
    np.testing.assert_array_equal(axs[1].lines[0].get_ydata(), [2, 1])
