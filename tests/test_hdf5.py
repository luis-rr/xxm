"""Tiny, deterministic storage fixtures; no initialization or inference required."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xxm
from xxm import arhmm, hmm, lds, slds
from xxm.core.data import Dataset, DatasetCollection, Sequences
from xxm.core.dists.discrete import DiscreteChainMarginals
from xxm.core.dists.gaussian import GaussianChainMarginals
from xxm.core.io import hdf5
from xxm.core.lengths import Lengths
from xxm.core.mask import ArbitraryMask
from xxm.hmm.io import HMMParams
from xxm.lds.io import LDSParams
from xxm.slds.io import SLDSParams

FACADES = (
    xxm.GaussianHMM,
    xxm.PoissonHMM,
    xxm.GaussianARHMM,
    xxm.PoissonARHMM,
    xxm.GaussianLDS,
    xxm.PoissonLDS,
    xxm.GaussianSLDS,
    xxm.PoissonSLDS,
)

GAUSSIAN_FACADES = (
    xxm.GaussianHMM,
    xxm.GaussianARHMM,
    xxm.GaussianLDS,
    xxm.GaussianSLDS,
)

# Gaussian facades run with both observation-noise structures.
FACADE_NOISE = [
    (facade, noise)
    for facade in FACADES
    for noise in (('full', 'diagonal') if facade in GAUSSIAN_FACADES else ('full',))
]


def assert_tree_equal(left, right):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for a, b in zip(left_leaves, right_leaves, strict=True):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def make_model(facade, sessions=2, input_dim=1, noise='full'):
    # Explicit dtypes keep promotion independent of JAX's weak scalar typing.
    gaussian = facade in GAUSSIAN_FACADES
    diagonal = noise == 'diagonal'
    discrete = facade in (xxm.GaussianHMM, xxm.PoissonHMM)
    ar = facade in (xxm.GaussianARHMM, xxm.PoissonARHMM)
    switching = facade in (xxm.GaussianSLDS, xxm.PoissonSLDS)
    emissions = []
    for dim in range(1, sessions + 1):
        if discrete:
            item = (
                (
                    {
                        'mean': jnp.full((2, dim), -0.25 * dim, dtype=jnp.float32),
                        'variances': jnp.ones((2, dim), dtype=jnp.float32),
                    }
                    if diagonal
                    else {
                        'mean': jnp.full((2, dim), -0.25 * dim, dtype=jnp.float32),
                        'covariance': jnp.tile(
                            jnp.eye(dim, dtype=jnp.float32)[None], (2, 1, 1)
                        ),
                    }
                )
                if gaussian
                else {'log_rates': jnp.full((2, dim), -0.5 * dim, dtype=jnp.float32)}
            )
        else:
            shape = (2, dim, 2, dim) if ar else (dim, 1)
            item = {
                'coefficients': jnp.full(shape, 0.25 * dim, dtype=jnp.float32),
                'bias': jnp.full(
                    (2, dim) if ar else (dim,), -0.5 * dim, dtype=jnp.float16
                ),
            }
            if gaussian and diagonal:
                item['variances'] = jnp.ones(
                    (2, dim) if ar else (dim,), dtype=jnp.float32
                )
            elif gaussian:
                item['covariance'] = (
                    jnp.tile(jnp.eye(dim, dtype=jnp.float32)[None], (2, 1, 1))
                    if ar
                    else jnp.eye(dim, dtype=jnp.float32)
                )
        emissions.append(item)
    categorical = {
        'initial_probs': jnp.array([0.3, 0.7], dtype=jnp.float32),
        'transition_probs': jnp.array([[0.75, 0.25], [0.5, 0.5]], dtype=jnp.float32),
    }
    if discrete or ar:
        return facade.from_params(HMMParams(**categorical, emissions=tuple(emissions)))
    prefix = (2,) if switching else ()
    dynamics = {
        'dynamics_coefficients': jnp.full((*prefix, 1, 1), 0.75, dtype=jnp.float32),
        'dynamics_bias': jnp.full((*prefix, 1), -0.125, dtype=jnp.float32),
        'dynamics_input_coefficients': jnp.full(
            (*prefix, 1, input_dim), 0.5, dtype=jnp.float32
        ),
    }
    if switching:
        return facade.from_params(
            SLDSParams(
                **categorical,
                **dynamics,
                latent_initial_means=jnp.array([[-0.25], [0.5]], dtype=jnp.float32),
                latent_initial_covariances=jnp.array(
                    [[[0.5]], [[1.5]]], dtype=jnp.float32
                ),
                dynamics_covariances=jnp.array([[[0.75]], [[1.25]]], dtype=jnp.float32),
                emissions=tuple(emissions),
            )
        )
    return facade.from_params(
        LDSParams(
            **dynamics,
            initial_mean=jnp.array([-0.25], dtype=jnp.float32),
            initial_covariance=jnp.array([[0.5]], dtype=jnp.float32),
            dynamics_covariance=jnp.array([[0.75]], dtype=jnp.float32),
            emissions=tuple(emissions),
        )
    )


@pytest.mark.parametrize(('facade', 'noise'), FACADE_NOISE)
@pytest.mark.parametrize('sessions', [1, 2])
def test_model_roundtrips(tmp_path, facade, noise, sessions):
    model = make_model(facade, sessions=sessions, input_dim=sessions - 1, noise=noise)
    params = model.to_params()
    restored = facade.from_params(params)
    assert type(restored) is facade
    assert_tree_equal(model.core, restored.core)
    assert_tree_equal(params, restored.to_params())
    path = tmp_path / 'model.h5'
    model.to_hdf5(path)
    restored = facade.from_hdf5(path)
    assert type(restored) is facade
    assert_tree_equal(model.core, restored.core)
    assert_tree_equal(params, restored.to_params())


def make_data(input_dim=1):
    datasets = []
    for dim, sizes in ((1, [4, 3]), (2, [3])):
        lengths = Lengths(jnp.array(sizes, dtype=jnp.int32))
        count = len(sizes)
        # Nonzero padding is intentional: storage must not clear or repad it.
        observations = jnp.arange(count * 4 * dim, dtype=jnp.float32).reshape(
            count, 4, dim
        )
        inputs = jnp.full((count, 4, input_dim), 0.25, dtype=jnp.float32)
        visible = jnp.array([[True, False, True, True], [True, True, False, False]])[
            :count
        ]
        if dim == 2:
            visible = jnp.array([[True, False, True, False]])
        datasets.append(
            Dataset(
                Sequences(observations, lengths),
                Sequences(inputs, lengths),
                ArbitraryMask(visible),
            )
        )
    return DatasetCollection(tuple(datasets))


def test_dataset_roundtrip(tmp_path):
    data = make_data()
    path = tmp_path / 'data.h5'
    data.to_hdf5(path)
    restored = xxm.DatasetCollection.from_hdf5(path)
    assert_tree_equal(data, restored)


def make_posterior(data, *, num_lags=0):
    lengths = Lengths(
        jnp.concatenate(tuple(d.lengths.values for d in data.datasets)) - num_lags
    )
    steps, count = data.max_steps - num_lags, data.partition.total
    discrete = DiscreteChainMarginals(
        jnp.full((count, steps, 2), 0.5, dtype=jnp.float32),
        jnp.full((count, steps - 1, 2, 2), 0.25, dtype=jnp.float32),
        lengths,
    )
    gaussian = GaussianChainMarginals(
        jnp.full((count, steps, 1), -0.25, dtype=jnp.float32),
        jnp.ones((count, steps, 1, 1), dtype=jnp.float32),
        jnp.full((count, steps - 1, 1, 1), 0.125, dtype=jnp.float32),
        lengths,
    )
    return discrete, gaussian


@pytest.mark.parametrize('family', [hmm, arhmm, lds, slds])
def test_posterior_roundtrip(tmp_path, family):
    discrete, gaussian = make_posterior(make_data())
    posterior = (
        slds.Posterior(discrete, gaussian)
        if family is slds
        else gaussian
        if family is lds
        else discrete
    )
    path = tmp_path / 'posterior.h5'
    family.save_posterior(posterior, path)
    assert_tree_equal(posterior, family.load_posterior(path))


@pytest.mark.parametrize(('facade', 'noise'), FACADE_NOISE)
def test_inferred_roundtrip(tmp_path, facade, noise):
    model, data = make_model(facade, noise=noise), make_data()
    discrete, gaussian = make_posterior(
        data,
        num_lags=2 if isinstance(model, (xxm.GaussianARHMM, xxm.PoissonARHMM)) else 0,
    )
    match model:
        case xxm.GaussianHMM():
            inferred = hmm.Inferred(model, discrete, data)
        case xxm.PoissonHMM():
            inferred = hmm.Inferred(model, discrete, data)
        case xxm.GaussianARHMM():
            inferred = arhmm.Inferred(model, discrete, data)
        case xxm.PoissonARHMM():
            inferred = arhmm.Inferred(model, discrete, data)
        case xxm.GaussianLDS():
            inferred = lds.Inferred(model, gaussian, data)
        case xxm.PoissonLDS():
            inferred = lds.Inferred(model, gaussian, data)
        case xxm.GaussianSLDS():
            inferred = slds.Inferred(model, slds.Posterior(discrete, gaussian), data)
        case xxm.PoissonSLDS():
            inferred = slds.Inferred(model, slds.Posterior(discrete, gaussian), data)
        case _:
            raise TypeError(f'Unexpected test model: {type(model).__name__}')
    posterior = inferred.posterior
    path = tmp_path / 'inferred.h5'
    inferred.to_hdf5(path)
    restored = type(inferred).from_hdf5(path)
    assert type(restored.model) is facade
    assert_tree_equal(model.core, restored.model.core)
    assert_tree_equal(posterior, restored.posterior)
    assert_tree_equal(data, restored.data)
    assert_tree_equal(inferred.observation_mean(1), restored.observation_mean(1))


def test_overwrite(tmp_path):
    model = make_model(xxm.GaussianLDS)
    path = tmp_path / 'model.h5'
    model.to_hdf5(path)
    with pytest.raises(FileExistsError):
        model.to_hdf5(path)
    model.to_hdf5(path, overwrite=True)
    assert_tree_equal(model.core, xxm.GaussianLDS.from_hdf5(path).core)


def test_array_storage(tmp_path):
    path = tmp_path / 'arrays.h5'
    arrays = {
        'scalar': np.array(7, dtype=np.int16),
        'empty': np.empty((2, 0), dtype=np.float32),
        'unsigned': np.array([255], dtype=np.uint8),
        'bool': np.array([True, False]),
        'complex': np.array([1 + 2j], dtype=np.complex64),
    }
    hdf5.save(path, arrays, object_type='test')
    restored = hdf5.load(path)
    assert_tree_equal(arrays, restored)
