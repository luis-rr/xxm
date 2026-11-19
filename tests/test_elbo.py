import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xxm
from xxm import arhmm, hmm, lds
from xxm.arhmm.data import ARDatasetCollection
from xxm.core.data import DatasetCollection
from xxm.hmm.model import GaussianHMMCollection

from .test_hdf5 import FACADE_NOISE, make_data, make_model

AR_FACADES = (xxm.GaussianARHMM, xxm.PoissonARHMM)


def data_for(facade):
    """AR-HMMs take no known inputs; the other test models take one."""
    return make_data(input_dim=0 if facade in AR_FACADES else 1)


def as_float32(model):
    """The model with all its parameters converted to float32."""
    params = jax.tree.map(lambda value: value.astype(jnp.float32), model.to_params())
    return type(model).from_params(params)


def infer(model, data):
    if isinstance(model, (xxm.GaussianSLDS, xxm.PoissonSLDS)):
        return model.infer(data, num_iters=2)
    return model.infer(data)


@pytest.mark.parametrize(('facade', 'noise'), FACADE_NOISE)
def test_elbo_is_finite_scalar(facade, noise):
    elbo = infer(make_model(facade, noise=noise), data_for(facade)).elbo()
    assert elbo.shape == ()
    assert jnp.isfinite(elbo)


@pytest.mark.parametrize(
    'facade', [xxm.GaussianHMM, xxm.GaussianARHMM, xxm.GaussianLDS]
)
@pytest.mark.parametrize('noise', ['full', 'diagonal'])
def test_exact_elbo_is_log_likelihood(facade, noise):
    model, data = make_model(facade, noise=noise), data_for(facade)
    match model:
        case xxm.GaussianHMM():
            objective = hmm.inference.infer_exact(model.core, data).objective
            elbo = model.infer(data).elbo()
        case xxm.GaussianARHMM():
            prepared = ARDatasetCollection.from_datasets(data, model.num_lags)
            objective = arhmm.inference.infer_exact(model.core, prepared).objective
            elbo = model.infer(data).elbo()
        case xxm.GaussianLDS():
            objective = lds.inference.infer_exact(model.core, data).objective
            elbo = model.infer(data).elbo()
        case _:
            raise TypeError(f'Unexpected test model: {type(model).__name__}')
    np.testing.assert_allclose(elbo, objective, rtol=1e-4)


@pytest.mark.parametrize(('facade', 'noise'), FACADE_NOISE)
def test_elbo_sums_sequences(facade, noise):
    """Every ELBO term is per sequence: duplicating a sequence doubles the ELBO."""
    model = make_model(facade, sessions=1, noise=noise)
    sequence = data_for(facade).datasets[0].select(0)
    single = DatasetCollection.from_datasets((sequence,))
    duplicated = DatasetCollection.from_datasets((sequence.broadcast((2,)),))
    np.testing.assert_allclose(
        infer(model, duplicated).elbo(), 2 * infer(model, single).elbo(), rtol=1e-5
    )


@pytest.mark.parametrize('noise', ['full', 'diagonal'])
def test_variational_em_does_not_decrease_elbo(noise):
    """E-steps continue from the previous posterior, so EM ascends the ELBO.

    One inference sweep per M-step suffices for the guarantee. The tolerance
    covers float32 and the M-step ridge and floors.
    """
    # the test models mix in float16 parameters; fitting needs one dtype
    model = as_float32(make_model(xxm.GaussianSLDS, noise=noise))
    assert isinstance(model, xxm.GaussianSLDS)
    fit = model.fit(
        data_for(xxm.GaussianSLDS),
        num_iters=10,
        num_inference_iters=1,
        progress=False,
    )
    trace = np.asarray(fit.objective_trace)
    assert np.all(np.diff(trace) >= -1e-5 * np.abs(trace).max())


def test_collection_elbo_sums_sessions():
    model = make_model(xxm.GaussianHMM, sessions=1)
    assert isinstance(model, xxm.GaussianHMM)
    session = DatasetCollection((make_data().datasets[0],))
    data = DatasetCollection((session.datasets[0], session.datasets[0]))
    collection = GaussianHMMCollection(models=(model, model))
    np.testing.assert_allclose(
        collection.infer(data).elbo(), 2 * model.infer(session).elbo(), rtol=1e-5
    )
