import jax.numpy as jnp
import numpy as np
import pytest

import xxm
from xxm import arhmm, hmm, lds
from xxm.arhmm.data import ARDatasetCollection
from xxm.core.data import DatasetCollection

from .test_hdf5 import FACADES, make_data, make_model

AR_FACADES = (xxm.GaussianARHMM, xxm.PoissonARHMM)


def data_for(facade):
    """AR-HMMs take no known inputs; the other test models take one."""
    return make_data(input_dim=0 if facade in AR_FACADES else 1)


def infer(model, data):
    if isinstance(model, (xxm.GaussianSLDS, xxm.PoissonSLDS)):
        return model.infer(data, num_iters=2)
    return model.infer(data)


@pytest.mark.parametrize('facade', FACADES)
def test_elbo_is_finite_scalar(facade):
    elbo = infer(make_model(facade), data_for(facade)).elbo()
    assert elbo.shape == ()
    assert jnp.isfinite(elbo)


@pytest.mark.parametrize(
    'facade', [xxm.GaussianHMM, xxm.GaussianARHMM, xxm.GaussianLDS]
)
def test_exact_elbo_is_log_likelihood(facade):
    model, data = make_model(facade), data_for(facade)
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


def test_collection_elbo_sums_sessions():
    model = make_model(xxm.GaussianHMM, sessions=1)
    assert isinstance(model, xxm.GaussianHMM)
    session = DatasetCollection((make_data().datasets[0],))
    data = DatasetCollection((session.datasets[0], session.datasets[0]))
    collection = hmm.GaussianHMMCollection(models=(model, model))
    np.testing.assert_allclose(
        collection.infer(data).elbo(), 2 * model.infer(session).elbo(), rtol=1e-5
    )
