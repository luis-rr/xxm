import dataclasses

import jax
import numpy as np
import pytest
from jax import numpy as jnp

from xxm.arhmm.emissions import AREmissionsCollection
from xxm.arhmm.model import GaussianARHMM
from xxm.core.data import Dataset, DatasetCollection
from xxm.core.dists.gaussian import LinearGaussian, TemporalGaussian
from xxm.core.emissions.collection import (
    ContinuousEmissionsCollection,
    DiscreteEmissionsCollection,
)
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.hmm.model import GaussianHMM
from xxm.lds.model import GaussianLDS


def test_selected_session_density_preserves_sequence_sum_and_model_batch():
    emission_params = {
        'coefficients': jnp.ones((1, 1)),
        'bias': jnp.zeros(1),
        'covariance': jnp.eye(1),
    }
    initial_params = {
        'initial_mean': jnp.zeros(1),
        'initial_covariance': jnp.eye(1),
    }

    model = GaussianLDS.from_params(
        **initial_params,
        emissions=(emission_params,),
        dynamics_coefficients=jnp.eye(1),
        dynamics_bias=jnp.zeros(1),
        dynamics_covariance=jnp.eye(1),
    ).core

    # Different observation dimensions and sequence shapes expose mismatched
    # session selection. The second emission mean is 2 at zero latents.
    selected_emission = GaussianEmissions(
        LinearGaussian.from_components(
            jnp.ones((2, 1)), jnp.full((2,), 2.0), jnp.eye(2)
        )
    )
    model = model._replace(
        emissions=ContinuousEmissionsCollection(
            (model.emissions.get(), selected_emission)
        )
    ).broadcast((2,))
    data = DatasetCollection.from_datasets(
        (
            Dataset.from_array(jnp.zeros((1, 1))),
            Dataset.from_arrays((jnp.zeros((3, 2)), jnp.zeros((2, 2)))),
        )
    )
    latents = jnp.zeros((2, 2, 3, 1))

    def density(m, d, x, session=None):
        return m.log_joint(d, x, session=session)

    # Five initial/transition scalar Gaussian factors, plus ten observation
    # factors at residual -2. Padding and the other session do not contribute.
    expected = -7.5 * np.log(2 * np.pi) - 20.0
    actual = density(model, data, latents, session=1)
    assert actual.shape == (2,)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(
        jax.jit(lambda m, d, x: density(m, d, x, session=1))(model, data, latents),
        actual,
        rtol=1e-6,
    )
    singleton_data = DatasetCollection.from_datasets((data.get(1),))
    singleton_model = model._replace(
        emissions=ContinuousEmissionsCollection((model.emissions.get(1),))
    )
    np.testing.assert_allclose(
        density(singleton_model, singleton_data, latents), expected, rtol=1e-6
    )


def test_hmm_observation_mean_selects_matching_data_and_emission():
    model = GaussianHMM.from_params(
        initial_probs=jnp.ones(1),
        transition_probs=jnp.ones((1, 1)),
        emissions=(
            {
                'mean': jnp.full((1, 1), 2.0),
                'covariance': jnp.ones((1, 1, 1)),
            },
        ),
    )
    data = DatasetCollection.from_datasets((Dataset.from_array(jnp.zeros((3, 1))),))
    inferred = model.infer(data)
    emissions = model.core.emissions
    multi_model = GaussianHMM(
        model.core._replace(
            emissions=DiscreteEmissionsCollection((emissions.get(), emissions.get()))
        )
    )
    inferred = dataclasses.replace(inferred, model=multi_model)

    np.testing.assert_allclose(inferred.observation_mean(session=0).get(0), 2.0)
    # Emission 1 exists, but dataset 1 does not; selection must check both.
    with pytest.raises(IndexError):
        inferred.observation_mean(session=1)


def test_arhmm_inferred_conditionals_forward_session():
    model = GaussianARHMM.from_params(
        initial_probs=jnp.ones(1),
        transition_probs=jnp.ones((1, 1)),
        emissions=(
            {
                'coefficients': jnp.ones((1, 1, 1, 1)),
                'bias': jnp.full((1, 1), 2.0),
                'covariance': jnp.ones((1, 1, 1)),
            },
        ),
    )
    data = DatasetCollection.from_datasets(
        (Dataset.from_array(jnp.array([[1.0], [2.0], [3.0]])),)
    )
    inferred = model.infer(data)
    emissions = model.core.emissions
    multi_model = GaussianARHMM(
        model.core._replace(
            emissions=AREmissionsCollection((emissions.get(), emissions.get()))
        )
    )
    inferred = dataclasses.replace(inferred, model=multi_model)

    conditional = inferred.states_conditional(session=0)
    assert isinstance(conditional, TemporalGaussian)
    np.testing.assert_allclose(conditional.dist.mean, [[[[3.0]], [[4.0]]]])
