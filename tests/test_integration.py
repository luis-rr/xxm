import typing
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS
from xxm.hmm.inference import infer_exact as infer_hmm
from xxm.hmm.init import (
    init_gaussian,
    init_gaussian_ar,
    init_poisson,
    init_poisson_ar,
)
from xxm.hmm.learning import em_step as hmm_em_step
from xxm.lds.inference import infer_exact as infer_lds
from xxm.lds.inference import infer_laplace as infer_lds_laplace
from xxm.lds.init import init_pca_gaussian as initialize_lds_gaussian
from xxm.lds.init import init_pca_poisson as initialize_lds_poisson
from xxm.lds.learning import em_step as lds_em_step
from xxm.lds.learning import laplace_em_step as lds_laplace_em_step
from xxm.slds.inference import infer_laplace as infer_slds_laplace
from xxm.slds.inference import infer_variational as infer_slds
from xxm.slds.init import init_pca_gaussian as initialize_slds_gaussian
from xxm.slds.init import init_pca_poisson as initialize_slds_poisson
from xxm.slds.learning import laplace_em_step as slds_laplace_em_step
from xxm.slds.learning import variational_em_step as slds_variational_em_step


class ModelCase(typing.NamedTuple):
    name: str
    initialize: Callable
    infer: Callable
    infer_kwargs: dict
    em_step: Callable
    observations: jax.Array
    init_kwargs: dict


GAUSSIAN_OBSERVATIONS = jnp.array(
    [
        [0.0, 0.1],
        [0.2, 0.0],
        [1.0, 1.1],
        [0.9, 1.0],
        [0.1, 0.2],
        [1.1, 0.9],
    ]
)

AR_GAUSSIAN_OBSERVATIONS = jnp.array(
    [
        [0.0, 0.1],
        [0.2, 0.0],
        [1.0, 1.1],
        [0.9, 1.0],
        [0.1, 0.2],
        [1.1, 0.9],
        [0.0, 0.1],
        [0.2, 0.0],
        [1.0, 1.1],
        [0.9, 1.0],
        [0.1, 0.2],
        [1.1, 0.9],
    ]
)

POISSON_OBSERVATIONS = jnp.array(
    [
        [0, 1],
        [1, 0],
        [2, 1],
        [3, 2],
        [1, 1],
        [2, 3],
    ],
    dtype=jnp.float32,
)


MODEL_CASES = [
    ModelCase(
        name='hmm-gaussian',
        initialize=init_gaussian,
        infer=infer_hmm,
        infer_kwargs={},
        em_step=hmm_em_step,
        observations=GAUSSIAN_OBSERVATIONS,
        init_kwargs={'num_states': 2, 'key': jax.random.key(0)},
    ),
    ModelCase(
        name='hmm-poisson',
        initialize=init_poisson,
        infer=infer_hmm,
        infer_kwargs={},
        em_step=hmm_em_step,
        observations=POISSON_OBSERVATIONS,
        init_kwargs={'num_states': 2, 'key': jax.random.key(0)},
    ),
    ModelCase(
        name='arhmm-gaussian',
        initialize=init_gaussian_ar,
        infer=infer_hmm,
        infer_kwargs={},
        em_step=hmm_em_step,
        observations=AR_GAUSSIAN_OBSERVATIONS,
        init_kwargs={'num_states': 2, 'num_lags': 1, 'key': jax.random.key(0)},
    ),
    ModelCase(
        name='arhmm-poisson',
        initialize=init_poisson_ar,
        infer=infer_hmm,
        infer_kwargs={},
        em_step=hmm_em_step,
        observations=POISSON_OBSERVATIONS,
        init_kwargs={'num_states': 2, 'num_lags': 1, 'key': jax.random.key(0)},
    ),
    ModelCase(
        name='lds-gaussian',
        initialize=initialize_lds_gaussian,
        infer=infer_lds,
        infer_kwargs={},
        em_step=lds_em_step,
        observations=GAUSSIAN_OBSERVATIONS,
        init_kwargs={'latent_dim': 1},
    ),
    ModelCase(
        name='lds-poisson',
        initialize=initialize_lds_poisson,
        infer=infer_lds_laplace,
        infer_kwargs={'params': DEFAULT_OPTIM_PARAMS},
        em_step=partial(
            lds_laplace_em_step,
            params=DEFAULT_OPTIM_PARAMS,
        ),
        observations=POISSON_OBSERVATIONS,
        init_kwargs={'latent_dim': 1},
    ),
    ModelCase(
        name='slds-gaussian',
        initialize=initialize_slds_gaussian,
        infer=infer_slds,
        infer_kwargs={'num_iters': 2},
        em_step=partial(
            slds_variational_em_step,
            num_inference_iters=2,
        ),
        observations=GAUSSIAN_OBSERVATIONS,
        init_kwargs={
            'num_states': 2,
            'latent_dim': 1,
            'key': jax.random.key(0),
        },
    ),
    ModelCase(
        name='slds-poisson',
        initialize=initialize_slds_poisson,
        infer=infer_slds_laplace,
        infer_kwargs={
            'num_iters': 2,
            'params': DEFAULT_OPTIM_PARAMS,
        },
        em_step=partial(
            slds_laplace_em_step,
            params=DEFAULT_OPTIM_PARAMS,
            num_inference_iters=2,
        ),
        observations=POISSON_OBSERVATIONS,
        init_kwargs={
            'num_states': 2,
            'latent_dim': 1,
            'key': jax.random.key(0),
        },
    ),
]


def assert_tree_finite(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            assert jnp.all(jnp.isfinite(leaf))


@pytest.mark.parametrize(
    'case',
    MODEL_CASES,
    ids=lambda case: case.name,
)
def test_one_em_step(case):
    model = case.initialize(
        observations=case.observations,
        **case.init_kwargs,
    )

    inferred = case.infer(
        model,
        case.observations,
        **case.infer_kwargs,
    )
    result = case.em_step(inferred, case.observations)

    assert result.objective.shape == ()
    assert jnp.isfinite(result.objective)
    assert_tree_finite(result.model)


@pytest.mark.parametrize(
    'case',
    MODEL_CASES,
    ids=lambda case: case.name,
)
def test_one_em_step_is_jittable(case):
    model = case.initialize(
        observations=case.observations,
        **case.init_kwargs,
    )

    inferred = case.infer(
        model,
        case.observations,
        **case.infer_kwargs,
    )
    eager = case.em_step(inferred, case.observations)
    jitted = jax.jit(case.em_step)(
        inferred,
        case.observations,
    )

    np.testing.assert_allclose(
        jitted.objective,
        eager.objective,
        rtol=1e-4,
        atol=1e-5,
    )

    for eager_leaf, jit_leaf in zip(
        jax.tree_util.tree_leaves(eager.model),
        jax.tree_util.tree_leaves(jitted.model),
        strict=True,
    ):
        np.testing.assert_allclose(
            jit_leaf,
            eager_leaf,
            rtol=1e-4,
            atol=1e-5,
        )
