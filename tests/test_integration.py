import typing
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.hmm.init import (
    init_gaussian,
    init_gaussian_ar,
    init_poisson,
    init_poisson_ar,
)
from xxm.hmm.learning import em_step as hmm_em_step
from xxm.lds.init import init_pca_gaussian as initialize_lds_gaussian
from xxm.lds.init import init_pca_poisson as initialize_lds_poisson
from xxm.lds.learning import em_step as lds_em_step
from xxm.lds.learning import laplace_em_step as lds_laplace_em_step


class ModelCase(typing.NamedTuple):
    name: str
    initialize: Callable
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
        em_step=hmm_em_step,
        observations=GAUSSIAN_OBSERVATIONS,
        init_kwargs={'num_states': 2, 'key': jax.random.key(0)},
    ),
    ModelCase(
        name='hmm-poisson',
        initialize=init_poisson,
        em_step=hmm_em_step,
        observations=POISSON_OBSERVATIONS,
        init_kwargs={'num_states': 2, 'key': jax.random.key(0)},
    ),
    ModelCase(
        name='arhmm-gaussian',
        initialize=init_gaussian_ar,
        em_step=hmm_em_step,
        observations=GAUSSIAN_OBSERVATIONS,
        init_kwargs={'num_states': 2, 'num_lags': 1, 'key': jax.random.key(0)},
    ),
    ModelCase(
        name='arhmm-poisson',
        initialize=init_poisson_ar,
        em_step=hmm_em_step,
        observations=POISSON_OBSERVATIONS,
        init_kwargs={'num_states': 2, 'num_lags': 1, 'key': jax.random.key(0)},
    ),
    ModelCase(
        name='lds-gaussian',
        initialize=initialize_lds_gaussian,
        em_step=lds_em_step,
        observations=GAUSSIAN_OBSERVATIONS,
        init_kwargs={'latent_dim': 1},
    ),
    ModelCase(
        name='lds-poisson',
        initialize=initialize_lds_poisson,
        em_step=lds_laplace_em_step,
        observations=POISSON_OBSERVATIONS,
        init_kwargs={'latent_dim': 1},
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
def test_em_step(case):
    model = case.initialize(
        observations=case.observations,
        **case.init_kwargs,
    )

    updated_model, objective = case.em_step(
        model,
        case.observations,
    )

    assert objective.shape == ()
    assert jnp.isfinite(objective)
    assert_tree_finite(updated_model)


@pytest.mark.parametrize(
    'case',
    MODEL_CASES,
    ids=lambda case: case.name,
)
def test_em_step_jittable(case):
    model = case.initialize(
        observations=case.observations,
        **case.init_kwargs,
    )

    eager_model, eager_objective = case.em_step(
        model,
        case.observations,
    )

    jit_model, jit_objective = jax.jit(case.em_step)(
        model,
        case.observations,
    )

    np.testing.assert_allclose(
        jit_objective,
        eager_objective,
        rtol=1e-4,
        atol=1e-5,
    )

    for eager_leaf, jit_leaf in zip(
        jax.tree_util.tree_leaves(eager_model),
        jax.tree_util.tree_leaves(jit_model),
        strict=True,
    ):
        np.testing.assert_allclose(
            jit_leaf,
            eager_leaf,
            rtol=1e-4,
            atol=1e-5,
        )
