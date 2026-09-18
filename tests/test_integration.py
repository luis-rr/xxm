import typing
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm import arhmm, lds, slds
from xxm.arhmm.inference import infer_exact as infer_arhmm
from xxm.arhmm.learning import em_step as arhmm_em_step
from xxm.core.optim.newton import OptimParams
from xxm.lds.inference import infer_laplace as infer_lds_laplace
from xxm.lds.learning import laplace_em_step as lds_laplace_em_step
from xxm.slds.inference import infer_laplace as infer_slds_laplace
from xxm.slds.inference import infer_variational as infer_slds
from xxm.slds.learning import laplace_em_step as slds_laplace_em_step
from xxm.slds.learning import variational_em_step as slds_variational_em_step


class ModelCase(typing.NamedTuple):
    name: str
    make_model: Callable
    infer: Callable
    infer_kwargs: dict
    em_step: Callable
    observations: jax.Array


GAUSSIAN_OBSERVATIONS = jnp.array([[0.0], [0.5], [-0.25]])
POISSON_OBSERVATIONS = jnp.ones((3, 1), dtype=jnp.float32)

INITIAL_PROBS = jnp.array([0.6, 0.4])
TRANSITION_PROBS = jnp.array([[0.8, 0.2], [0.2, 0.8]])

# Integration tests only need to traverse the Laplace machinery once. Newton
# convergence and backtracking are covered by dedicated optimizer/inference
# tests, so a single full-step attempt is sufficient here.
FAST_LAPLACE_PARAMS = OptimParams(
    max_iter=1,
    tol=1e-4,
    max_line_search_iters=0,
)


def _arhmm_gaussian():
    return arhmm.GaussianARHMM.from_params(
        initial_probs=INITIAL_PROBS,
        transition_probs=TRANSITION_PROBS,
        emission_coefficients=jnp.array([[[[0.2]]], [[[-0.1]]]]),
        emission_bias=jnp.zeros((2, 1)),
        emission_covariances=jnp.array([[[1.0]], [[1.0]]]),
    )._model


def _lds_poisson():
    return lds.PoissonLDS.from_params(
        initial_mean=jnp.array([0.0]),
        initial_covariance=jnp.array([[1.0]]),
        dynamics_coefficients=jnp.array([[0.8]]),
        dynamics_bias=jnp.array([0.0]),
        dynamics_covariance=jnp.array([[0.5]]),
        emission_coefficients=jnp.zeros((1, 1)),
        emission_bias=jnp.zeros(1),
    )._model


def _slds_gaussian():
    return slds.GaussianSLDS.from_params(
        initial_probs=INITIAL_PROBS,
        transition_probs=TRANSITION_PROBS,
        latent_initial_means=jnp.array([[0.0], [0.5]]),
        latent_initial_covariances=jnp.array([[[1.0]], [[1.0]]]),
        dynamics_coefficients=jnp.array([[[0.8]], [[0.2]]]),
        dynamics_bias=jnp.zeros((2, 1)),
        dynamics_covariances=jnp.array([[[0.5]], [[0.5]]]),
        emission_coefficients=jnp.array([[1.0]]),
        emission_bias=jnp.zeros(1),
        emission_covariance=jnp.array([[1.0]]),
    )._model


def _slds_poisson():
    return slds.PoissonSLDS.from_params(
        initial_probs=INITIAL_PROBS,
        transition_probs=TRANSITION_PROBS,
        latent_initial_means=jnp.array([[0.0], [0.5]]),
        latent_initial_covariances=jnp.array([[[1.0]], [[1.0]]]),
        dynamics_coefficients=jnp.array([[[0.8]], [[0.2]]]),
        dynamics_bias=jnp.zeros((2, 1)),
        dynamics_covariances=jnp.array([[[0.5]], [[0.5]]]),
        emission_coefficients=jnp.zeros((1, 1)),
        emission_bias=jnp.zeros(1),
    )._model


# These four cases represent the distinct expensive learning stacks. HMM,
# Gaussian LDS, and Poisson AR-HMM behavior are already exercised directly in
# their family/statistics tests, so repeating them here only recompiles kernels.
MODEL_CASES = [
    ModelCase(
        name='arhmm-gaussian',
        make_model=_arhmm_gaussian,
        infer=infer_arhmm,
        infer_kwargs={},
        em_step=arhmm_em_step,
        observations=GAUSSIAN_OBSERVATIONS,
    ),
    ModelCase(
        name='lds-poisson',
        make_model=_lds_poisson,
        infer=infer_lds_laplace,
        infer_kwargs={'params': FAST_LAPLACE_PARAMS},
        em_step=partial(
            lds_laplace_em_step,
            params=FAST_LAPLACE_PARAMS,
        ),
        observations=POISSON_OBSERVATIONS,
    ),
    ModelCase(
        name='slds-gaussian',
        make_model=_slds_gaussian,
        infer=infer_slds,
        infer_kwargs={
            'num_iters': 0,
            'initial_latents': jnp.zeros((GAUSSIAN_OBSERVATIONS.shape[0], 1)),
        },
        em_step=partial(
            slds_variational_em_step,
            num_inference_iters=0,
        ),
        observations=GAUSSIAN_OBSERVATIONS,
    ),
    ModelCase(
        name='slds-poisson',
        make_model=_slds_poisson,
        infer=infer_slds_laplace,
        infer_kwargs={
            'num_iters': 0,
            'initial_latents': jnp.zeros((POISSON_OBSERVATIONS.shape[0], 1)),
            'params': FAST_LAPLACE_PARAMS,
        },
        em_step=partial(
            slds_laplace_em_step,
            params=FAST_LAPLACE_PARAMS,
            num_inference_iters=0,
        ),
        observations=POISSON_OBSERVATIONS,
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
def test_representative_em_steps_are_jittable(case):
    model = case.make_model()
    inferred = case.infer(
        model,
        case.observations,
        **case.infer_kwargs,
    )

    # The eager calculation is already needed as the numerical reference for
    # the JIT calculation, so a separate eager integration test is redundant.
    eager = case.em_step(inferred, case.observations)
    jitted = jax.jit(case.em_step)(
        inferred,
        case.observations,
    )

    assert eager.objective.shape == ()
    assert jnp.isfinite(eager.objective)
    assert_tree_finite(eager.model)

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
