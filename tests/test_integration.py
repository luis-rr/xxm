import typing
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm import arhmm, hmm, lds, slds
from xxm.arhmm.inference import infer_exact as infer_arhmm
from xxm.arhmm.learning import em_step as arhmm_em_step
from xxm.core.optim.newton import OptimParams
from xxm.hmm.inference import infer_exact as infer_hmm
from xxm.hmm.learning import em_step as hmm_em_step
from xxm.lds.inference import infer_exact as infer_lds
from xxm.lds.inference import infer_laplace as infer_lds_laplace
from xxm.lds.learning import em_step as lds_em_step
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


GAUSSIAN_OBSERVATIONS = jnp.array(
    [
        [0.0],
        [0.5],
        [-0.25],
        [0.25],
    ]
)

# Constant unit counts keep the Poisson M-step well conditioned and make the
# intercept-only model an exact optimum. The integration test is about wiring,
# finiteness, and JAX compatibility, not optimizer convergence speed.
POISSON_OBSERVATIONS = jnp.ones((4, 1), dtype=jnp.float32)

INITIAL_PROBS = jnp.array([0.6, 0.4])
TRANSITION_PROBS = jnp.array([[0.8, 0.2], [0.2, 0.8]])

# Laplace inference only needs a couple of Newton iterations here. Dedicated
# optimizer tests cover convergence behavior in detail.
FAST_LAPLACE_PARAMS = OptimParams(
    max_iter=2,
    tol=1e-4,
    max_line_search_iters=2,
)


def _hmm_gaussian():
    return hmm.GaussianHMM.from_params(
        initial_probs=INITIAL_PROBS,
        transition_probs=TRANSITION_PROBS,
        emission_means=jnp.array([[0.0], [0.5]]),
        emission_covariances=jnp.array([[[1.0]], [[1.0]]]),
    )._model


def _hmm_poisson():
    return hmm.PoissonHMM.from_params(
        initial_probs=INITIAL_PROBS,
        transition_probs=TRANSITION_PROBS,
        emission_log_rates=jnp.zeros((2, 1)),
    )._model


def _arhmm_gaussian():
    return arhmm.GaussianARHMM.from_params(
        initial_probs=INITIAL_PROBS,
        transition_probs=TRANSITION_PROBS,
        emission_coefficients=jnp.array([[[[0.2]]], [[[-0.1]]]]),
        emission_bias=jnp.zeros((2, 1)),
        emission_covariances=jnp.array([[[1.0]], [[1.0]]]),
    )._model


def _arhmm_poisson():
    return arhmm.PoissonARHMM.from_params(
        initial_probs=INITIAL_PROBS,
        transition_probs=TRANSITION_PROBS,
        emission_coefficients=jnp.zeros((2, 1, 1, 1)),
        emission_bias=jnp.zeros((2, 1)),
    )._model


def _lds_gaussian():
    return lds.GaussianLDS.from_params(
        initial_mean=jnp.array([0.0]),
        initial_covariance=jnp.array([[1.0]]),
        dynamics_coefficients=jnp.array([[0.8]]),
        dynamics_bias=jnp.array([0.0]),
        dynamics_covariance=jnp.array([[0.5]]),
        emission_coefficients=jnp.array([[1.0]]),
        emission_bias=jnp.array([0.0]),
        emission_covariance=jnp.array([[1.0]]),
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


MODEL_CASES = [
    ModelCase(
        name='hmm-gaussian',
        make_model=_hmm_gaussian,
        infer=infer_hmm,
        infer_kwargs={},
        em_step=hmm_em_step,
        observations=GAUSSIAN_OBSERVATIONS,
    ),
    ModelCase(
        name='hmm-poisson',
        make_model=_hmm_poisson,
        infer=infer_hmm,
        infer_kwargs={},
        em_step=hmm_em_step,
        observations=POISSON_OBSERVATIONS,
    ),
    ModelCase(
        name='arhmm-gaussian',
        make_model=_arhmm_gaussian,
        infer=infer_arhmm,
        infer_kwargs={},
        em_step=arhmm_em_step,
        observations=GAUSSIAN_OBSERVATIONS,
    ),
    ModelCase(
        name='arhmm-poisson',
        make_model=_arhmm_poisson,
        infer=infer_arhmm,
        infer_kwargs={},
        em_step=arhmm_em_step,
        observations=POISSON_OBSERVATIONS,
    ),
    ModelCase(
        name='lds-gaussian',
        make_model=_lds_gaussian,
        infer=infer_lds,
        infer_kwargs={},
        em_step=lds_em_step,
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
            'num_iters': 1,
            'initial_latents': jnp.zeros((GAUSSIAN_OBSERVATIONS.shape[0], 1)),
        },
        em_step=partial(
            slds_variational_em_step,
            num_inference_iters=1,
        ),
        observations=GAUSSIAN_OBSERVATIONS,
    ),
    ModelCase(
        name='slds-poisson',
        make_model=_slds_poisson,
        infer=infer_slds_laplace,
        infer_kwargs={
            'num_iters': 1,
            'initial_latents': jnp.zeros((POISSON_OBSERVATIONS.shape[0], 1)),
            'params': FAST_LAPLACE_PARAMS,
        },
        em_step=partial(
            slds_laplace_em_step,
            params=FAST_LAPLACE_PARAMS,
            num_inference_iters=1,
        ),
        observations=POISSON_OBSERVATIONS,
    ),
]

# JIT compilation is intentionally representative rather than exhaustive.
# Family-specific and low-level tests separately exercise Gaussian/Poisson
# fitting and inference kernels.
JIT_CASES = [
    MODEL_CASES[0],  # exact discrete EM
    MODEL_CASES[3],  # AR + Poisson Newton M-step
    MODEL_CASES[5],  # continuous Laplace EM
    MODEL_CASES[6],  # switching variational EM
    MODEL_CASES[7],  # switching Laplace EM
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
    model = case.make_model()
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
    JIT_CASES,
    ids=lambda case: case.name,
)
def test_representative_em_steps_are_jittable(case):
    model = case.make_model()
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
