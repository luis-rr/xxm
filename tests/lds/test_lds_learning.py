import jax
import numpy as np

from tests.lds.lds_helpers import make_model, make_observations
from xxm.lds.learning import em_step, fit_em


def test_em_step_returns_a_model_and_finite_objective():
    model, objective = em_step(make_model(), make_observations())

    assert model.initial.model.mean.shape == (2,)
    assert model.dynamics.model.affine.coefficients.shape == (2, 2)
    assert np.isfinite(objective)


def test_em_step_is_jittable():
    model = make_model()
    observations = make_observations()

    eager_model, eager_objective = em_step(model, observations)
    jitted_model, jitted_objective = jax.jit(em_step)(model, observations)

    np.testing.assert_allclose(
        jitted_model.dynamics.model.affine.coefficients,
        eager_model.dynamics.model.affine.coefficients,
    )
    np.testing.assert_allclose(jitted_objective, eager_objective)


def test_fit_em_includes_the_final_objective():
    fit = fit_em(make_model(), make_observations(), num_iters=1)

    assert fit.objective_trace.shape == (2,)
    assert np.isfinite(fit.objective_trace).all()
