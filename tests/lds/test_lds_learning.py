import jax
import numpy as np

from xxm.core.data import Dataset
from xxm.lds.inference import infer_exact
from xxm.lds.learning import em_step, fit_em

from .lds_helpers import make_model, make_observations


def test_em_step_returns_a_model_and_finite_objective():
    data = Dataset.from_sequence(make_observations())
    inferred = infer_exact(make_model(), data)
    result = em_step(inferred, data)
    model = result.model
    objective = result.objective

    assert model.initial.dist.mean.shape == (2,)
    assert model.dynamics.dist.affine.coefficients.shape == (2, 2)
    assert np.isfinite(objective)


def test_em_step_is_jittable():
    model = make_model()
    observations = Dataset.from_sequence(make_observations())

    inferred = infer_exact(model, observations)
    eager = em_step(inferred, observations)
    jitted = jax.jit(em_step)(inferred, observations)

    np.testing.assert_allclose(
        jitted.model.dynamics.dist.affine.coefficients,
        eager.model.dynamics.dist.affine.coefficients,
    )
    np.testing.assert_allclose(jitted.objective, eager.objective)


def test_fit_em_includes_the_final_objective():
    fit = fit_em(make_model(), Dataset.from_sequence(make_observations()), num_iters=1)

    assert fit.objective_trace.shape == (2,)
    assert np.isfinite(fit.objective_trace).all()
