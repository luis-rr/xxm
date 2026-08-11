import jax
import jax.numpy as jnp
import numpy as np

from xxm.lds.core import GaussianEmissions, Model

jax.config.update('jax_enable_x64', True)

RTOL = 1e-7
ATOL = 1e-6

T = 3


def make_emissions() -> GaussianEmissions:
    emissions = GaussianEmissions()
    emissions.readout = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    emissions.bias = jnp.array([0.1, -0.2])
    emissions.noise_covariance = jnp.array([[1.5, 0.1], [0.1, 1.0]])
    return emissions


def make_model() -> Model:
    return Model(
        initial_mean=jnp.array([0.5, -0.3]),
        initial_covariance=jnp.array([[2.0, 0.2], [0.2, 1.5]]),
        dynamics_matrix=jnp.array([[0.9, 0.1], [0.0, 0.8]]),
        dynamics_bias=jnp.array([0.05, -0.05]),
        dynamics_noise_covariance=jnp.array([[0.5, 0.05], [0.05, 0.4]]),
        emissions=make_emissions(),
    )


def make_observations() -> jax.Array:
    return jnp.array([[1.0, 0.5], [0.2, -0.3], [-0.5, 0.8]])


def test_to_chain_lower_precision_blocks():
    model = make_model()
    chain = model._to_chain(make_observations())

    A = np.array(model.dynamics_matrix)
    Q_inv = np.linalg.inv(np.array(model.dynamics_noise_covariance))
    expected_lower = -Q_inv @ A

    np.testing.assert_allclose(
        chain.lower_precision_blocks,
        np.broadcast_to(expected_lower, (T - 1, 2, 2)),
        atol=ATOL,
        rtol=RTOL,
    )


def test_to_chain_diagonal_precision_blocks():
    model = make_model()
    observations = make_observations()
    chain = model._to_chain(observations)

    A = np.array(model.dynamics_matrix)
    Q_inv = np.linalg.inv(np.array(model.dynamics_noise_covariance))
    C = np.array(model.emissions.readout)
    R_inv = np.linalg.inv(np.array(model.emissions.noise_covariance))
    J0 = np.linalg.inv(np.array(model.initial_covariance))

    obs_prec = C.T @ R_inv @ C
    dynamics_left = A.T @ Q_inv @ A
    dynamics_right = Q_inv

    expected = np.array(
        [
            obs_prec + J0 + dynamics_left,
            obs_prec + dynamics_left + dynamics_right,
            obs_prec + dynamics_right,
        ]
    )

    np.testing.assert_allclose(chain.diagonal_precision_blocks, expected, atol=ATOL, rtol=RTOL)


def test_to_chain_information_vectors():
    model = make_model()
    observations = make_observations()
    chain = model._to_chain(observations)

    A = np.array(model.dynamics_matrix)
    b = np.array(model.dynamics_bias)
    Q_inv = np.linalg.inv(np.array(model.dynamics_noise_covariance))
    C = np.array(model.emissions.readout)
    d_obs = np.array(model.emissions.bias)
    R_inv = np.linalg.inv(np.array(model.emissions.noise_covariance))
    mu0 = np.array(model.initial_mean)
    J0 = np.linalg.inv(np.array(model.initial_covariance))
    y = np.array(observations)

    obs_info = (y - d_obs) @ (R_inv @ C)
    initial_info = J0 @ mu0
    left_info = -A.T @ Q_inv @ b
    right_info = Q_inv @ b

    expected = np.array(
        [
            obs_info[0] + initial_info + left_info,
            obs_info[1] + left_info + right_info,
            obs_info[2] + right_info,
        ]
    )

    np.testing.assert_allclose(chain.information_vectors, expected, atol=ATOL, rtol=RTOL)


def test_inference_means_match_dense_reference():
    model = make_model()
    observations = make_observations()

    posterior = model.inference(observations)

    chain = model._to_chain(observations)
    dense_J = np.asarray(chain.dense_precision())
    dense_h = np.asarray(chain.information_vectors).reshape(-1)
    expected = np.linalg.solve(dense_J, dense_h).reshape((T, 2))

    np.testing.assert_allclose(posterior.means, expected, atol=ATOL, rtol=RTOL)


def test_inference_covariances_match_dense_reference():
    model = make_model()
    observations = make_observations()

    posterior = model.inference(observations)

    chain = model._to_chain(observations)
    dense_inv = np.linalg.inv(np.asarray(chain.dense_precision()))
    expected = [dense_inv[t * 2 : (t + 1) * 2, t * 2 : (t + 1) * 2] for t in range(T)]

    np.testing.assert_allclose(posterior.covariances, expected, atol=ATOL, rtol=RTOL)


def test_inference_covariances_are_symmetric_and_positive_definite():
    model = make_model()
    posterior = model.inference(make_observations())

    for cov in posterior.covariances:
        np.testing.assert_allclose(cov, cov.T, atol=ATOL)
        assert np.all(np.linalg.eigvalsh(np.asarray(cov)) > 0.0)


def test_inference_jit():
    model = make_model()
    observations = make_observations()

    eager = model.inference(observations)
    jitted = jax.jit(model.inference)(observations)

    np.testing.assert_allclose(np.asarray(jitted.means), np.asarray(eager.means))
    np.testing.assert_allclose(np.asarray(jitted.covariances), np.asarray(eager.covariances))
    np.testing.assert_allclose(jitted.log_normalizer, eager.log_normalizer)
