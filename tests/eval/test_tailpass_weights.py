import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from scorio.eval.tailpass_weights import (
    beta_weights,
    discovery_weights,
    maxent_weights,
    moment_weights,
    payoff_weights,
    stability_weights,
    threshold_weights,
    uniform_weights,
)


@pytest.mark.parametrize("k", [1, 2, 7, 64])
def test_weight_family_identities(k: int) -> None:
    np.testing.assert_allclose(moment_weights(k, 1), uniform_weights(k), atol=1e-15)
    np.testing.assert_allclose(maxent_weights(k, 0.5), uniform_weights(k), atol=1e-15)
    np.testing.assert_allclose(beta_weights(k, 0.5, 2), uniform_weights(k), atol=1e-15)
    for lam in (0.25, 0.5, 2, 4):
        weights = moment_weights(k, lam)
        np.testing.assert_allclose(
            weights, beta_weights(k, lam / (lam + 1), lam + 1), atol=2e-15
        )
        np.testing.assert_allclose(
            weights, payoff_weights((np.arange(k + 1) / k) ** lam), atol=2e-15
        )
        assert weights.min() >= 0
        assert weights.sum() == pytest.approx(1)
    np.testing.assert_array_equal(discovery_weights(k), threshold_weights(k, 1))
    np.testing.assert_array_equal(stability_weights(k), threshold_weights(k, k))


def test_maxent_weights_against_numerical_density_integration() -> None:
    target = 0.7

    def average(tau: float) -> float:
        return (
            quad(lambda x: x * np.exp(tau * x), 0, 1)[0]
            / quad(lambda x: np.exp(tau * x), 0, 1)[0]
        )

    tau = brentq(lambda x: average(x) - target, 0, 20)
    normalization = quad(lambda x: np.exp(tau * x), 0, 1)[0]
    expected = [
        quad(lambda x: np.exp(tau * x), t / 5, (t + 1) / 5)[0] / normalization
        for t in range(5)
    ]
    np.testing.assert_allclose(maxent_weights(5, target), expected, atol=1e-14)
    np.testing.assert_allclose(
        maxent_weights(5, 0.3), np.array(expected)[::-1], atol=1e-14
    )


def test_extreme_weight_parameters() -> None:
    np.testing.assert_allclose(
        moment_weights(8, 1e-12), discovery_weights(8), atol=3e-12
    )
    np.testing.assert_array_equal(moment_weights(8, 1e300), stability_weights(8))
    np.testing.assert_array_equal(maxent_weights(8, 1e-300), discovery_weights(8))
    for mean in (np.nextafter(0.5, 0), np.nextafter(0.5, 1), 1e-10, 1 - 1e-10):
        weights = maxent_weights(8, mean)
        assert np.all(np.isfinite(weights)) and np.all(weights >= 0)
        assert weights.sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "call",
    [
        lambda: uniform_weights(0),
        lambda: threshold_weights(2, 3),
        lambda: moment_weights(2, 0),
        lambda: moment_weights(2, np.inf),
        lambda: moment_weights(2, True),
        lambda: beta_weights(2, 0, 1),
        lambda: beta_weights(2, 1, 1),
        lambda: beta_weights(2, 0.5, -1),
        lambda: maxent_weights(2, 0),
        lambda: maxent_weights(2, 1),
        lambda: payoff_weights([0.1, 0.5, 1]),
        lambda: payoff_weights([0, 0.8, 0.5, 1]),
        lambda: payoff_weights([0, np.nan, 1]),
    ],
)
def test_invalid_weights(call) -> None:
    with pytest.raises(ValueError):
        call()
