from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betaln, gammaln
from scipy.stats import beta, binom

from scorio import eval
from scorio.eval.tailpass import TailPassDraws, TailPassProfile
from scorio.eval.tailpass_weights import (
    moment_weights,
    threshold_weights,
    uniform_weights,
)

R = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])


def test_binary_profile_matches_independent_beta_integration() -> None:
    prior_bank = np.array([[1, 0], [0, 0], [1, 1], [0, 1]])
    profile = eval.tailpass(R, 3, R0=prior_bank, eta=0.25, prior=[0.5, 1.5])
    means, covariances = [], []
    for row, prior_row in zip(R, prior_bank, strict=True):
        a = 1.5 + row.sum() + 0.25 * prior_row.sum()
        b = 0.5 + len(row) - row.sum() + 0.25 * (len(prior_row) - prior_row.sum())
        mean = np.array(
            [
                quad(
                    lambda p, t=t, a=a, b=b: binom.sf(t - 1, 3, p) * beta.pdf(p, a, b),
                    0,
                    1,
                )[0]
                for t in range(1, 4)
            ]
        )
        second = np.array(
            [
                [
                    quad(
                        lambda p, t=t, s=s, a=a, b=b: binom.sf(t - 1, 3, p)
                        * binom.sf(s - 1, 3, p)
                        * beta.pdf(p, a, b),
                        0,
                        1,
                    )[0]
                    for s in range(1, 4)
                ]
                for t in range(1, 4)
            ]
        )
        means.append(mean)
        covariances.append(second - np.outer(mean, mean))
    np.testing.assert_allclose(profile.question_mean, means, atol=2e-10)
    np.testing.assert_allclose(profile.mean, np.mean(means, axis=0), atol=2e-10)
    np.testing.assert_allclose(
        profile.covariance, np.sum(covariances, axis=0) / len(R) ** 2, atol=2e-10
    )
    assert np.linalg.eigvalsh(profile.covariance).min() >= -1e-14
    assert np.all(np.diff(profile.mean) <= 0)


@pytest.mark.parametrize("k", [1, 3, 4, 7])
def test_linear_identities_and_posterior_budget(k: int) -> None:
    profile = eval.tailpass(R, k)
    np.testing.assert_allclose(profile.moment(1), eval.bayes(R), atol=2e-14)
    np.testing.assert_allclose(
        profile.linear(uniform_weights(k)), eval.bayes(R), atol=2e-14
    )
    for lam in (0.25, 1.0, 2.0, 4.0):
        weights = moment_weights(k, lam)
        actual = profile.linear(weights)
        np.testing.assert_allclose(actual, profile.moment(lam), atol=2e-14)
        np.testing.assert_allclose(
            actual,
            (weights @ profile.mean, np.sqrt(weights @ profile.covariance @ weights)),
            atol=2e-13,
        )
    if k <= R.shape[1]:
        np.testing.assert_allclose(
            profile.linear(threshold_weights(k, 1)),
            eval.pass_at_k_ci(R, k)[:2],
            atol=2e-14,
        )
        np.testing.assert_allclose(
            profile.linear(threshold_weights(k, k)),
            eval.pass_hat_k_ci(R, k)[:2],
            atol=2e-14,
        )


def test_prior_transfer_and_complete_state_grouping() -> None:
    results = np.array([[0, 1], [0, 1]])
    prior = np.array([[0, 0], [1, 1]])
    profile = eval.tailpass(results, 3, R0=prior)
    assert np.all(profile.question_mean[0] < profile.question_mean[1])
    np.testing.assert_allclose(profile.moment(1), eval.bayes(results, R0=prior))
    ignored = eval.tailpass(results, 3, R0=prior, eta=0)
    np.testing.assert_allclose(ignored.mean, eval.tailpass(results, 3).mean)
    np.testing.assert_allclose(ignored.covariance, eval.tailpass(results, 3).covariance)
    duplicated = eval.tailpass(np.tile(R, (3, 1)), 4)
    np.testing.assert_allclose(duplicated.mean, eval.tailpass(R, 4).mean)
    np.testing.assert_allclose(
        duplicated.covariance, eval.tailpass(R, 4).covariance / 3
    )


def test_empirical_profile_matches_subset_enumeration() -> None:
    results = np.array([[0, 1, 2, 1], [2, 1, 1, 2]])
    scores = np.array([0.0, 0.5, 1.0])
    thresholds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = np.mean(
        [
            np.mean(
                [
                    scores[row[list(subset)]].mean() >= thresholds
                    for subset in combinations(range(4), 2)
                ],
                axis=0,
            )
            for row in results
        ],
        axis=0,
    )
    actual = eval.tailpass_empirical(results, 2, w=scores, thresholds=thresholds)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    binary = eval.tailpass_empirical(R, 3)
    assert binary[0] == pytest.approx(eval.pass_at_k(R, 3))
    assert binary[-1] == pytest.approx(eval.pass_hat_k(R, 3))
    assert binary.mean() == pytest.approx(R.mean())
    for t in range(1, 4):
        assert binary[t - 1] == pytest.approx(eval.g_pass_at_k_tau(R, 3, t / 3))
    np.testing.assert_allclose(
        eval.tailpass_empirical(R, 4),
        np.mean(R.sum(axis=1)[:, None] >= np.arange(1, 5), axis=0),
    )


def _categorical_oracle(
    alpha: np.ndarray, scores: np.ndarray, k: int, payoff: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Enumerate ordered future sequences, independently of the count-grid code.
    sequences = np.array(list(product(range(len(scores)), repeat=k)))
    counts = np.array([np.bincount(row, minlength=len(scores)) for row in sequences])

    def log_beta(values: np.ndarray) -> np.ndarray:
        return gammaln(values).sum(axis=-1) - gammaln(values.sum(axis=-1))

    base = log_beta(alpha)
    first = np.exp(log_beta(alpha + counts) - base)
    joint = np.exp(log_beta(alpha + counts[:, None] + counts[None, :]) - base)
    mean = first @ payoff
    covariance = payoff.T @ joint @ payoff - np.outer(mean, mean)
    return mean, covariance


def test_categorical_exact_moments_against_ordered_sequence_enumeration() -> None:
    results = np.array([0, 1, 2, 2])
    scores = np.array([0.0, 0.3, 1.0])
    prior = np.array([0.5, 1.5, 2.0])
    k = 3
    profile = eval.tailpass(results, k, w=scores, prior=prior)
    bank_scores = scores[np.array(list(product(range(3), repeat=k)))].mean(axis=1)
    alpha = prior + np.bincount(results, minlength=3)
    expected_mean, expected_cov = _categorical_oracle(
        alpha, scores, k, (bank_scores[:, None] >= profile.thresholds).astype(float)
    )
    np.testing.assert_allclose(profile.mean, expected_mean, atol=2e-13)
    np.testing.assert_allclose(profile.covariance, expected_cov, atol=2e-13)
    for lam in (0.25, 1, 2, 4):
        mean, covariance = _categorical_oracle(
            alpha, scores, k, bank_scores[:, None] ** lam
        )
        np.testing.assert_allclose(
            profile.moment(lam), (mean[0], np.sqrt(covariance[0, 0])), atol=2e-12
        )
    for weights in (uniform_weights(k), moment_weights(k, 2)):
        np.testing.assert_allclose(
            profile.linear(weights),
            (weights @ expected_mean, np.sqrt(weights @ expected_cov @ weights)),
            atol=2e-12,
        )


def test_categorical_off_grid_credit_and_binary_reduction() -> None:
    constant = eval.tailpass([[0, 0]], 1, w=[0.5])
    assert constant.mean[0] == 0
    assert constant.linear([1]) == (0, 0)
    assert constant.moment(1) == (0.5, 0)
    assert constant.moment(2) == (0.25, 0)
    assert constant.moment(4) == (0.0625, 0)
    assert constant.discovery() == (1, 0)
    assert constant.stability() == (0, 0)
    np.testing.assert_array_equal(
        constant.sample(10, rng=0).moment(2), np.full(10, 0.25)
    )
    implicit, explicit = eval.tailpass(R, 5), eval.tailpass(R, 5, w=[0, 1])
    np.testing.assert_array_equal(implicit.mean, explicit.mean)
    np.testing.assert_array_equal(implicit.covariance, explicit.covariance)


def test_equal_score_categories_preserve_prior_mass() -> None:
    refined = eval.tailpass([[0, 1, 2, 3]], 4, w=[0, 1, 0, 1], prior=[0.5, 1, 2, 3])
    coarse = eval.tailpass([[0, 1, 0, 1]], 4, prior=[2.5, 4])
    np.testing.assert_allclose(refined.mean, coarse.mean)
    np.testing.assert_allclose(refined.covariance, coarse.covariance)
    categorical = eval.tailpass([[0, 1, 2, 3]], 8, w=[0, 1, 0, 0.3])
    np.testing.assert_allclose(
        categorical.moment(1), eval.bayes([[0, 1, 2, 3]], w=[0, 1, 0, 0.3])
    )


def test_grid_boundaries_and_empirical_vs_posterior() -> None:
    boundary = 7 / 25
    thresholds = [0, boundary, np.nextafter(boundary, 1), 1]
    results = [1] * 7 + [0] * 18
    np.testing.assert_array_equal(
        eval.tailpass_empirical(results, 25, thresholds=thresholds), [1, 1, 0, 0]
    )
    profile = eval.tailpass(results, 25, thresholds=thresholds)
    full = eval.tailpass(results, 25)
    np.testing.assert_allclose(profile.mean[1:3], full.mean[6:8])
    assert profile.mean[0] == pytest.approx(1)
    np.testing.assert_allclose(profile.covariance[0], 0, atol=1e-28)
    assert eval.tailpass([0, 0], 2).mean[0] > 0
    assert eval.tailpass_empirical([0, 0], 2)[0] == 0


def test_sampling_joint_uncertainty_and_independent_questions() -> None:
    profile = eval.tailpass([[0, 1], [0, 1]], 4)
    draws = profile.sample(30_000, rng=23)
    np.testing.assert_allclose(draws.profile.mean(axis=0), profile.mean, atol=0.004)
    np.testing.assert_allclose(
        np.cov(draws.profile, rowvar=False), profile.covariance, rtol=0.04, atol=0.0002
    )
    weights = moment_weights(4, 2)
    np.testing.assert_allclose(draws.linear(weights), draws.moment(2), atol=2e-15)
    np.testing.assert_allclose(
        draws.power_mean(1, weights), draws.linear(weights), atol=2e-15
    )
    np.testing.assert_allclose(draws.at_k(7).moment(1), draws.moment(1), atol=2e-15)
    assert draws.at_k(7)._probabilities is draws._probabilities
    np.testing.assert_array_equal(
        profile.sample(10, rng=5).profile, profile.sample(10, rng=5).profile
    )
    summary = draws.summary(draws.moment(2))
    np.testing.assert_allclose(
        summary[2:], np.quantile(draws.moment(2), [0.025, 0.975])
    )
    np.testing.assert_array_equal(
        draws.summary(draws.moment(2) - draws.moment(2)), [0, 0, 0, 0]
    )


def test_nonlinear_question_first_definitions() -> None:
    source = eval.tailpass([[0, 1], [0, 1]], 2)
    p = np.array([[0.15, 0.75], [0.25, 0.5], [0, 1]])
    draws = TailPassDraws(source, np.stack([1 - p, p], axis=-1))
    tails = np.stack([1 - (1 - p) ** 2, p**2], axis=-1)
    expected = np.sqrt(np.mean(tails**2, axis=-1)).mean(axis=1)
    np.testing.assert_allclose(draws.qrs(), expected)
    np.testing.assert_allclose(
        draws.qrs(aggregation="profile"),
        np.sqrt(np.mean(tails.mean(axis=1) ** 2, axis=-1)),
    )
    assert draws.qrs()[0] != pytest.approx(draws.qrs(aggregation="profile")[0])
    np.testing.assert_allclose(draws.rollout(2), np.mean(tails**2, axis=(1, 2)))
    d, s = tails[..., 0], tails[..., 1]
    with np.errstate(divide="ignore"):
        expected_harmonic = 1 / (0.5 / d + 0.5 / s)
    np.testing.assert_allclose(draws.harmonic(), expected_harmonic.mean(axis=1))
    np.testing.assert_allclose(draws.harmonic(1), d.mean(axis=1))
    np.testing.assert_allclose(draws.harmonic(0), s.mean(axis=1))
    np.testing.assert_allclose(draws.discovery(), d.mean(axis=1))
    np.testing.assert_allclose(draws.stability(), s.mean(axis=1))
    gaps = np.maximum([0.8, 0.5] - tails, 0) * [2, 1]
    np.testing.assert_allclose(
        draws.shortfall([0.8, 0.5], [2, 1], epsilon=0.1),
        (gaps.max(axis=-1) + 0.1 * gaps.sum(axis=-1)).mean(axis=1),
    )
    np.testing.assert_array_equal(draws.shortfall([0, 0]), np.zeros(3))
    for q in (0.01, 2, 1000):
        np.testing.assert_allclose(
            draws.power_mean(q, [0, 1]), s.mean(axis=1), atol=1e-14
        )


def test_interval_methods_and_categorical_sampling() -> None:
    profile = eval.tailpass([[0, 1, 2], [1, 1, 2]], 2, w=[0, 0.5, 1])
    draws = profile.sample(20_000, rng=5)
    np.testing.assert_allclose(draws.profile.mean(axis=0), profile.mean, atol=0.004)
    np.testing.assert_allclose(
        np.cov(draws.profile, rowvar=False), profile.covariance, rtol=0.05, atol=0.0003
    )
    for lam in (0.5, 1, 2, 4):
        np.testing.assert_allclose(
            draws.summary(draws.moment(lam))[:2], profile.moment(lam), atol=0.004
        )
    normal = profile.ci(method="normal")
    np.testing.assert_array_equal(normal[0], profile.mean)
    np.testing.assert_array_equal(normal[1], profile.std)
    assert np.all(normal[2] >= 0) and np.all(normal[3] <= 1)
    np.testing.assert_array_equal(
        profile.ci(n_draws=20, rng=7), profile.sample(20, rng=7).summary()
    )
    assert len(profile.linear_ci([0.5, 0.5], n_draws=20, rng=7)) == 4
    assert len(profile.moment_ci(2, n_draws=20, rng=7)) == 4
    np.testing.assert_allclose(
        profile.moment_ci(2, method="normal")[:2], profile.moment(2)
    )


def test_large_categorical_budgets_use_moment_shortcuts_and_explicit_limits() -> None:
    profile = eval.tailpass([[0, 1, 2, 3, 4]], 64, w=[0, 0.25, 0.5, 0.75, 1])
    with pytest.raises(ValueError, match="count states"):
        _ = profile.mean
    draws = profile.sample(3000, rng=17)
    for lam in (1, 2, 4):
        mean, std = profile.moment(lam)
        assert 0 <= mean <= 1 and std >= 0
        np.testing.assert_allclose(
            draws.summary(draws.moment(lam))[:2], (mean, std), atol=0.006
        )
    medium = eval.tailpass([[0, 1, 2, 3]], 24, w=[0, 0.3, 0.5, 1])
    assert np.all(np.isfinite(medium.mean))
    with pytest.raises(ValueError, match="covariance.*large"):
        _ = medium.covariance


def test_large_binary_budget_and_concentrated_prior() -> None:
    profile = eval.tailpass([0, 1], 1000)
    assert np.all(np.isfinite(profile.mean))
    assert "_events" not in vars(profile)
    concentrated = eval.tailpass([0, 1], 8, prior=1e12)
    weights = uniform_weights(8)
    expected_sigma = concentrated.moment(1)[1]
    assert np.sqrt(weights @ concentrated.covariance @ weights) == pytest.approx(
        expected_sigma, rel=1e-7
    )
    assert np.linalg.eigvalsh(concentrated.covariance).min() >= -1e-25
    endpoints = eval.tailpass([0, 0], 1000, thresholds=[1 / 1000, 1])
    expected = -np.expm1(betaln(1, 3 + 1000) - betaln(1, 3))
    assert endpoints.mean[0] == pytest.approx(expected, abs=1e-12)


def test_input_copies_and_permutations() -> None:
    results = R.copy()
    prior = np.ones((4, 1), dtype=int)
    profile = eval.tailpass(results, 4, R0=prior)
    expected = profile.mean.copy()
    results[:] = 0
    prior[:] = 0
    np.testing.assert_array_equal(profile.mean, expected)
    for array in (
        profile.mean,
        profile.std,
        profile.covariance,
        profile.question_mean,
        profile.thresholds,
    ):
        assert not array.flags.writeable
    a = eval.tailpass(R, 4)
    b = eval.tailpass(R[::-1, ::-1], 4)
    np.testing.assert_allclose(a.mean, b.mean)
    np.testing.assert_allclose(a.covariance, b.covariance)
    np.testing.assert_allclose(
        eval.tailpass(R[0], 4).mean, eval.tailpass(R[:1], 4).mean
    )
    assert isinstance(a, TailPassProfile)


@pytest.mark.parametrize("concentration", [1e12, 1e20])
def test_concentrated_endpoint_covariance_preserves_reflection(
    concentration: float,
) -> None:
    high = eval.tailpass([0, 1], 8, prior=[1, concentration])
    low = eval.tailpass([0, 1], 8, prior=[concentration, 1])
    np.testing.assert_allclose(
        high.covariance, low.covariance[::-1, ::-1], rtol=2e-12, atol=0
    )
    np.testing.assert_allclose(
        high.std[[0, -1]],
        [high.discovery()[1], high.stability()[1]],
        rtol=2e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        low.std[[0, -1]], [low.discovery()[1], low.stability()[1]], rtol=2e-12, atol=0
    )
    assert high.stability()[1] > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": True},
        {"k": 1.5},
        {"eta": -0.1},
        {"eta": 1.1},
        {"eta": np.nan},
        {"prior": 0},
        {"prior": [1, -1]},
        {"prior": [1, 2, 3]},
        {"prior": np.inf},
        {"w": [0, 1.1]},
        {"w": [-0.1, 1]},
        {"thresholds": []},
        {"thresholds": [0.5, 0.5]},
        {"thresholds": [1, 0.5]},
        {"thresholds": [np.nan]},
        {"thresholds": [-0.1]},
        {"R0": [[0], [1]]},
    ],
)
def test_invalid_profile_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        eval.tailpass([0, 1], **({"k": 3} | kwargs))


@pytest.mark.parametrize("weights", [[], [1], [-1, 2], [0, 0], [0.4, 0.4], [np.nan, 1]])
def test_invalid_utility_weights(weights: list) -> None:
    with pytest.raises(ValueError):
        eval.tailpass([0, 1], 2).linear(weights)


def test_invalid_sampling_and_functional_parameters() -> None:
    profile = eval.tailpass([0, 1], 2)
    for size in (0, 1, True, 2.5):
        with pytest.raises(ValueError):
            profile.sample(size)
    draws = profile.sample(5, rng=3)
    for power in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            profile.moment(power)
        with pytest.raises(ValueError):
            draws.power_mean(power)
    with pytest.raises(ValueError):
        profile.ci(method="predictive")
    with pytest.raises(ValueError):
        profile.linear_ci([0.5, 0.5], confidence=1)
    with pytest.raises(ValueError):
        draws.power_mean(2, aggregation="unknown")
    with pytest.raises(ValueError):
        draws.rollout(1)
    with pytest.raises(ValueError):
        draws.shortfall([0.5, 0.8])
    with pytest.raises(ValueError):
        draws.shortfall([0.8, 0.5], [1, 0])
    with pytest.raises(ValueError):
        draws.summary([1, 2])
    with pytest.raises(ValueError):
        eval.tailpass_empirical([0, 1], 3)
