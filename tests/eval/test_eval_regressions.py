from __future__ import annotations

import warnings
from collections.abc import Callable
from itertools import combinations

import numpy as np
import pytest

from scorio import eval as scorio_eval
from scorio.eval.gpass import _tau_threshold

EvalCall = Callable[[np.ndarray], object]
KCall = Callable[[np.ndarray, float], object]
PriorCall = Callable[..., object]


_PUBLIC_R_CALLS: tuple[tuple[str, EvalCall], ...] = (
    ("bayes", lambda R: scorio_eval.bayes(R)),
    ("bayes_ci", lambda R: scorio_eval.bayes_ci(R)),
    ("avg", lambda R: scorio_eval.avg(R)),
    ("avg_ci", lambda R: scorio_eval.avg_ci(R)),
    ("pass_at_k", lambda R: scorio_eval.pass_at_k(R, 1)),
    ("pass_at_k_ci", lambda R: scorio_eval.pass_at_k_ci(R, 1)),
    ("pass_hat_k", lambda R: scorio_eval.pass_hat_k(R, 1)),
    ("pass_hat_k_ci", lambda R: scorio_eval.pass_hat_k_ci(R, 1)),
    ("g_pass_at_k", lambda R: scorio_eval.g_pass_at_k(R, 1)),
    ("g_pass_at_k_ci", lambda R: scorio_eval.g_pass_at_k_ci(R, 1)),
    ("g_pass_at_k_tau", lambda R: scorio_eval.g_pass_at_k_tau(R, 1, 0.5)),
    (
        "g_pass_at_k_tau_ci",
        lambda R: scorio_eval.g_pass_at_k_tau_ci(R, 1, 0.5),
    ),
    ("mg_pass_at_k", lambda R: scorio_eval.mg_pass_at_k(R, 1)),
    ("mg_pass_at_k_ci", lambda R: scorio_eval.mg_pass_at_k_ci(R, 1)),
    ("maj_at_k", lambda R: scorio_eval.maj_at_k(R, 1)),
    ("maj_at_k_ci", lambda R: scorio_eval.maj_at_k_ci(R, 1)),
    ("auc_at_k", lambda R: scorio_eval.auc_at_k(R, 1)),
    ("auc_at_k_ci", lambda R: scorio_eval.auc_at_k_ci(R, 1)),
    ("max_at_k", lambda R: scorio_eval.max_at_k(R, 1)),
    ("max_at_k_ci", lambda R: scorio_eval.max_at_k_ci(R, 1)),
    (
        "threshold_spectrum_at_k",
        lambda R: scorio_eval.threshold_spectrum_at_k(R, 1, [1.0]),
    ),
    (
        "threshold_spectrum_at_k_ci",
        lambda R: scorio_eval.threshold_spectrum_at_k_ci(R, 1, [1.0]),
    ),
    ("geom_at_k", lambda R: scorio_eval.geom_at_k(R, 1)),
    ("geom_at_k_ci", lambda R: scorio_eval.geom_at_k_ci(R, 1)),
    ("geom_ds_at_k", lambda R: scorio_eval.geom_ds_at_k(R, 1)),
    ("geom_ds_at_k_ci", lambda R: scorio_eval.geom_ds_at_k_ci(R, 1)),
    ("geo_spectrum_at_k", lambda R: scorio_eval.geo_spectrum_at_k(R, 1)),
    ("geo_spectrum_at_k_ci", lambda R: scorio_eval.geo_spectrum_at_k_ci(R, 1)),
    (
        "geo_spectrum_star_at_k",
        lambda R: scorio_eval.geo_spectrum_star_at_k(R, 1),
    ),
    (
        "geo_spectrum_star_at_k_ci",
        lambda R: scorio_eval.geo_spectrum_star_at_k_ci(R, 1),
    ),
)


_K_CALLS: tuple[tuple[str, KCall], ...] = (
    ("pass_at_k", lambda R, k: scorio_eval.pass_at_k(R, k)),
    ("pass_at_k_ci", lambda R, k: scorio_eval.pass_at_k_ci(R, k)),
    ("pass_hat_k", lambda R, k: scorio_eval.pass_hat_k(R, k)),
    ("pass_hat_k_ci", lambda R, k: scorio_eval.pass_hat_k_ci(R, k)),
    ("g_pass_at_k", lambda R, k: scorio_eval.g_pass_at_k(R, k)),
    ("g_pass_at_k_ci", lambda R, k: scorio_eval.g_pass_at_k_ci(R, k)),
    ("g_pass_at_k_tau", lambda R, k: scorio_eval.g_pass_at_k_tau(R, k, 0.5)),
    (
        "g_pass_at_k_tau_ci",
        lambda R, k: scorio_eval.g_pass_at_k_tau_ci(R, k, 0.5),
    ),
    ("mg_pass_at_k", lambda R, k: scorio_eval.mg_pass_at_k(R, k)),
    ("mg_pass_at_k_ci", lambda R, k: scorio_eval.mg_pass_at_k_ci(R, k)),
    ("maj_at_k", lambda R, k: scorio_eval.maj_at_k(R, k)),
    ("maj_at_k_ci", lambda R, k: scorio_eval.maj_at_k_ci(R, k)),
    ("auc_at_k", lambda R, k: scorio_eval.auc_at_k(R, k)),
    ("auc_at_k_ci", lambda R, k: scorio_eval.auc_at_k_ci(R, k)),
    ("max_at_k", lambda R, k: scorio_eval.max_at_k(R, k)),
    ("max_at_k_ci", lambda R, k: scorio_eval.max_at_k_ci(R, k)),
    (
        "threshold_spectrum_at_k",
        lambda R, k: scorio_eval.threshold_spectrum_at_k(R, k, [1.0]),
    ),
    (
        "threshold_spectrum_at_k_ci",
        lambda R, k: scorio_eval.threshold_spectrum_at_k_ci(R, k, [1.0]),
    ),
    ("geom_at_k", lambda R, k: scorio_eval.geom_at_k(R, k)),
    ("geom_at_k_ci", lambda R, k: scorio_eval.geom_at_k_ci(R, k)),
    ("geom_ds_at_k", lambda R, k: scorio_eval.geom_ds_at_k(R, k)),
    ("geom_ds_at_k_ci", lambda R, k: scorio_eval.geom_ds_at_k_ci(R, k)),
    ("geo_spectrum_at_k", lambda R, k: scorio_eval.geo_spectrum_at_k(R, k)),
    ("geo_spectrum_at_k_ci", lambda R, k: scorio_eval.geo_spectrum_at_k_ci(R, k)),
    (
        "geo_spectrum_star_at_k",
        lambda R, k: scorio_eval.geo_spectrum_star_at_k(R, k),
    ),
    (
        "geo_spectrum_star_at_k_ci",
        lambda R, k: scorio_eval.geo_spectrum_star_at_k_ci(R, k),
    ),
)


_BETA_PRIOR_CALLS: tuple[tuple[str, PriorCall], ...] = (
    ("pass_at_k_ci", lambda R, **kw: scorio_eval.pass_at_k_ci(R, 2, **kw)),
    (
        "pass_hat_k_ci",
        lambda R, **kw: scorio_eval.pass_hat_k_ci(R, 2, **kw),
    ),
    ("g_pass_at_k_ci", lambda R, **kw: scorio_eval.g_pass_at_k_ci(R, 2, **kw)),
    (
        "g_pass_at_k_tau_ci",
        lambda R, **kw: scorio_eval.g_pass_at_k_tau_ci(R, 2, 0.5, **kw),
    ),
    (
        "mg_pass_at_k_ci",
        lambda R, **kw: scorio_eval.mg_pass_at_k_ci(R, 2, **kw),
    ),
    ("maj_at_k_ci", lambda R, **kw: scorio_eval.maj_at_k_ci(R, 2, **kw)),
    ("auc_at_k_ci", lambda R, **kw: scorio_eval.auc_at_k_ci(R, 2, **kw)),
    (
        "threshold_spectrum_at_k_ci",
        lambda R, **kw: scorio_eval.threshold_spectrum_at_k_ci(R, 2, [1.0, 0.0], **kw),
    ),
    ("geom_at_k_ci", lambda R, **kw: scorio_eval.geom_at_k_ci(R, 2, **kw)),
    (
        "geom_ds_at_k_ci",
        lambda R, **kw: scorio_eval.geom_ds_at_k_ci(R, 2, **kw),
    ),
    (
        "geo_spectrum_at_k_ci",
        lambda R, **kw: scorio_eval.geo_spectrum_at_k_ci(R, 2, **kw),
    ),
    (
        "geo_spectrum_star_at_k_ci",
        lambda R, **kw: scorio_eval.geo_spectrum_star_at_k_ci(R, 2, **kw),
    ),
)


def _value_error_problems(
    calls: tuple[tuple[str, EvalCall], ...], R: np.ndarray
) -> list[str]:
    problems: list[str] = []
    for name, call in calls:
        with warnings.catch_warnings():
            # NaN/Inf must be rejected before NumPy attempts a lossy integer cast.
            warnings.simplefilter("error", RuntimeWarning)
            try:
                call(R)
            except ValueError:
                continue
            except Exception as exc:  # noqa: BLE001 - report the public behavior
                problems.append(f"{name}: raised {type(exc).__name__}, not ValueError")
            else:
                problems.append(f"{name}: accepted invalid R")
    return problems


@pytest.mark.parametrize(
    "invalid_entry",
    [
        pytest.param(0.5, id="fractional"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="positive-inf"),
        pytest.param(-np.inf, id="negative-inf"),
    ],
)
def test_public_eval_apis_reject_non_integer_or_nonfinite_outcomes_before_cast(
    invalid_entry: float,
) -> None:
    R = np.array([[invalid_entry, 1.0]], dtype=float)
    problems = _value_error_problems(_PUBLIC_R_CALLS, R)
    assert not problems, "\n".join(problems)


def test_public_eval_apis_require_at_least_one_question() -> None:
    R = np.empty((0, 2), dtype=int)
    problems = _value_error_problems(_PUBLIC_R_CALLS, R)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("fractional_k", [1.0, 1.5])
def test_public_k_apis_reject_fractional_k_consistently(
    fractional_k: float,
) -> None:
    R = np.array([[0, 1, 1]], dtype=int)
    problems: list[str] = []
    for name, call in _K_CALLS:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                call(R, fractional_k)
            except (TypeError, ValueError):
                continue
            except Exception as exc:  # noqa: BLE001 - report the public behavior
                problems.append(f"{name}: raised unexpected {type(exc).__name__}")
            else:
                problems.append(f"{name}: accepted fractional k")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("parameter", ["alpha0", "beta0"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="inf"),
    ],
)
def test_beta_prior_parameters_must_be_finite_and_positive(
    parameter: str, invalid_value: float
) -> None:
    R = np.array([[0, 1, 0]], dtype=int)
    problems: list[str] = []
    kwargs = {parameter: invalid_value}
    for name, call in _BETA_PRIOR_CALLS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                call(R, **kwargs)
            except ValueError:
                continue
            except Exception as exc:  # noqa: BLE001 - report the public behavior
                problems.append(f"{name}: raised {type(exc).__name__}, not ValueError")
            else:
                problems.append(f"{name}: accepted {parameter}={invalid_value}")
    assert not problems, "\n".join(problems)


@pytest.fixture(scope="module")
def large_binary_extremes() -> tuple[np.ndarray, int]:
    N = 2000
    k = 1000
    return np.vstack([np.ones(N, dtype=int), np.zeros(N, dtype=int)]), k


def test_large_finite_binary_point_metrics_stay_finite(
    large_binary_extremes: tuple[np.ndarray, int],
) -> None:
    R, k = large_binary_extremes
    scores = {
        "pass": scorio_eval.pass_at_k(R, k),
        "unanimous": scorio_eval.pass_hat_k(R, k),
        "g_pass": scorio_eval.g_pass_at_k(R, k),
        "threshold": scorio_eval.g_pass_at_k_tau(R, k, 0.75),
        "mg_pass": scorio_eval.mg_pass_at_k(R, k),
        "majority": scorio_eval.maj_at_k(R, k),
        "auc": scorio_eval.auc_at_k(R, k),
    }

    assert np.all(np.isfinite(list(scores.values()))), scores
    np.testing.assert_allclose(list(scores.values()), 0.5)


def test_large_finite_binary_max_equals_pass(
    large_binary_extremes: tuple[np.ndarray, int],
) -> None:
    R, k = large_binary_extremes
    max_score = scorio_eval.max_at_k(R, k)
    pass_score = scorio_eval.pass_at_k(R, k)
    assert np.isfinite(max_score)
    assert max_score == pytest.approx(pass_score)


def test_large_finite_geom_and_spectrum_scores_stay_finite(
    large_binary_extremes: tuple[np.ndarray, int],
) -> None:
    R, k = large_binary_extremes
    unanimous_weights = np.zeros(k, dtype=float)
    unanimous_weights[-1] = 1.0
    mg_weights = np.zeros(k, dtype=float)
    mg_weights[k // 2 :] = 2.0 / k

    scores = {
        "geom_at_k": scorio_eval.geom_at_k(R, k),
        "geom_ds_at_k": scorio_eval.geom_ds_at_k(R, k),
        "threshold_unanimous": scorio_eval.threshold_spectrum_at_k(
            R, k, unanimous_weights
        ),
        "threshold_mg": scorio_eval.threshold_spectrum_at_k(R, k, mg_weights),
        "geo_spectrum_unanimous": scorio_eval.geo_spectrum_at_k(
            R, k, weights=unanimous_weights
        ),
        "geo_spectrum_default": scorio_eval.geo_spectrum_at_k(R, k),
        "geo_spectrum_star": scorio_eval.geo_spectrum_star_at_k(R, k),
    }

    assert np.all(np.isfinite(list(scores.values()))), scores
    np.testing.assert_allclose(list(scores.values()), 0.5)


def test_geo_spectrum_star_validates_k_before_allocating_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scorio.eval.geom as geom_module

    def fail_if_called(k: int) -> np.ndarray:
        raise AssertionError(f"weights allocated for invalid k={k}")

    monkeypatch.setattr(geom_module, "_mg_spectrum_weights", fail_if_called)
    with pytest.raises(ValueError, match="1 <= k <= N"):
        scorio_eval.geo_spectrum_star_at_k(np.array([[0, 1]]), 1_000_000)


def test_large_k_pass_threshold_ci_identity() -> None:
    # With w_1=1 and all other weights zero, the spectrum is exactly Pass@k.
    k = 517
    R = np.zeros((1, k), dtype=int)
    pass_weights = np.zeros(k, dtype=float)
    pass_weights[0] = 1.0

    spectrum_ci = scorio_eval.threshold_spectrum_at_k_ci(R, k, pass_weights)
    pass_ci = scorio_eval.pass_at_k_ci(R, k)

    np.testing.assert_allclose(spectrum_ci, pass_ci, rtol=1e-10, atol=1e-12)


def test_small_finite_binary_identities_characterize_expected_semantics() -> None:
    N = 8
    k = 4
    R = np.vstack([np.ones(N, dtype=int), np.zeros(N, dtype=int)])
    unanimous_weights = np.zeros(k, dtype=float)
    unanimous_weights[-1] = 1.0

    pass_score = scorio_eval.pass_at_k(R, k)
    assert scorio_eval.auc_at_k(R, k) == pytest.approx(0.5)
    assert scorio_eval.max_at_k(R, k) == pytest.approx(pass_score)
    assert scorio_eval.geom_at_k(R, k) == pytest.approx(0.5)
    assert scorio_eval.geom_ds_at_k(R, k) == pytest.approx(0.5)
    assert scorio_eval.threshold_spectrum_at_k(
        R, k, unanimous_weights
    ) == pytest.approx(0.5)
    assert scorio_eval.geo_spectrum_at_k(
        R, k, weights=unanimous_weights
    ) == pytest.approx(0.5)


def test_small_k_pass_threshold_ci_identity() -> None:
    R = np.array([[0, 1, 0, 1, 1], [1, 1, 0, 1, 1]], dtype=int)
    k = 3
    pass_weights = np.zeros(k, dtype=float)
    pass_weights[0] = 1.0

    np.testing.assert_allclose(
        scorio_eval.threshold_spectrum_at_k_ci(R, k, pass_weights),
        scorio_eval.pass_at_k_ci(R, k),
        rtol=1e-12,
        atol=1e-12,
    )


def test_questionwise_and_dataset_geom_aggregation_remain_distinct() -> None:
    R = np.array([[1, 0, 0], [1, 1, 1]], dtype=int)

    assert scorio_eval.geom_at_k(R, 2) == pytest.approx(0.5)
    assert scorio_eval.geom_ds_at_k(R, 2) == pytest.approx(np.sqrt(5.0 / 12.0))


def test_clipped_geom_interval_cannot_invert_when_mean_exceeds_bounds() -> None:
    R = np.tile([1, 1, 1, 0], (100, 1))

    mu, sigma, lower, upper = scorio_eval.geom_at_k_ci(
        R,
        2,
        pass_power=0.5,
        unanimous_power=-0.5,
        bounds=(0.0, 1.0),
    )

    assert mu > 1.0
    assert sigma > 0.0
    assert (lower, upper) == pytest.approx((1.0, 1.0))


def test_threshold_spectrum_matches_literal_subset_enumeration() -> None:
    R = np.array(
        [
            [0, 1, 1, 0, 1, 0],
            [1, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=int,
    )
    k = 3
    weights = np.array([0.2, 0.1, 0.4], dtype=float)
    levels = np.concatenate(([0.0], np.cumsum(weights)))

    subsets = tuple(combinations(range(R.shape[1]), k))
    row_scores = []
    for row in R:
        credits = [levels[int(np.sum(row[list(indices)]))] for indices in subsets]
        row_scores.append(float(np.mean(credits)))
    expected = float(np.mean(row_scores))

    assert scorio_eval.threshold_spectrum_at_k(R, k, weights) == pytest.approx(expected)


def _count_ladder(k: int) -> np.ndarray:
    """Rows with exactly 0, 1, ..., k successes out of ``N = k`` trials."""
    return (np.arange(k)[None, :] < np.arange(k + 1)[:, None]).astype(int)


# (j, k) pairs where ``(j / k) * k`` rounds above ``j`` in IEEE doubles, so a
# bare ``ceil(tau * k)`` asks for one success too many.
_WHOLE_TAU_K_ROUNDING_UP = ((7, 25), (14, 25), (15, 29), (29, 35), (21, 38))


@pytest.mark.parametrize(("j", "k"), _WHOLE_TAU_K_ROUNDING_UP)
def test_g_pass_tau_threshold_is_exact_when_tau_k_is_whole(j: int, k: int) -> None:
    exactly_j = np.array([[1] * j + [0] * (k - j)], dtype=int)
    one_short = np.array([[1] * (j - 1) + [0] * (k - j + 1)], dtype=int)

    assert scorio_eval.g_pass_at_k_tau(exactly_j, k, j / k) == pytest.approx(1.0)
    assert scorio_eval.g_pass_at_k_tau(one_short, k, j / k) == pytest.approx(0.0)

    # The posterior target must use the same threshold as tau just below j/k.
    R = _count_ladder(k)
    assert scorio_eval.g_pass_at_k_tau_ci(R, k, j / k) == pytest.approx(
        scorio_eval.g_pass_at_k_tau_ci(R, k, (j - 0.5) / k)
    )


@pytest.mark.parametrize(
    ("j", "k"), ((1, 3), (7, 25), (15, 29), (7, 100), (29, 100), (24, 25))
)
def test_g_pass_tau_preserves_adjacent_thresholds(j: int, k: int) -> None:
    exactly_j = np.array([[1] * j + [0] * (k - j)], dtype=int)
    boundary = j / k
    lower_ci = scorio_eval.g_pass_at_k_tau_ci(exactly_j, k, (j - 0.5) / k)
    upper_ci = scorio_eval.g_pass_at_k_tau_ci(exactly_j, k, (j + 0.5) / k)

    for tau in (np.nextafter(boundary, 0.0), boundary):
        assert scorio_eval.g_pass_at_k_tau(exactly_j, k, tau) == pytest.approx(1.0)
        assert scorio_eval.g_pass_at_k_tau_ci(exactly_j, k, tau) == pytest.approx(
            lower_ci
        )
    for tau in (np.nextafter(boundary, 1.0), boundary + 2e-11):
        assert scorio_eval.g_pass_at_k_tau(exactly_j, k, tau) == pytest.approx(0.0)
        assert scorio_eval.g_pass_at_k_tau_ci(exactly_j, k, tau) == pytest.approx(
            upper_ci
        )


@pytest.mark.parametrize(
    ("j", "k"), ((25_000_007, 50_000_003), (50_000_006, 100_000_007))
)
def test_g_pass_tau_threshold_at_large_sample_counts(j: int, k: int) -> None:
    # Exercise the threshold directly without allocating an enormous outcome bank.
    # At these grid points the multiplication error exceeds the old 1e-9 offset.
    boundary = j / k
    assert _tau_threshold(np.nextafter(boundary, 0.0), k) == j
    assert _tau_threshold(boundary, k) == j
    assert _tau_threshold(np.nextafter(boundary, 1.0), k) == j + 1


def test_g_pass_tau_threshold_on_every_j_over_k_grid_point() -> None:
    for k in range(1, 65):
        R = _count_ladder(k)
        for j in range(1, k + 1):
            # Rows with at least j successes pass with certainty; others fail.
            expected = (k - j + 1) / (k + 1)
            assert scorio_eval.g_pass_at_k_tau(R, k, j / k) == pytest.approx(
                expected, abs=1e-12
            ), (j, k)


def test_g_pass_tau_at_strict_majority_matches_maj_at_k() -> None:
    # Maj@k is G-Pass@k_tau at tau = (floor(k/2) + 1) / k; the JavaScript and
    # Julia ports compute it that way. (k // 2 + 1) / k * k rounds up for
    # k = 29, 52, 56, 58, 60, ...
    for k in range(1, 65):
        R = _count_ladder(k)
        tau = (k // 2 + 1) / k
        assert scorio_eval.g_pass_at_k_tau(R, k, tau) == pytest.approx(
            scorio_eval.maj_at_k(R, k), abs=1e-12
        ), k
    for k in (29, 52):
        R = _count_ladder(k)
        tau = (k // 2 + 1) / k
        assert scorio_eval.g_pass_at_k_tau_ci(R, k, tau) == pytest.approx(
            scorio_eval.maj_at_k_ci(R, k)
        ), k


@pytest.mark.parametrize("k", [25, 38, 41])
def test_mg_pass_equals_mean_of_g_pass_on_upper_tau_grid(k: int) -> None:
    rng = np.random.default_rng(k)
    R = rng.integers(0, 2, size=(20, 41))
    majority = -(-k // 2)
    grid_mean = (2.0 / k) * sum(
        scorio_eval.g_pass_at_k_tau(R, k, i / k) for i in range(majority + 1, k + 1)
    )

    assert scorio_eval.mg_pass_at_k(R, k) == pytest.approx(grid_mean, abs=1e-12)
