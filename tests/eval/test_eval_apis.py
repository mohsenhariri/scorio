from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from scorio import eval as scorio_eval

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "tests" / "data"
TOP_P_PATH = DATA_DIR / "R_top_p.npz"


@pytest.fixture(scope="module")
def binary_ref() -> np.ndarray:
    return np.array(
        [
            [0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1],
        ],
        dtype=int,
    )


@pytest.fixture(scope="module")
def multiclass_ref() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.array(
        [
            [0, 1, 2, 2, 1],
            [1, 1, 0, 2, 2],
        ],
        dtype=int,
    )
    w = np.array([0.0, 0.5, 1.0], dtype=float)
    R0 = np.array(
        [
            [0, 2],
            [1, 2],
        ],
        dtype=int,
    )
    return R, w, R0


@pytest.fixture(scope="module")
def top_p_model_slice() -> np.ndarray:
    with np.load(TOP_P_PATH, allow_pickle=True) as data:
        R = data["aime25"].astype(int, copy=False)
    return R[0, :10, :12]


def _expected_normal_ci(
    mu: float,
    sigma: float,
    confidence: float,
    bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    z = float(norm.ppf(0.5 + confidence / 2.0))
    lo = mu - z * sigma
    hi = mu + z * sigma
    if bounds is not None:
        lo = max(lo, bounds[0])
        hi = min(hi, bounds[1])
    return lo, hi


def test_bayes_multiclass_matches_documented_values(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, w, R0 = multiclass_ref
    mu_prior, sigma_prior = scorio_eval.bayes(R, w, R0)
    mu_noprior, sigma_noprior = scorio_eval.bayes(R, w)

    assert mu_prior == pytest.approx(0.575)
    assert sigma_prior == pytest.approx(0.08427498280790524)
    assert mu_noprior == pytest.approx(0.5625)
    assert sigma_noprior == pytest.approx(0.0919975090242484)


def test_bayes_binary_default_weights_equal_explicit(binary_ref: np.ndarray) -> None:
    mu_auto, sigma_auto = scorio_eval.bayes(binary_ref)
    mu_explicit, sigma_explicit = scorio_eval.bayes(
        binary_ref, w=np.array([0.0, 1.0], dtype=float)
    )

    assert mu_auto == pytest.approx(mu_explicit)
    assert sigma_auto == pytest.approx(sigma_explicit)


def test_bayes_requires_weights_for_multiclass(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, _, _ = multiclass_ref
    with pytest.raises(ValueError, match="must be provided"):
        scorio_eval.bayes(R)


def test_bayes_validates_R0_row_count(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, w, _ = multiclass_ref
    bad_R0 = np.zeros((R.shape[0] + 1, 2), dtype=int)
    with pytest.raises(ValueError, match="same number of rows"):
        scorio_eval.bayes(R, w=w, R0=bad_R0)


def test_bayes_ci_matches_normal_interval_formula(binary_ref: np.ndarray) -> None:
    confidence = 0.9
    bounds = (0.0, 1.0)
    mu, sigma, lo, hi = scorio_eval.bayes_ci(
        binary_ref,
        confidence=confidence,
        bounds=bounds,
    )
    exp_lo, exp_hi = _expected_normal_ci(mu, sigma, confidence, bounds)

    assert lo == pytest.approx(exp_lo)
    assert hi == pytest.approx(exp_hi)
    assert lo <= mu <= hi


def test_avg_binary_and_weighted_match_documented_values(
    binary_ref: np.ndarray,
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    a_binary, s_binary = scorio_eval.avg(binary_ref)
    assert a_binary == pytest.approx(0.7)
    assert s_binary == pytest.approx(0.16583123951776998)

    R, w, _ = multiclass_ref
    a_weighted, s_weighted = scorio_eval.avg(R, w)
    assert a_weighted == pytest.approx(0.6)
    assert s_weighted == pytest.approx(0.14719601443879746)


def test_avg_requires_binary_when_weights_omitted(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, _, _ = multiclass_ref
    with pytest.raises(ValueError, match="Entries of R must be integers in \\[0, 1\\]"):
        scorio_eval.avg(R)


def test_avg_ci_matches_normal_interval_formula(binary_ref: np.ndarray) -> None:
    confidence = 0.8
    bounds = (0.0, 1.0)
    a, sigma, lo, hi = scorio_eval.avg_ci(binary_ref, confidence=confidence, bounds=bounds)
    exp_lo, exp_hi = _expected_normal_ci(a, sigma, confidence, bounds)

    assert lo == pytest.approx(exp_lo)
    assert hi == pytest.approx(exp_hi)
    assert lo <= a <= hi


def test_pass_point_metrics_match_documented_values(binary_ref: np.ndarray) -> None:
    assert scorio_eval.pass_at_k(binary_ref, 1) == pytest.approx(0.7)
    assert scorio_eval.pass_at_k(binary_ref, 2) == pytest.approx(0.95)
    assert scorio_eval.pass_hat_k(binary_ref, 1) == pytest.approx(0.7)
    assert scorio_eval.pass_hat_k(binary_ref, 2) == pytest.approx(0.45)
    assert scorio_eval.g_pass_at_k(binary_ref, 2) == pytest.approx(0.45)
    assert scorio_eval.g_pass_at_k_tau(binary_ref, 2, 0.5) == pytest.approx(0.95)
    assert scorio_eval.g_pass_at_k_tau(binary_ref, 2, 1.0) == pytest.approx(0.45)
    assert scorio_eval.mg_pass_at_k(binary_ref, 2) == pytest.approx(0.45)
    assert scorio_eval.mg_pass_at_k(binary_ref, 3) == pytest.approx(1.0 / 6.0)


def test_pass_family_ci_match_documented_values(binary_ref: np.ndarray) -> None:
    np.testing.assert_allclose(
        scorio_eval.pass_at_k_ci(binary_ref, 1),
        (0.6428571428571428, 0.11845088536983554, 0.41069767359538273, 0.8750166121189029),
    )
    np.testing.assert_allclose(
        scorio_eval.pass_at_k_ci(binary_ref, 2),
        (0.8392857142857142, 0.09726270618076298, 0.6486543131325174, 1.0),
    )
    np.testing.assert_allclose(
        scorio_eval.pass_hat_k_ci(binary_ref, 2),
        (
            0.44642857142857134,
            0.14616701378343672,
            0.15994648868526573,
            0.732910654171877,
        ),
    )
    np.testing.assert_allclose(
        scorio_eval.g_pass_at_k_tau_ci(binary_ref, 2, 1.0),
        (
            0.44642857142857134,
            0.14616701378343672,
            0.15994648868526573,
            0.732910654171877,
        ),
    )
    np.testing.assert_allclose(
        scorio_eval.mg_pass_at_k_ci(binary_ref, 3),
        (
            0.21825396825396823,
            0.09881597074420659,
            0.024578224497959683,
            0.41192971200997675,
        ),
    )


def test_pass_aliases_and_tau_edge_equivalences(binary_ref: np.ndarray) -> None:
    k = 3
    assert scorio_eval.g_pass_at_k(binary_ref, k) == pytest.approx(
        scorio_eval.pass_hat_k(binary_ref, k)
    )

    np.testing.assert_allclose(
        scorio_eval.g_pass_at_k_ci(binary_ref, k),
        scorio_eval.pass_hat_k_ci(binary_ref, k),
    )

    assert scorio_eval.g_pass_at_k_tau(binary_ref, k, tau=0.0) == pytest.approx(
        scorio_eval.pass_at_k(binary_ref, k)
    )
    assert scorio_eval.g_pass_at_k_tau(binary_ref, k, tau=1.0) == pytest.approx(
        scorio_eval.pass_hat_k(binary_ref, k)
    )

    np.testing.assert_allclose(
        scorio_eval.g_pass_at_k_tau_ci(binary_ref, k, tau=0.0),
        scorio_eval.pass_at_k_ci(binary_ref, k),
    )
    np.testing.assert_allclose(
        scorio_eval.g_pass_at_k_tau_ci(binary_ref, k, tau=1.0),
        scorio_eval.pass_hat_k_ci(binary_ref, k),
    )


def test_pass_mg_k1_edge_case(binary_ref: np.ndarray) -> None:
    assert scorio_eval.mg_pass_at_k(binary_ref, 1) == pytest.approx(0.0)
    np.testing.assert_allclose(scorio_eval.mg_pass_at_k_ci(binary_ref, 1), (0.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "fn",
    [
        scorio_eval.pass_at_k,
        scorio_eval.pass_hat_k,
        scorio_eval.g_pass_at_k,
        scorio_eval.mg_pass_at_k,
        scorio_eval.pass_at_k_ci,
        scorio_eval.pass_hat_k_ci,
        scorio_eval.g_pass_at_k_ci,
        scorio_eval.mg_pass_at_k_ci,
    ],
)
def test_pass_family_invalid_k_raises(binary_ref: np.ndarray, fn) -> None:
    with pytest.raises(ValueError, match="k must satisfy 1 <= k <= N"):
        fn(binary_ref, 0)


@pytest.mark.parametrize("fn", [scorio_eval.g_pass_at_k_tau, scorio_eval.g_pass_at_k_tau_ci])
def test_g_pass_tau_invalid_tau_raises(binary_ref: np.ndarray, fn) -> None:
    with pytest.raises(ValueError, match="tau must be in \\[0, 1\\]"):
        fn(binary_ref, 2, tau=1.1)


def test_pass_family_rejects_non_binary_values() -> None:
    R = np.array([[0, 1, 2], [1, 0, 1]], dtype=int)
    with pytest.raises(ValueError, match="Entries of R must be integers in \\[0, 1\\]"):
        scorio_eval.pass_at_k(R, 1)


def test_eval_apis_on_simulation_dataset_slice(top_p_model_slice: np.ndarray) -> None:
    R = top_p_model_slice

    a, a_sigma = scorio_eval.avg(R)
    assert 0.0 <= a <= 1.0
    assert a_sigma >= 0.0
    assert scorio_eval.pass_at_k(R, 1) == pytest.approx(a)

    b_mu, b_sigma = scorio_eval.bayes(R)
    assert 0.0 <= b_mu <= 1.0
    assert b_sigma >= 0.0

    p1 = scorio_eval.pass_at_k(R, 3)
    ph = scorio_eval.pass_hat_k(R, 3)
    gt = scorio_eval.g_pass_at_k_tau(R, 3, 0.7)
    mg = scorio_eval.mg_pass_at_k(R, 3)
    assert p1 >= gt >= ph
    assert 0.0 <= mg <= 1.0

    ci_outputs = [
        scorio_eval.bayes_ci(R),
        scorio_eval.avg_ci(R),
        scorio_eval.pass_at_k_ci(R, 3),
        scorio_eval.pass_hat_k_ci(R, 3),
        scorio_eval.g_pass_at_k_ci(R, 3),
        scorio_eval.g_pass_at_k_tau_ci(R, 3, 0.7),
        scorio_eval.mg_pass_at_k_ci(R, 3),
    ]
    for mu, sigma, lo, hi in ci_outputs:
        assert np.isfinite(mu)
        assert np.isfinite(sigma)
        assert np.isfinite(lo)
        assert np.isfinite(hi)
        assert sigma >= 0.0
        assert lo <= hi
        assert lo <= mu <= hi
