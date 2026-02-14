import math

import numpy as np
import pytest
from scipy.special import comb
from scipy.stats import norm

from scorio import eval as scorio_eval


@pytest.fixture(scope="module")
def binary_ref(top_p_task_aime25: np.ndarray) -> np.ndarray:
    return top_p_task_aime25[0, :12, :20]


@pytest.fixture(scope="module")
def multiclass_ref(
    top_p_task_aime25: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = (top_p_task_aime25[1, :12, :20] + top_p_task_aime25[2, :12, :20]).astype(
        int, copy=False
    )
    w = np.array([0.0, 0.5, 1.0], dtype=float)
    R0 = (top_p_task_aime25[3, :12, :6] + top_p_task_aime25[4, :12, :6]).astype(
        int, copy=False
    )
    return R, w, R0


@pytest.fixture(scope="module")
def top_p_model_slice(top_p_task_aime25: np.ndarray) -> np.ndarray:
    return top_p_task_aime25[0, :10, :12]


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


def _bayes_reference(
    R: np.ndarray,
    w: np.ndarray,
    R0: np.ndarray | None = None,
) -> tuple[float, float]:
    Rm = np.asarray(R, dtype=int)
    wv = np.asarray(w, dtype=float)
    M, N = Rm.shape
    C = int(wv.size - 1)

    if R0 is None:
        R0m = np.zeros((M, 0), dtype=int)
    else:
        R0m = np.asarray(R0, dtype=int)
        if R0m.ndim == 1:
            R0m = R0m.reshape(M, -1)
        if R0m.shape[0] != M:
            raise ValueError(
                "R0 must have same row count as R for reference computation."
            )

    D = int(R0m.shape[1])
    T = float(1 + C + D + N)
    delta_w = wv - wv[0]

    mu_rows = np.empty(M, dtype=float)
    var_rows = np.empty(M, dtype=float)

    for row in range(M):
        nu = np.ones(C + 1, dtype=float)

        for value in Rm[row]:
            nu[int(value)] += 1.0
        for value in R0m[row]:
            nu[int(value)] += 1.0

        row_mean_component = float(np.dot(nu / T, delta_w))
        second_moment_component = float(np.dot(nu / T, delta_w**2))

        mu_rows[row] = row_mean_component
        var_rows[row] = max(0.0, second_moment_component - row_mean_component**2)

    mu = float(wv[0] + np.mean(mu_rows))
    sigma = float(math.sqrt(np.sum(var_rows) / (M**2 * (T + 1.0))))
    return mu, sigma


def _pass_at_k_reference(R: np.ndarray, k: int) -> float:
    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    denom = float(comb(N, k))
    values = np.empty(M, dtype=float)
    for row in range(M):
        nu = int(np.sum(Rm[row]))
        values[row] = 1.0 - float(comb(N - nu, k)) / denom
    return float(np.mean(values))


def _pass_hat_k_reference(R: np.ndarray, k: int) -> float:
    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    denom = float(comb(N, k))
    values = np.empty(M, dtype=float)
    for row in range(M):
        nu = int(np.sum(Rm[row]))
        values[row] = float(comb(nu, k)) / denom
    return float(np.mean(values))


def _g_pass_at_k_tau_reference(R: np.ndarray, k: int, tau: float) -> float:
    if tau <= 0.0:
        return _pass_at_k_reference(R, k)

    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    denom = float(comb(N, k))
    j0 = int(math.ceil(tau * k))

    values = np.empty(M, dtype=float)
    for row in range(M):
        nu = int(np.sum(Rm[row]))
        total = 0.0
        for j in range(j0, k + 1):
            total += float(comb(nu, j) * comb(N - nu, k - j)) / denom
        values[row] = total
    return float(np.mean(values))


def _mg_pass_at_k_reference(R: np.ndarray, k: int) -> float:
    Rm = np.asarray(R, dtype=int)
    M, N = Rm.shape
    denom = float(comb(N, k))
    majority = int(math.ceil(0.5 * k))
    if majority >= k:
        return 0.0

    values = np.empty(M, dtype=float)
    for row in range(M):
        nu = int(np.sum(Rm[row]))
        total = 0.0
        for j in range(majority + 1, k + 1):
            total += (j - majority) * float(comb(nu, j) * comb(N - nu, k - j)) / denom
        values[row] = (2.0 / k) * total
    return float(np.mean(values))


def test_bayes_multiclass_matches_closed_form_reference(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, w, R0 = multiclass_ref
    mu_prior, sigma_prior = scorio_eval.bayes(R, w, R0)
    exp_mu_prior, exp_sigma_prior = _bayes_reference(R, w, R0)

    mu_noprior, sigma_noprior = scorio_eval.bayes(R, w)
    exp_mu_noprior, exp_sigma_noprior = _bayes_reference(R, w)

    assert mu_prior == pytest.approx(exp_mu_prior)
    assert sigma_prior == pytest.approx(exp_sigma_prior)
    assert mu_noprior == pytest.approx(exp_mu_noprior)
    assert sigma_noprior == pytest.approx(exp_sigma_noprior)


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


def test_avg_binary_and_weighted_match_manual_formulas(
    binary_ref: np.ndarray,
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    a_binary, sigma_binary = scorio_eval.avg(binary_ref)
    assert a_binary == pytest.approx(float(np.mean(binary_ref)))
    assert sigma_binary >= 0.0

    R, w, _ = multiclass_ref
    a_weighted, sigma_weighted = scorio_eval.avg(R, w)
    assert a_weighted == pytest.approx(float(np.mean(w[R])))
    assert sigma_weighted >= 0.0

    _, sigma_bayes_weighted = scorio_eval.bayes(R, w)
    T = 1 + (w.size - 1) + R.shape[1]
    assert sigma_weighted == pytest.approx((T / R.shape[1]) * sigma_bayes_weighted)


def test_avg_requires_binary_when_weights_omitted(
    multiclass_ref: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    R, _, _ = multiclass_ref
    with pytest.raises(ValueError, match="Entries of R must be integers in \\[0, 1\\]"):
        scorio_eval.avg(R)


def test_avg_ci_matches_normal_interval_formula(binary_ref: np.ndarray) -> None:
    confidence = 0.8
    bounds = (0.0, 1.0)
    a, sigma, lo, hi = scorio_eval.avg_ci(
        binary_ref, confidence=confidence, bounds=bounds
    )
    exp_lo, exp_hi = _expected_normal_ci(a, sigma, confidence, bounds)

    assert lo == pytest.approx(exp_lo)
    assert hi == pytest.approx(exp_hi)
    assert lo <= a <= hi


def test_pass_point_metrics_match_closed_form_references(
    binary_ref: np.ndarray,
) -> None:
    k = 3
    assert scorio_eval.pass_at_k(binary_ref, k) == pytest.approx(
        _pass_at_k_reference(binary_ref, k)
    )
    assert scorio_eval.pass_hat_k(binary_ref, k) == pytest.approx(
        _pass_hat_k_reference(binary_ref, k)
    )
    assert scorio_eval.g_pass_at_k(binary_ref, k) == pytest.approx(
        _pass_hat_k_reference(binary_ref, k)
    )
    assert scorio_eval.g_pass_at_k_tau(binary_ref, k, 0.7) == pytest.approx(
        _g_pass_at_k_tau_reference(binary_ref, k, 0.7)
    )
    assert scorio_eval.mg_pass_at_k(binary_ref, k) == pytest.approx(
        _mg_pass_at_k_reference(binary_ref, k)
    )


def test_pass_family_monotonicity_and_bounds(binary_ref: np.ndarray) -> None:
    N = binary_ref.shape[1]
    k_values = list(range(1, min(N, 8) + 1))

    pass_vals = [scorio_eval.pass_at_k(binary_ref, k) for k in k_values]
    pass_hat_vals = [scorio_eval.pass_hat_k(binary_ref, k) for k in k_values]

    for idx in range(1, len(k_values)):
        assert pass_vals[idx] >= pass_vals[idx - 1]
        assert pass_hat_vals[idx] <= pass_hat_vals[idx - 1]

    for p, ph in zip(pass_vals, pass_hat_vals):
        assert p >= ph
        assert 0.0 <= ph <= 1.0
        assert 0.0 <= p <= 1.0


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
    np.testing.assert_allclose(
        scorio_eval.mg_pass_at_k_ci(binary_ref, 1), (0.0, 0.0, 0.0, 0.0)
    )


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


@pytest.mark.parametrize(
    "fn", [scorio_eval.g_pass_at_k_tau, scorio_eval.g_pass_at_k_tau_ci]
)
def test_g_pass_tau_invalid_tau_raises(binary_ref: np.ndarray, fn) -> None:
    with pytest.raises(ValueError, match="tau must be in \\[0, 1\\]"):
        fn(binary_ref, 2, tau=1.1)


def test_pass_family_rejects_non_binary_values(binary_ref: np.ndarray) -> None:
    R_bad = binary_ref.copy()
    R_bad[0, 0] = 2
    with pytest.raises(ValueError, match="Entries of R must be integers in \\[0, 1\\]"):
        scorio_eval.pass_at_k(R_bad, 1)


def test_eval_apis_are_invariant_to_question_and_trial_permutations(
    top_p_model_slice: np.ndarray,
) -> None:
    R = top_p_model_slice
    R_perm = R[::-1, :][:, ::-1]

    assert scorio_eval.avg(R)[0] == pytest.approx(scorio_eval.avg(R_perm)[0])
    assert scorio_eval.bayes(R)[0] == pytest.approx(scorio_eval.bayes(R_perm)[0])
    assert scorio_eval.pass_at_k(R, 3) == pytest.approx(
        scorio_eval.pass_at_k(R_perm, 3)
    )
    assert scorio_eval.pass_hat_k(R, 3) == pytest.approx(
        scorio_eval.pass_hat_k(R_perm, 3)
    )
    assert scorio_eval.g_pass_at_k_tau(R, 3, 0.7) == pytest.approx(
        scorio_eval.g_pass_at_k_tau(R_perm, 3, 0.7)
    )
    assert scorio_eval.mg_pass_at_k(R, 3) == pytest.approx(
        scorio_eval.mg_pass_at_k(R_perm, 3)
    )


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


def test_public_eval_api_exports_have_valid_smoke_calls(binary_ref: np.ndarray) -> None:
    api_calls = {
        "bayes": lambda: scorio_eval.bayes(binary_ref),
        "bayes_ci": lambda: scorio_eval.bayes_ci(binary_ref),
        "avg": lambda: scorio_eval.avg(binary_ref),
        "avg_ci": lambda: scorio_eval.avg_ci(binary_ref),
        "pass_at_k": lambda: scorio_eval.pass_at_k(binary_ref, 2),
        "pass_hat_k": lambda: scorio_eval.pass_hat_k(binary_ref, 2),
        "g_pass_at_k": lambda: scorio_eval.g_pass_at_k(binary_ref, 2),
        "g_pass_at_k_tau": lambda: scorio_eval.g_pass_at_k_tau(binary_ref, 2, tau=0.7),
        "mg_pass_at_k": lambda: scorio_eval.mg_pass_at_k(binary_ref, 2),
        "pass_at_k_ci": lambda: scorio_eval.pass_at_k_ci(binary_ref, 2),
        "pass_hat_k_ci": lambda: scorio_eval.pass_hat_k_ci(binary_ref, 2),
        "g_pass_at_k_ci": lambda: scorio_eval.g_pass_at_k_ci(binary_ref, 2),
        "g_pass_at_k_tau_ci": lambda: scorio_eval.g_pass_at_k_tau_ci(
            binary_ref, 2, tau=0.7
        ),
        "mg_pass_at_k_ci": lambda: scorio_eval.mg_pass_at_k_ci(binary_ref, 2),
    }

    assert set(api_calls) == set(scorio_eval.__all__)

    for name, fn in api_calls.items():
        out = fn()
        if name.endswith("_ci"):
            mu, sigma, lo, hi = out
            assert np.isfinite(mu)
            assert np.isfinite(sigma)
            assert np.isfinite(lo)
            assert np.isfinite(hi)
            assert sigma >= 0.0
            assert lo <= hi
            assert lo <= mu <= hi
            continue

        if name in {"bayes", "avg"}:
            mu, sigma = out
            assert np.isfinite(mu)
            assert np.isfinite(sigma)
            assert sigma >= 0.0
            continue

        assert np.isfinite(out)
        assert 0.0 <= out <= 1.0
