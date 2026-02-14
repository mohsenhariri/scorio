from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from scorio import eval
from scorio.rank import eval_ranking
from scorio.utils import rank_scores


ROOT = Path(__file__).resolve().parents[2]
TOP_P_PATH = ROOT / "notebooks" / "R_top_p.npz"
GREEDY_PATH = ROOT / "notebooks" / "R_greedy.npz"


def _load_task(path: Path, task: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        return data[task].astype(int, copy=False)


def _expected_scores(
    R: np.ndarray,
    metric_fn,
    *args,
    **kwargs,
) -> np.ndarray:
    return np.array(
        [metric_fn(R[model], *args, **kwargs) for model in range(R.shape[0])],
        dtype=float,
    )


@pytest.fixture(scope="module")
def top_p_subset() -> np.ndarray:
    # Keep tests fast while preserving the same (L, M, N) contract as README.
    R = _load_task(TOP_P_PATH, "aime25")
    return R[:6, :10, :12]


@pytest.fixture(scope="module")
def greedy_subset() -> np.ndarray:
    R = _load_task(GREEDY_PATH, "aime25")
    return R[:6, :10, :]


def test_avg_wrapper_matches_eval_scores_and_ranks(top_p_subset: np.ndarray) -> None:
    ranking, scores = eval_ranking.avg(top_p_subset, method="dense", return_scores=True)

    expected_scores = np.array(
        [eval.avg(top_p_subset[model])[0] for model in range(top_p_subset.shape[0])],
        dtype=float,
    )
    expected_ranking = rank_scores(expected_scores)["dense"]

    np.testing.assert_allclose(scores, expected_scores)
    np.testing.assert_allclose(ranking, expected_ranking)


def test_bayes_wrapper_matches_eval_scores_and_ranks(top_p_subset: np.ndarray) -> None:
    ranking, scores = eval_ranking.bayes(
        top_p_subset,
        method="competition_max",
        return_scores=True,
    )

    expected_scores = np.array(
        [eval.bayes(top_p_subset[model])[0] for model in range(top_p_subset.shape[0])],
        dtype=float,
    )
    expected_ranking = rank_scores(expected_scores)["competition_max"]

    np.testing.assert_allclose(scores, expected_scores)
    np.testing.assert_allclose(ranking, expected_ranking)


def test_bayes_quantile_uses_mu_plus_z_sigma(top_p_subset: np.ndarray) -> None:
    quantile = 0.1
    z = float(norm.ppf(quantile))

    _, scores = eval_ranking.bayes(top_p_subset, quantile=quantile, return_scores=True)
    expected_scores = np.empty(top_p_subset.shape[0], dtype=float)
    for model in range(top_p_subset.shape[0]):
        mu, sigma = eval.bayes(top_p_subset[model])
        expected_scores[model] = mu + z * sigma

    np.testing.assert_allclose(scores, expected_scores)


def test_bayes_R0_shared_and_per_model(top_p_subset: np.ndarray) -> None:
    shared_R0 = top_p_subset[0, :, :4]
    ranking_shared, scores_shared = eval_ranking.bayes(
        top_p_subset,
        R0=shared_R0,
        return_scores=True,
    )
    expected_shared = np.array(
        [
            eval.bayes(top_p_subset[model], R0=shared_R0)[0]
            for model in range(top_p_subset.shape[0])
        ],
        dtype=float,
    )
    expected_rank_shared = rank_scores(expected_shared)["competition"]
    np.testing.assert_allclose(scores_shared, expected_shared)
    np.testing.assert_allclose(ranking_shared, expected_rank_shared)

    per_model_R0 = top_p_subset[:, :, :4]
    ranking_per_model, scores_per_model = eval_ranking.bayes(
        top_p_subset,
        R0=per_model_R0,
        return_scores=True,
    )
    expected_per_model = np.array(
        [
            eval.bayes(top_p_subset[model], R0=per_model_R0[model])[0]
            for model in range(top_p_subset.shape[0])
        ],
        dtype=float,
    )
    expected_rank_per_model = rank_scores(expected_per_model)["competition"]
    np.testing.assert_allclose(scores_per_model, expected_per_model)
    np.testing.assert_allclose(ranking_per_model, expected_rank_per_model)


def test_bayes_validation_errors(top_p_subset: np.ndarray) -> None:
    L, M, _ = top_p_subset.shape

    with pytest.raises(ValueError, match="quantile must be in \\[0, 1\\]"):
        eval_ranking.bayes(top_p_subset, quantile=-0.1)

    with pytest.raises(ValueError, match="quantile must be in \\[0, 1\\]"):
        eval_ranking.bayes(top_p_subset, quantile=1.1)

    with pytest.raises(ValueError, match="Shared R0 must have shape"):
        eval_ranking.bayes(top_p_subset, R0=np.zeros((M - 1, 3), dtype=int))

    with pytest.raises(ValueError, match="Model-specific R0 must have shape"):
        eval_ranking.bayes(top_p_subset, R0=np.zeros((L - 1, M, 3), dtype=int))

    with pytest.raises(ValueError, match="R0 must be shape"):
        eval_ranking.bayes(top_p_subset, R0=np.zeros((3,), dtype=int))


@pytest.mark.parametrize(
    ("rank_fn", "eval_fn", "kwargs", "method"),
    [
        (eval_ranking.pass_at_k, eval.pass_at_k, {"k": 3}, "competition"),
        (eval_ranking.pass_hat_k, eval.pass_hat_k, {"k": 3}, "avg"),
        (
            eval_ranking.g_pass_at_k_tau,
            eval.g_pass_at_k_tau,
            {"k": 3, "tau": 0.7},
            "dense",
        ),
        (eval_ranking.mg_pass_at_k, eval.mg_pass_at_k, {"k": 3}, "competition_max"),
    ],
)
def test_pass_family_wrappers_match_eval_scores_and_ranks(
    top_p_subset: np.ndarray,
    rank_fn,
    eval_fn,
    kwargs: dict[str, float | int],
    method: str,
) -> None:
    ranking, scores = rank_fn(top_p_subset, method=method, return_scores=True, **kwargs)
    expected_scores = _expected_scores(top_p_subset, eval_fn, **kwargs)
    expected_ranking = rank_scores(expected_scores)[method]

    np.testing.assert_allclose(scores, expected_scores)
    np.testing.assert_allclose(ranking, expected_ranking)


def test_g_pass_tau_edge_equivalences(top_p_subset: np.ndarray) -> None:
    _, scores_tau0 = eval_ranking.g_pass_at_k_tau(
        top_p_subset, k=3, tau=0.0, return_scores=True
    )
    _, scores_pass = eval_ranking.pass_at_k(top_p_subset, k=3, return_scores=True)
    np.testing.assert_allclose(scores_tau0, scores_pass)

    _, scores_tau1 = eval_ranking.g_pass_at_k_tau(
        top_p_subset, k=3, tau=1.0, return_scores=True
    )
    _, scores_hat = eval_ranking.pass_hat_k(top_p_subset, k=3, return_scores=True)
    np.testing.assert_allclose(scores_tau1, scores_hat)


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (eval_ranking.pass_at_k, {"k": 0}),
        (eval_ranking.pass_hat_k, {"k": 0}),
        (eval_ranking.g_pass_at_k_tau, {"k": 0, "tau": 0.5}),
        (eval_ranking.mg_pass_at_k, {"k": 0}),
    ],
)
def test_pass_family_invalid_k_raises(
    top_p_subset: np.ndarray, fn, kwargs: dict[str, float | int]
) -> None:
    with pytest.raises(ValueError, match="k must satisfy 1 <= k <= N"):
        fn(top_p_subset, **kwargs)


def test_g_pass_tau_invalid_tau_raises(top_p_subset: np.ndarray) -> None:
    with pytest.raises(ValueError, match="tau must be in \\[0, 1\\]"):
        eval_ranking.g_pass_at_k_tau(top_p_subset, k=3, tau=1.1)


def test_2d_input_promotion_matches_3d(greedy_subset: np.ndarray) -> None:
    matrix = greedy_subset[:, :, 0]

    ranking_3d, scores_3d = eval_ranking.pass_at_k(greedy_subset, k=1, return_scores=True)
    ranking_2d, scores_2d = eval_ranking.pass_at_k(matrix, k=1, return_scores=True)

    np.testing.assert_allclose(scores_2d, scores_3d)
    np.testing.assert_allclose(ranking_2d, ranking_3d)
