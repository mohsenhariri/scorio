r"""
HodgeRank: statistical ranking via combinatorial Hodge theory.

Based on:
    Jiang, Lim, Yao, Ye (2009). "Statistical ranking and combinatorial Hodge theory".

This implements the core :math:`\ell_2` HodgeRank estimator: given a (possibly
incomplete) pairwise comparison graph with an observed skew-symmetric edge flow
:math:`\bar{Y}` (average pairwise rankings) and symmetric edge weights
:math:`w_{ij}`, it solves

.. math::
    s^{\star}
    =
    \arg\min_s \lVert \operatorname{grad}(s) - \bar{Y} \rVert_{2,w}^2

The normal equations reduce to a weighted graph Laplacian system:

.. math::
    s^{\star} = -\Delta_0^{\dagger}\operatorname{div}(\bar{Y})

where :math:`\Delta_0` is the weighted graph Laplacian and :math:`\dagger`
denotes the Moore-Penrose pseudoinverse.

In scorio's evaluation setting, the input is a binary tensor :math:`R` of shape
:math:`(L, M, N)` where :math:`R_{lmn} = 1` means model :math:`l` solved
question :math:`m` on trial :math:`n`. We form pairwise statistics
:math:`\bar{Y}_{ij}` using the paper's binary comparison statistic.
"""

import numpy as np

from scorio.utils import rank_scores

from ._base import build_pairwise_counts, validate_input


def _pairwise_flow_binary(wins: np.ndarray, ties: np.ndarray) -> np.ndarray:
    """
    Paper (Section 2.2.1), binary comparison statistic:

        Ȳ_ij = Pr{j > i} - Pr{i > j}

    For scorio's binary outcomes, we treat:
        j > i  ⇔  j correct and i incorrect.
    """
    total = wins + wins.T + ties
    Y = np.zeros_like(wins, dtype=float)
    mask = total > 0
    # wins[j,i] = wins.T[i,j]
    Y[mask] = (wins.T[mask] - wins[mask]) / total[mask]
    np.fill_diagonal(Y, 0.0)
    return Y


def _pairwise_flow_log_odds(
    wins: np.ndarray, ties: np.ndarray, *, epsilon: float = 0.5
) -> np.ndarray:
    """
    Paper (Section 2.2.1), logarithmic odds ratio statistic:

        Ȳ_ij = log( Pr{j >= i} / Pr{j <= i} )

    For binary outcomes, j >= i fails only when (i=1, j=0).
    """
    epsilon = float(epsilon)
    if (not np.isfinite(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be > 0 for log-odds smoothing")

    total = wins + wins.T + ties
    Y = np.zeros_like(wins, dtype=float)

    # Pr{j >= i} = 1 - Pr{i > j} = (total - wins[i,j]) / total
    # Pr{j <= i} = 1 - Pr{j > i} = (total - wins[j,i]) / total
    # => ratio = (total - wins[i,j]) / (total - wins[j,i])
    mask = total > 0
    numerator = (total - wins + epsilon)[mask]
    denom = (total - wins.T + epsilon)[mask]
    Y[mask] = np.log(numerator / denom)

    np.fill_diagonal(Y, 0.0)
    return Y


def _weights_from_counts(
    wins: np.ndarray,
    ties: np.ndarray,
    weight_method: str,
) -> np.ndarray:
    total = wins + wins.T + ties

    weight_method = str(weight_method)
    if weight_method == "total":
        w = total.astype(float)
    elif weight_method == "decisive":
        w = (wins + wins.T).astype(float)
    elif weight_method == "uniform":
        w = (total > 0).astype(float)
    else:
        raise ValueError('weight_method must be one of: "total", "decisive", "uniform"')

    np.fill_diagonal(w, 0.0)
    return w


def _laplacian_from_weights(w: np.ndarray) -> np.ndarray:
    L = -w.astype(float).copy()
    np.fill_diagonal(L, 0.0)
    np.fill_diagonal(L, -L.sum(axis=1))
    return L


def _divergence(w: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # (div Y)(i) = Σ_j w_ij Y_ij
    return (w * Y).sum(axis=1)


def _grad(scores: np.ndarray) -> np.ndarray:
    # (grad s)_ij = s_j - s_i
    s = np.asarray(scores, dtype=float)
    return s[None, :] - s[:, None]


def hodge_rank(
    R: np.ndarray,
    pairwise_stat: str = "binary",
    weight_method: str = "total",
    epsilon: float = 0.5,
    method: str = "competition",
    return_scores: bool = False,
    return_diagnostics: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, dict[str, float]]
):
    """
    Rank models with l2 HodgeRank on pairwise-comparison graphs.

    Method context:
        HodgeRank treats aggregated pairwise outcomes as an edge flow and finds
        global potentials whose gradient best matches that flow in weighted least
        squares. The minimum-norm solution is obtained from a graph Laplacian
        pseudoinverse.

    References:
        Jiang, X., Lim, L.-H., Yao, Y., & Ye, Y. (2009).
        Statistical Ranking and Combinatorial Hodge Theory.
        https://arxiv.org/abs/0811.1067

    Args:
        R: Binary tensor of shape (L, M, N) (or (L, M) treated as N=1).
        pairwise_stat:
            - "binary": Ȳ_ij = P(j>i) - P(i>j)
            - "log_odds": Ȳ_ij = log(P(j>=i)/P(j<=i)) with additive smoothing
        weight_method:
            - "total": w_ij = #comparisons (including ties)
            - "decisive": w_ij = #non-tie comparisons (wins+losses)
            - "uniform": w_ij = 1 if comparable else 0
        epsilon: Additive smoothing (counts) used only for pairwise_stat="log_odds".
        method: Ranking method passed to `scorio.utils.rank_scores`.
        return_scores: If True, return (ranking, scores) where scores are the
            HodgeRank potentials s (higher ⇒ better).
        return_diagnostics: If True, also returns a small diagnostics dict with
            least-squares residual norms.

    Returns:
        Ranking array of shape ``(L,)`` by default.
        If ``return_scores=True``, returns ``(ranking, scores)``.
        If ``return_diagnostics=True``, returns
        ``(ranking, scores, diagnostics)``.

    Notation:
        ``Y``: skew-symmetric observed edge flow from pairwise outcomes.
        ``w_ij``: symmetric nonnegative edge weights.
        ``(grad s)_ij = s_j - s_i``.
        ``(div Y)_i = \\sum_j w_{ij} Y_{ij}``.
        ``\\Delta_0``: weighted graph Laplacian.

    Formula:
        .. math::
            s^{\\star} \\in
            \\arg\\min_{s\\in\\mathbb{R}^{L}}
            \\sum_{i<j} w_{ij}\\left((s_j-s_i)-Y_{ij}\\right)^2

        .. math::
            \\Delta_0 s^{\\star} = -\\operatorname{div}(Y),
            \\qquad
            s^{\\star} = -\\Delta_0^{\\dagger}\\operatorname{div}(Y)

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
        >>> rank.hodge_rank(R).tolist()
        [1, 2]

    Notes:
        Scores are identifiable only up to an additive constant; rankings depend
        on score differences. Diagnostics report weighted residual norms
        ``||Y - grad(s)||_{2,w}``.
    """
    R = validate_input(R)
    L = R.shape[0]

    wins, ties = build_pairwise_counts(R)

    pairwise_stat = str(pairwise_stat)
    if pairwise_stat == "binary":
        Y = _pairwise_flow_binary(wins, ties)
    elif pairwise_stat == "log_odds":
        Y = _pairwise_flow_log_odds(wins, ties, epsilon=epsilon)
    else:
        raise ValueError('pairwise_stat must be one of: "binary", "log_odds"')

    w = _weights_from_counts(wins, ties, weight_method=weight_method)
    if not np.any(w > 0):
        scores = np.ones(L, dtype=float) / L
        ranking = rank_scores(scores)[method]
        if not return_scores:
            return ranking
        if not return_diagnostics:
            return ranking, scores
        return ranking, scores, {"residual_l2": 0.0, "relative_residual_l2": 0.0}

    Lap = _laplacian_from_weights(w)
    div = _divergence(w, Y)

    # Minimum-norm solution to Lap s = -div via Moore–Penrose inverse.
    scores = -np.linalg.pinv(Lap) @ div

    ranking = rank_scores(scores)[method]

    if not return_diagnostics and not return_scores:
        return ranking
    if not return_diagnostics and return_scores:
        return ranking, scores

    # Diagnostics: weighted residual norms for grad(s) vs Y.
    grad_s = _grad(scores)
    resid = Y - grad_s
    mask = np.triu(np.ones((L, L), dtype=bool), 1) & (w > 0)
    w_half = w[mask]
    r_half = resid[mask]
    y_half = Y[mask]

    residual_l2 = float(np.sqrt(np.sum(w_half * (r_half**2))))
    denom = float(np.sqrt(np.sum(w_half * (y_half**2))))
    rel = residual_l2 / denom if denom > 0 else 0.0

    return ranking, scores, {"residual_l2": residual_l2, "relative_residual_l2": rel}


__all__ = ["hodge_rank"]
