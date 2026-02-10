r"""
SerialRank: spectral ranking using seriation.

Based on:
    Fogel, d'Aspremont, Vojnovic (2014/2016). "SerialRank: Spectral Ranking using Seriation".

Given a pairwise comparison matrix :math:`C` (skew-symmetric, with entries in
:math:`\{-1, 0, 1\}` or more generally in :math:`[-1, 1]`), SerialRank builds a
similarity matrix

.. math::
    S = \frac{1}{2}\left(n \mathbf{1}\mathbf{1}^{\top} + C C^{\top}\right)

and ranks items by sorting the Fiedler vector (second-smallest eigenvector) of
the graph Laplacian

.. math::
    L_S = \operatorname{diag}(S\mathbf{1}) - S.

In scorio's setting, the input is a binary tensor :math:`R` of shape
:math:`(L, M, N)` where :math:`R_{lmn} = 1` means model :math:`l` solved
question :math:`m` on trial :math:`n`. We derive pairwise comparisons from the
number of decisive wins/losses between models.
"""

import numpy as np

from scorio.utils import rank_scores

from ._base import build_pairwise_counts, validate_input
from ._types import RankMethod


def _comparison_matrix_from_counts(
    wins: np.ndarray, ties: np.ndarray, comparison: str
) -> np.ndarray:
    """
    Build a skew-symmetric comparison matrix C from pairwise win/tie counts.

    C[i,j] > 0  => i is preferred to j.
    """
    comparison = str(comparison)

    if comparison in {"prob_diff", "fractional"}:
        total = wins + wins.T + ties
        C = np.zeros_like(wins, dtype=float)
        mask = total > 0
        C[mask] = (wins[mask] - wins.T[mask]) / total[mask]
        np.fill_diagonal(C, 0.0)
        return C

    if comparison in {"sign", "majority"}:
        diff = wins - wins.T
        C = np.sign(diff).astype(float)
        np.fill_diagonal(C, 0.0)
        return C

    raise ValueError('comparison must be "prob_diff" or "sign"')


def _serialrank_similarity(C: np.ndarray) -> np.ndarray:
    """
    Similarity matrix S_match from the SerialRank paper:

        S = Σ_k (1 + C_{ik} C_{jk})/2  = 1/2 (n 11^T + C C^T)
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    ones = np.ones((n, n), dtype=float)
    S = 0.5 * (n * ones + (C @ C.T))
    return S


def _laplacian(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    d = S.sum(axis=1)
    return np.diag(d) - S


def _fiedler_vector(L: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Return a Fiedler vector and whether the Fiedler eigenspace is one-dimensional.

    SerialRank requires sorting the eigenvector associated with the second-smallest
    Laplacian eigenvalue. If that eigenvalue is repeated, the Fiedler vector is not
    unique and any basis vector is arbitrary for ranking purposes.
    """
    w, V = np.linalg.eigh(L)
    if V.shape[1] < 2:
        return np.ones(L.shape[0], dtype=float), False

    v = V[:, 1]
    if V.shape[1] == 2:
        return v, True

    scale = max(1.0, float(np.max(np.abs(w))))
    eigengap = float(w[2] - w[1])
    unique = bool(np.isfinite(eigengap) and eigengap > 1e-10 * scale)
    return v, unique


def _orientation_key(scores: np.ndarray, C: np.ndarray) -> tuple[int, float, float]:
    """
    Decide between the two possible orientations (scores vs -scores).

    Returns a key to be minimized: (unweighted_upsets, weighted_upsets, -corr).
    """
    scores = np.asarray(scores, dtype=float)
    C = np.asarray(C, dtype=float)
    n = scores.size

    diff = scores[:, None] - scores[None, :]

    mask = np.triu(np.ones((n, n), dtype=bool), 1)
    c = C[mask]
    pred = np.sign(diff[mask])

    nz = c != 0
    if not np.any(nz):
        return (0, 0.0, 0.0)

    c = c[nz]
    pred = pred[nz]

    disagree = (pred == 0) | ((pred * c) < 0)
    upsets = int(np.sum(disagree))
    w_upsets = float(np.sum(np.abs(c[disagree])))
    corr = float(np.sum(pred * c))

    return (upsets, w_upsets, -corr)


def serial_rank(
    R: np.ndarray,
    comparison: str = "prob_diff",
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Rank models with SerialRank spectral seriation.

    Method context:
        SerialRank builds a similarity matrix from pairwise comparisons, computes
        its graph Laplacian, and orders models by a Fiedler vector (second-smallest
        Laplacian eigenvector). This is a seriation-based ranking method designed to
        be robust to noisy pairwise outcomes.

    References:
        Fogel, F., d'Aspremont, A., & Vojnovic, M. (2016).
        Spectral Ranking Using Seriation. Journal of Machine Learning Research.
        https://jmlr.org/papers/v17/16-035.html

    Args:
        R: Binary tensor of shape (L, M, N) (or (L, M) treated as N=1).
        comparison: How to aggregate multiple comparisons between a pair:
            - "prob_diff": C_ij = (wins_ij - wins_ji) / total_ij in [-1, 1]
            - "sign":      C_ij = sign(wins_ij - wins_ji) in {-1, 0, 1}
        method: Ranking method passed to `scorio.utils.rank_scores`.
        return_scores: If True, return (ranking, scores) where scores are the
            oriented Fiedler vector (higher ⇒ better).

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, returns ``(ranking, scores)`` with scores of
        shape ``(L,)``.

    Notation:
        ``L``: number of models.
        ``W_ij``: number of decisive outcomes where model ``i`` beats ``j``.
        ``T_ij``: number of ties between models ``i`` and ``j``.
        ``C``: skew-symmetric comparison matrix.
        ``S``: SerialRank similarity matrix.
        ``L_S``: graph Laplacian of ``S``.

    Formula:
        .. math::
            C_{ij} = \\frac{W_{ij} - W_{ji}}{W_{ij} + W_{ji} + T_{ij}},
            \\quad C_{ii} = 0

        .. math::
            S = \\frac{1}{2}\\left(L\\,\\mathbf{1}\\mathbf{1}^{\\top} + C C^{\\top}\\right),
            \\quad
            L_S = \\operatorname{diag}(S\\mathbf{1}) - S

        Rank by sorting a Fiedler vector of ``L_S``. The sign is chosen to best
        align with observed pairwise comparisons.

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([[[1, 1], [1, 0]], [[0, 0], [0, 1]]])
        >>> rank.serial_rank(R).tolist()
        [1, 2]

    Notes:
        The original SerialRank paper sets ``C_ii = 1`` for binary tournaments.
        In this implementation we keep ``C`` skew-symmetric with ``C_ii = 0``;
        this changes ``S`` only by a constant diagonal shift and leaves the
        Laplacian (hence the ranking) unchanged.

        If the Fiedler eigenvalue is not simple, the ordering is not identifiable.
        In that degenerate case, the method falls back to mean accuracy scores.
    """
    R = validate_input(R)

    wins, ties = build_pairwise_counts(R)
    C = _comparison_matrix_from_counts(wins, ties, comparison=comparison)

    S = _serialrank_similarity(C)
    Ls = _laplacian(S)

    v, is_unique = _fiedler_vector(Ls)
    if (not is_unique) or (not np.all(np.isfinite(v))) or np.allclose(v, v[0]):
        scores = R.mean(axis=(1, 2))
        ranking = rank_scores(scores)[method]
        return (ranking, scores) if return_scores else ranking

    # Choose the orientation that best agrees with observed comparisons.
    key_pos = _orientation_key(v, C)
    key_neg = _orientation_key(-v, C)
    scores = v if key_pos <= key_neg else -v

    # Stabilize for degenerate (nearly tied) cases.
    if np.std(scores) < 1e-12:
        scores = R.mean(axis=(1, 2))

    ranking = rank_scores(scores)[method]
    return (ranking, scores) if return_scores else ranking


__all__ = ["serial_rank"]
