r"""
Pointwise ranking methods.

Pointwise (item-wise) ranking methods assign each model a **single scalar score** by aggregating its per-item performance over the evaluation set, **without directly comparing models to each other** on any item.

Concretely, given per-question accuracies \( a_{lm} \) (e.g., the mean of \( R_{lmn} \) over trials), a pointwise method computes a score of the form:

\[
S_l = \sum_{m=1}^M w_m \, g(a_{lm}, \phi_m)
\]

where:

- \( \phi_m \) is an item-level statistic estimated from the dataset
  (e.g., solve rate \( p_m \), a discrimination index, or an uncertainty measure),
- \( g(\cdot) \) is a fixed transformation (often the identity),
- \( w_m \) are nonnegative weights (often normalized to sum to 1).

The defining characteristic is that ranking is induced solely by these **per-model aggregate scores**, rather than by explicit pairwise win/loss outcomes between models.

"""

import numpy as np

from scorio.utils import rank_scores

from ._base import validate_input
from ._types import RankMethod


def inverse_difficulty(
    R: np.ndarray,
    method: RankMethod = "competition",
    return_scores: bool = False,
    clip_range: tuple = (0.01, 0.99),
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Rank models by inverse-difficulty-weighted per-question accuracy.

    Method context:
        This pointwise method upweights hard questions by assigning each
        question an inverse weight proportional to the reciprocal global solve
        rate. It emphasizes rare successes while still aggregating all
        questions into one scalar score per model.

    References:
        Inverse probability weighting (Wikipedia):
        https://en.wikipedia.org/wiki/Inverse_probability_weighting

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
            One of ``"competition"``, ``"competition_max"``, ``"dense"``,
            or ``"avg"``.
        return_scores: If ``True``, return ``(ranking, scores)``.
        clip_range: Two-sided clipping interval ``(a, b)`` applied to global
            solve rates before inversion. Must satisfy
            ``0 < a < b <= 1``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns inverse-difficulty scores of
        shape ``(L,)``.

    Notation:
        ``R[l, m, n]`` is the binary outcome for model ``l``, question ``m``,
        and trial ``n``. Let
        ``p_hat[l, m] = (1/N) * sum_n R[l, m, n]`` and
        ``p[m] = (1/(L N)) * sum_{l,n} R[l, m, n]``.

    Formula:
        .. math::
            w_m \\propto \\frac{1}{\\operatorname{clip}(p_m, a, b)},
            \\quad \\sum_{m=1}^{M} w_m = 1

        .. math::
            s_l^{\\mathrm{inv\\text{-}diff}} =
            \\sum_{m=1}^{M} w_m \\, \\widehat{p}_{lm}

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [0, 0], [0, 0]],
        ...     [[0, 0], [1, 1], [0, 0]],
        ... ])
        >>> ranks, scores = rank.inverse_difficulty(R, return_scores=True)
        >>> ranks.tolist()
        [1, 1]
        >>> scores.shape
        (2,)

        >>> # tighter clipping is allowed
        >>> rank.inverse_difficulty(R, clip_range=(0.05, 0.95)).shape
        (2,)

    Notes:
        Very small clip lower bounds can make the weighting highly sensitive
        to a few rare solves. This implementation is a simple inverse-weighted
        pointwise scorer.
    """
    R = validate_input(R)

    if len(clip_range) != 2:
        raise ValueError("clip_range must be a length-2 tuple (low, high).")
    low = float(clip_range[0])
    high = float(clip_range[1])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("clip_range values must be finite.")
    if not (0.0 < low < high <= 1.0):
        raise ValueError("clip_range must satisfy 0 < low < high <= 1.")

    # Global difficulty: average across all models and trials for each question
    question_difficulty = R.mean(axis=(0, 2))  # Shape: (M,)

    # Clip to avoid extreme weights
    question_difficulty = np.clip(question_difficulty, low, high)

    # Inverse difficulty weights (normalized)
    weights = 1.0 / question_difficulty
    total_weight = float(np.sum(weights))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError(
            "inverse-difficulty weights are not finite; choose a stricter clip_range."
        )
    weights = weights / total_weight

    # Per-model accuracy on each question
    model_question_accuracy = R.mean(axis=2)  # Shape: (L, M)

    scores = model_question_accuracy @ weights

    ranking = rank_scores(scores)[method]

    return (ranking, scores) if return_scores else ranking


__all__ = ["inverse_difficulty"]
