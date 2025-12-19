import numpy as np
from scipy.stats import rankdata

def rank_scores(scores_in_id_order, tol=1e-12):
    """
    Rank scores with confidence tolerance

    Args:
        scores_in_id_order (list or np.ndarray): Scores aligned by ID order.
        tol (float): Tolerance threshold for treating scores as equal.

    Returns:
        dict: {
            "competition": np.ndarray of ranks (min-rank competition),
            "competition_max": np.ndarray of ranks (max-rank competition),
            "dense": np.ndarray of ranks (dense ranking),
            "avg": np.ndarray of ranks (average/fractional ranking)
        }
    """
    scores = np.asarray(scores_in_id_order, dtype=float)
    order = np.argsort(-scores)  # descending order
    sorted_scores = scores[order]

    # Group near-equal scores (within tolerance)
    grouped_scores = sorted_scores.copy()
    for i in range(1, len(grouped_scores)):
        if abs(grouped_scores[i] - grouped_scores[i - 1]) <= tol:
            grouped_scores[i] = grouped_scores[i - 1]

    def ranker(method):
        ranks_sorted = rankdata(-grouped_scores, method=method)
        ranks = np.empty_like(ranks_sorted)
        ranks[order] = ranks_sorted
        return ranks

    return {
        "competition": ranker("min"),        # 1,2,2,4,5
        "competition_max": ranker("max"),    # 1,3,3,4,5
        "dense": ranker("dense"),            # 1,2,2,3,4
        "avg": ranker("average"),            # 1.0,2.5,2.5,4.0,5.0
    }
