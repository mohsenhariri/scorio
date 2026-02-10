"""
Sequential pairwise rating methods.

These methods adapt classical rating systems (Elo, TrueSkill, and Glicko)
to binary model-outcome tensors by inducing head-to-head results from each
question-trial slice.
"""

import numpy as np
from scipy.stats import norm

from scorio.utils import rank_scores

from ._base import validate_input
from ._types import PairwiseTieHandling, RankMethod


def elo(
    R: np.ndarray,
    K: float = 32.0,
    initial_rating: float = 1500.0,
    tie_handling: PairwiseTieHandling = "correct_draw_only",
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Rank models with sequential Elo updates on induced pairwise matches.

    Method context:
        Each question-trial pair ``(m, n)`` induces head-to-head outcomes for
        every model pair. Ratings are updated online, so final ratings depend
        on stream order.

    References:
        Elo, A. E. (1978). The Rating of Chessplayers, Past and Present.
        Arco Publishing.
        https://archive.org/details/ratingofchesspla0000eloa

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        K: Positive Elo step size.
        initial_rating: Initial rating assigned to every model.
        tie_handling: Policy for ``(1,1)`` and ``(0,0)`` outcomes:
            - ``"skip"``: ignore ties
            - ``"draw"``: treat both tie types as draws
            - ``"correct_draw_only"``: draw only for ``(1,1)``, skip ``(0,0)``
        method: Tie-handling rule passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, ratings)``.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns final Elo ratings of shape
        ``(L,)``.

    Notation:
        ``r_i`` is model ``i`` rating, and ``S_{ij}`` is the observed score for
        model ``i`` against ``j`` in one induced match.

    Formula:
        .. math::
            E_{ij} = \\frac{1}{1 + 10^{(r_j-r_i)/400}}

        .. math::
            r_i \\leftarrow r_i + K(S_{ij} - E_{ij}), \\quad
            r_j \\leftarrow r_j + K((1-S_{ij}) - (1-E_{ij}))

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks, ratings = rank.elo(R, return_scores=True)
        >>> ranks.tolist()
        [1, 2]
        >>> float(ratings[0] > ratings[1])
        1.0

        >>> # tie handling can change updates
        >>> rank.elo(np.array([[[1]], [[1]]]), tie_handling="skip").tolist()
        [1, 1]

    Notes:
        The implementation processes pairs in index order for each
        ``(trial, question)`` event.
    """
    R = validate_input(R)
    L, M, N = R.shape
    K = float(K)
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError(f"K must be a positive finite scalar; got {K}")

    initial_rating = float(initial_rating)
    if not np.isfinite(initial_rating):
        raise ValueError("initial_rating must be finite.")

    tie_handling = str(tie_handling)
    if tie_handling not in {"skip", "draw", "correct_draw_only"}:
        raise ValueError(
            'tie_handling must be one of: "skip", "draw", "correct_draw_only"'
        )

    ratings = np.full(L, initial_rating, dtype=float)

    for t in range(N):
        for q in range(M):
            outcomes = R[:, q, t]

            for i in range(L):
                for j in range(i + 1, L):
                    r_i, r_j = int(outcomes[i]), int(outcomes[j])

                    # Determine actual scores S_i, S_j (or skip)
                    if r_i == r_j:
                        if tie_handling == "skip":
                            continue
                        if tie_handling == "draw":
                            S_i, S_j = 0.5, 0.5
                        elif tie_handling == "correct_draw_only":
                            if r_i == 1:  # (1,1)
                                S_i, S_j = 0.5, 0.5
                            else:  # (0,0)
                                continue
                    else:
                        S_i, S_j = (1.0, 0.0) if r_i > r_j else (0.0, 1.0)

                    # Expected scores
                    R_i, R_j = ratings[i], ratings[j]
                    E_i = 1.0 / (1.0 + 10.0 ** ((R_j - R_i) / 400.0))
                    E_j = 1.0 - E_i

                    # Update
                    ratings[i] = R_i + K * (S_i - E_i)
                    ratings[j] = R_j + K * (S_j - E_j)

    ranking = rank_scores(ratings)[method]

    return (ranking, ratings) if return_scores else ranking


def trueskill(
    R: np.ndarray,
    mu_initial: float = 25.0,
    sigma_initial: float = 25.0 / 3,
    beta: float = 25.0 / 6,
    tau: float = 25.0 / 300,
    method: RankMethod = "competition",
    return_scores: bool = False,
    tie_handling: PairwiseTieHandling = "skip",
    draw_margin: float = 0.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Rank models with a two-player TrueSkill update stream.

    Method context:
        Each model has latent skill belief
        :math:`\\mathcal{N}(\\mu_i, \\sigma_i^2)`. Every decisive pairwise
        result from ``R`` applies a two-player TrueSkill update. Ties can be
        skipped or treated as draw events with an optional draw margin.
        Ranking uses final ``mu``.

    References:
        Herbrich, R., Minka, T., & Graepel, T. (2006).
        TrueSkill(TM): A Bayesian Skill Rating System.
        NeurIPS 19.
        https://proceedings.neurips.cc/paper_files/paper/2006/file/f44ee263952e65b3610b8ba51229d1f9-Paper.pdf

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        mu_initial: Initial mean skill.
        sigma_initial: Initial skill standard deviation. Must be positive.
        beta: Performance-noise scale. Must be positive.
        tau: Per-round dynamics scale for sigma inflation. Must be nonnegative.
        method: Ranking variant passed to :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ``(ranking, mu)``.
        tie_handling: Policy for tied outcomes ``(1,1)`` and ``(0,0)``:
            - ``"skip"``: ignore ties
            - ``"draw"``: treat both tie types as draws
            - ``"correct_draw_only"``: draw only for ``(1,1)``, skip ``(0,0)``
        draw_margin: Nonnegative draw margin :math:`\\epsilon` in the
            performance-difference space. ``0`` recovers no-margin updates.

    Returns:
        Ranking array of shape ``(L,)``.
        If ``return_scores=True``, also returns final ``mu`` values
        (shape ``(L,)``).

    Notation:
        For one match between models ``i`` and ``j``:
        ``c = sqrt(2 beta^2 + sigma_i^2 + sigma_j^2)``,
        ``t = (mu_i - mu_j) / c``, ``epsilon = draw_margin / c``.

    Formula:
        .. math:: v_{win}(t,\\epsilon) = \\frac{\\phi(t-\\epsilon)}{\\Phi(t-\\epsilon)},\\quad w_{win}(t,\\epsilon)=v_{win}(t,\\epsilon)\\left(v_{win}(t,\\epsilon)+t-\\epsilon\\right)

        .. math:: \\mu_i' = \\mu_i + \\frac{\\sigma_i^2}{c} v_{win}(t,\\epsilon),\\quad \\mu_j' = \\mu_j - \\frac{\\sigma_j^2}{c} v_{win}(t,\\epsilon)

        .. math:: \\sigma_i'^2 = \\sigma_i^2\\left(1 - \\frac{\\sigma_i^2}{c^2} w_{win}(t,\\epsilon)\\right),\\quad \\sigma_j'^2 = \\sigma_j^2\\left(1 - \\frac{\\sigma_j^2}{c^2} w_{win}(t,\\epsilon)\\right)

        .. math:: v_{draw}(t,\\epsilon) = \\frac{\\phi(-\\epsilon-t)-\\phi(\\epsilon-t)}{\\Phi(\\epsilon-t)-\\Phi(-\\epsilon-t)}

        .. math:: w_{draw}(t,\\epsilon) = v_{draw}(t,\\epsilon)^2 + \\frac{(\\epsilon-t)\\phi(\\epsilon-t)-(-\\epsilon-t)\\phi(-\\epsilon-t)}{\\Phi(\\epsilon-t)-\\Phi(-\\epsilon-t)}

        .. math:: \\mu_i' = \\mu_i + \\frac{\\sigma_i^2}{c} v_{draw}(t,\\epsilon),\\quad \\mu_j' = \\mu_j - \\frac{\\sigma_j^2}{c} v_{draw}(t,\\epsilon)

        .. math:: \\sigma_i'^2 = \\sigma_i^2\\left(1 - \\frac{\\sigma_i^2}{c^2} w_{draw}(t,\\epsilon)\\right),\\quad \\sigma_j'^2 = \\sigma_j^2\\left(1 - \\frac{\\sigma_j^2}{c^2} w_{draw}(t,\\epsilon)\\right)

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks, mu = rank.trueskill(R, return_scores=True)
        >>> ranks.tolist()
        [1, 2]
        >>> float(mu[0] > mu[1])
        1.0

    Notes:
        This is a simplified two-player update stream, not full multiplayer
        factor-graph inference from the original TrueSkill paper.
    """
    R = validate_input(R)
    L, M, N = R.shape

    mu_initial = float(mu_initial)
    sigma_initial = float(sigma_initial)
    beta = float(beta)
    tau = float(tau)
    draw_margin = float(draw_margin)
    if not np.isfinite(mu_initial):
        raise ValueError("mu_initial must be finite.")
    if not np.isfinite(sigma_initial) or sigma_initial <= 0.0:
        raise ValueError("sigma_initial must be a positive finite scalar.")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be a positive finite scalar.")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be a nonnegative finite scalar.")
    if not np.isfinite(draw_margin) or draw_margin < 0.0:
        raise ValueError("draw_margin must be a nonnegative finite scalar.")

    tie_handling = str(tie_handling)
    if tie_handling not in {"skip", "draw", "correct_draw_only"}:
        raise ValueError(
            'tie_handling must be one of: "skip", "draw", "correct_draw_only"'
        )

    mu = np.full(L, mu_initial, dtype=float)
    sigma = np.full(L, sigma_initial, dtype=float)

    def v_win(t: float, epsilon: float) -> float:
        """Correction term for decisive outcomes."""
        x = t - epsilon
        denom = float(norm.cdf(x))
        if denom < 1e-12:
            # Asymptotic approximation for extreme tails.
            return float(-x)
        return float(norm.pdf(x) / denom)

    def w_win(t: float, epsilon: float) -> float:
        v = v_win(t, epsilon)
        return float(v * (v + t - epsilon))

    def v_draw(t: float, epsilon: float) -> float:
        """Correction term for draw outcomes."""
        a = -epsilon - t
        b = epsilon - t
        cdf_a = float(norm.cdf(a))
        cdf_b = float(norm.cdf(b))
        denom = cdf_b - cdf_a
        if denom < 1e-12:
            return 0.0
        return float((norm.pdf(a) - norm.pdf(b)) / denom)

    def w_draw(t: float, epsilon: float) -> float:
        a = -epsilon - t
        b = epsilon - t
        cdf_a = float(norm.cdf(a))
        cdf_b = float(norm.cdf(b))
        denom = cdf_b - cdf_a
        if denom < 1e-12:
            return 1.0
        v = v_draw(t, epsilon)
        term = ((b * norm.pdf(b)) - (a * norm.pdf(a))) / denom
        return float(v * v + term)

    def update_decisive(mu1, sigma1, mu2, sigma2, player1_wins):
        """Update ratings for a decisive 2-player match."""
        c = np.sqrt(2 * beta**2 + sigma1**2 + sigma2**2)
        epsilon = draw_margin / c

        if player1_wins:
            t = (mu1 - mu2) / c
            v = v_win(t, epsilon)
            w = w_win(t, epsilon)

            mu1_new = mu1 + (sigma1**2 / c) * v
            mu2_new = mu2 - (sigma2**2 / c) * v
        else:
            t = (mu2 - mu1) / c
            v = v_win(t, epsilon)
            w = w_win(t, epsilon)

            mu2_new = mu2 + (sigma2**2 / c) * v
            mu1_new = mu1 - (sigma1**2 / c) * v

        sigma1_new = sigma1 * np.sqrt(max(1 - (sigma1**2 / c**2) * w, 1e-12))
        sigma2_new = sigma2 * np.sqrt(max(1 - (sigma2**2 / c**2) * w, 1e-12))

        return mu1_new, sigma1_new, mu2_new, sigma2_new

    def update_draw(mu1, sigma1, mu2, sigma2):
        """Update ratings for a 2-player draw."""
        c = np.sqrt(2 * beta**2 + sigma1**2 + sigma2**2)
        epsilon = draw_margin / c
        t = (mu1 - mu2) / c
        v = v_draw(t, epsilon)
        w = w_draw(t, epsilon)

        mu1_new = mu1 + (sigma1**2 / c) * v
        mu2_new = mu2 - (sigma2**2 / c) * v
        sigma1_new = sigma1 * np.sqrt(max(1 - (sigma1**2 / c**2) * w, 1e-12))
        sigma2_new = sigma2 * np.sqrt(max(1 - (sigma2**2 / c**2) * w, 1e-12))

        return mu1_new, sigma1_new, mu2_new, sigma2_new

    # Process trials sequentially
    for t in range(N):
        for q in range(M):
            outcomes = R[:, q, t]

            # Process all pairs
            for i in range(L):
                for j in range(i + 1, L):
                    r_i, r_j = outcomes[i], outcomes[j]

                    if r_i == r_j:
                        if tie_handling == "skip":
                            continue
                        if tie_handling == "correct_draw_only" and int(r_i) == 0:
                            continue
                        mu[i], sigma[i], mu[j], sigma[j] = update_draw(
                            mu[i], sigma[i], mu[j], sigma[j]
                        )
                        continue

                    mu[i], sigma[i], mu[j], sigma[j] = update_decisive(
                        mu[i], sigma[i], mu[j], sigma[j], player1_wins=(r_i > r_j)
                    )

        # Apply dynamics (skill drift)
        sigma = np.sqrt(sigma**2 + tau**2)

    ranking = rank_scores(mu)[method]

    return (ranking, mu) if return_scores else ranking


def glicko(
    R: np.ndarray,
    initial_rating: float = 1500.0,
    initial_rd: float = 350.0,
    c: float = 0.0,
    rd_max: float = 350.0,
    tie_handling: PairwiseTieHandling = "correct_draw_only",
    return_deviation: bool = False,
    method: RankMethod = "competition",
    return_scores: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """
    Rank models with Glicko rating and rating-deviation updates.

    Method context:
        Glicko extends Elo by tracking both rating ``r`` and uncertainty
        ``RD``. Each ``(question, trial)`` is treated as one rating period
        containing all pairwise induced matches for that slice.

    References:
        Glickman, M. E. (1999).
        Parameter Estimation in Large Dynamic Paired Comparison Experiments.
        Journal of the Royal Statistical Society: Series C, 48(3), 377-394.
        https://doi.org/10.1111/1467-9876.00159

        Glickman, M. E. (n.d.). The Glicko System.
        https://www.glicko.net/glicko/glicko.pdf

    Args:
        R: Binary outcome tensor with shape ``(L, M, N)`` or matrix
            ``(L, M)`` (treated as ``N=1``).
        initial_rating: Initial rating ``r`` for every model.
        initial_rd: Initial rating deviation ``RD`` for every model.
        c: Nonnegative RD inflation parameter between rating periods.
        rd_max: Positive hard cap for ``RD``.
        tie_handling: Policy for tied outcomes:
            - ``"skip"``: ignore ``(1,1)`` and ``(0,0)``
            - ``"draw"``: treat both as draws
            - ``"correct_draw_only"``: draw for ``(1,1)``, skip ``(0,0)``
        return_deviation: If ``True``, return ``RD`` in addition to ratings.
        method: Tie-handling rule for :func:`scorio.utils.rank_scores`.
        return_scores: If ``True``, return ratings with ranking.

    Returns:
        If ``return_deviation=False`` and ``return_scores=False``:
        ranking array of shape ``(L,)``.

        If ``return_deviation=False`` and ``return_scores=True``:
        ``(ranking, rating)`` where ``rating`` has shape ``(L,)``.

        If ``return_deviation=True``:
        ``(ranking, rating, rd)`` with both vectors shape ``(L,)``.

    Notation:
        ``q = ln(10)/400``, ``g(RD) = 1 / sqrt(1 + 3 q^2 RD^2 / pi^2)``.
        For model ``i`` in one period with opponents ``j``:
        ``S_ij`` is observed score and
        ``E_ij = 1 / (1 + 10^{-g(RD_j)(r_i-r_j)/400})``.

    Formula:
        .. math::
            d_i^2 = \\left(q^2 \\sum_j g(RD_j)^2 E_{ij}(1-E_{ij})\\right)^{-1}

        .. math::
            RD_i' = \\left(\\frac{1}{RD_i^2} + \\frac{1}{d_i^2}\\right)^{-1/2},
            \\quad
            r_i' = r_i + \\frac{q}{\\frac{1}{RD_i^2}+\\frac{1}{d_i^2}}
            \\sum_j g(RD_j)(S_{ij}-E_{ij})

    Examples:
        >>> import numpy as np
        >>> from scorio import rank
        >>> R = np.array([
        ...     [[1, 1], [1, 1]],
        ...     [[0, 0], [0, 0]],
        ... ])
        >>> ranks, ratings, rds = rank.glicko(R, return_deviation=True)
        >>> ranks.tolist()
        [1, 2]
        >>> float(ratings[0] > ratings[1])
        1.0

    Notes:
        Models with no games in a rating period keep rating unchanged but can
        still receive RD inflation when ``c > 0``.
    """
    R = validate_input(R)
    L, M, N = R.shape

    initial_rating = float(initial_rating)
    initial_rd = float(initial_rd)
    if not np.isfinite(initial_rating):
        raise ValueError("initial_rating must be finite.")
    if not np.isfinite(initial_rd) or initial_rd <= 0:
        raise ValueError("initial_rd must be > 0 and finite.")

    rd_max = float(rd_max)
    if rd_max <= 0:
        raise ValueError("rd_max must be > 0")

    c = float(c)
    if c < 0:
        raise ValueError("c must be >= 0")

    tie_handling = str(tie_handling)
    if tie_handling not in {"skip", "draw", "correct_draw_only"}:
        raise ValueError(
            'tie_handling must be one of: "skip", "draw", "correct_draw_only"'
        )

    rating = np.full(L, initial_rating, dtype=float)
    rd = np.full(L, min(initial_rd, rd_max), dtype=float)

    # Glicko constants.
    q = float(np.log(10.0) / 400.0)

    def g(rd_opponent: np.ndarray) -> np.ndarray:
        rd_opponent = np.asarray(rd_opponent, dtype=float)
        return 1.0 / np.sqrt(1.0 + (3.0 * (q**2) * (rd_opponent**2)) / (np.pi**2))

    def expected_score(
        rating_i: float, rating_j: np.ndarray, g_j: np.ndarray
    ) -> np.ndarray:
        # E = 1 / (1 + 10^(-g(RD_j) * (r_i - r_j)/400))
        return 1.0 / (1.0 + 10.0 ** (-(g_j * (rating_i - rating_j) / 400.0)))

    for t in range(N):
        for m in range(M):
            # RD inflation at the start of each rating period.
            if c > 0:
                rd = np.minimum(np.sqrt(rd**2 + c**2), rd_max)

            outcomes = R[:, m, t]

            opponents: list[list[int]] = [[] for _ in range(L)]
            results: list[list[float]] = [[] for _ in range(L)]

            for i in range(L):
                for j in range(i + 1, L):
                    r_i = int(outcomes[i])
                    r_j = int(outcomes[j])

                    if r_i == r_j:
                        if tie_handling == "skip":
                            continue
                        if tie_handling == "draw":
                            s_i = s_j = 0.5
                        else:  # correct_draw_only
                            if r_i == 1:
                                s_i = s_j = 0.5
                            else:
                                continue
                    else:
                        if r_i > r_j:
                            s_i, s_j = 1.0, 0.0
                        else:
                            s_i, s_j = 0.0, 1.0

                    opponents[i].append(j)
                    results[i].append(s_i)
                    opponents[j].append(i)
                    results[j].append(s_j)

            new_rating = rating.copy()
            new_rd = rd.copy()

            for i in range(L):
                if not opponents[i]:
                    continue

                opp = np.asarray(opponents[i], dtype=int)
                s = np.asarray(results[i], dtype=float)

                rd_opp = rd[opp]
                rating_opp = rating[opp]

                g_opp = g(rd_opp)
                E = expected_score(rating[i], rating_opp, g_opp)

                # d^2 = 1 / (q^2 Σ g^2 E(1-E))
                denom = float(np.sum((g_opp**2) * E * (1.0 - E)))
                if denom <= 0 or not np.isfinite(denom):
                    continue

                d2 = 1.0 / ((q**2) * denom)

                inv_var = (1.0 / (rd[i] ** 2)) + (1.0 / d2)
                if inv_var <= 0 or not np.isfinite(inv_var):
                    continue

                rd_new = float(np.sqrt(1.0 / inv_var))
                rd_new = float(np.clip(rd_new, 1e-12, rd_max))

                delta = float(np.sum(g_opp * (s - E)))
                rating_new = float(rating[i] + (q / inv_var) * delta)

                new_rating[i] = rating_new
                new_rd[i] = rd_new

            rating = new_rating
            rd = new_rd

    ranking = rank_scores(rating)[method]

    if return_deviation:
        return ranking, rating, rd

    return (ranking, rating) if return_scores else ranking


__all__ = ["elo", "trueskill", "glicko"]
