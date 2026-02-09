"""
Ranking methods for comparing multiple models.

This module provides a collection of ranking methods for
evaluating and comparing LLMs used in "Ranking Reasoning LLMs under Test-Time Scaling".



Input Format
------------
All ranking methods expect a 3D numpy array R of shape (L, M, N) where:
- L = number of models to rank
- M = number of questions
- N = number of independent trials per question (e.g., using top-p sampling)

R[l, m, n] = 1 if model l answered question m correctly on trial n, else 0.

Alternatively, for single-trial scenarios (N=1), you can pass a 2D array of
shape (L, M) which will be automatically converted to (L, M, 1).

Output Format
-------------
Ranking methods compute raw model scores (e.g., estimated strengths
π_i, posterior means, or metric-based scores) and derive rankings from
those scores.

- Raw scores: `np.ndarray` of shape (L,) containing a score for each model.
- Default return: return a 1D `np.ndarray` of shape (L,) containing
  the ranking (for example the `'competition'` ranking produced by
  `scorio.utils.rank_scores(scores)`).
- Set `return_scores=True` to return a tuple `(ranking, scores)`
  where `ranking` is a 1D array of ranks and `scores` is the corresponding
  raw scores.

To convert raw scores to different ranking variants, use `scorio.utils.rank_scores()`
which returns a mapping of ranking variants (e.g., `'competition'`, `'dense'`, `'ordinal'`).

Available Methods
-----------------

**Evaluation metric-based:**
- `mean`: Rank by mean accuracy (alias for accuracy.mean)
- `bayes`: Rank by Bayes@N metric
- `pass_at_k`: Rank by Pass@k metric
- `pass_hat_k`: Rank by Pass^@k metric
- `g_pass_at_k_tau`: Rank by G-Pass@k_τ metric
- `mg_pass_at_k`: Rank by mG-Pass@k metric


**Paired-comparison probabilistic models:**
- `bradley_terry`: Bradley-Terry Maximum Likelihood (BT-ML) estimation of model strengths
- `bradley_terry_map`: Bradley-Terry Maximum A Posteriori (BT-MAP) with configurable priors on log-strengths
- `bradley_terry_davidson`: Davidson tie extension of Bradley-Terry (ML)
- `bradley_terry_davidson_map`: Davidson tie extension of Bradley-Terry (MAP)
- `rao_kupper`: Rao-Kupper tie model (ML)
- `rao_kupper_map`: Rao-Kupper tie model (MAP)

**Accuracy-based:**
- `inverse_difficulty`: Difficulty-weighted accuracy


**Pairwise rating systems:**
- `elo`: Elo rating system
- `glicko`: Glicko rating system
- `trueskill`: TrueSkill Bayesian rating


**Bayesian methods:**
- `thompson`: Thompson sampling
- `bayesian_mcmc`: Full Bayesian via MCMC

**Voting methods:**
- `borda`: Borda count
- `copeland`: Copeland score
- `win_rate`: Simple pairwise win rate
- `minimax`: Minimax (Simpson–Kramer) Condorcet method (question-level)
- `schulze`: Schulze beatpath Condorcet method (question-level)
- `ranked_pairs`: Ranked Pairs / Tideman Condorcet method (question-level)
- `kemeny_young`: Kemeny–Young rank aggregation (exact MILP)
- `nanson`: Nanson's Borda-elimination method
- `baldwin`: Baldwin's iterative Borda elimination
- `majority_judgment`: Majority Judgment (median-grade) ranking

**Item Response Theory:**
- `rasch`: 1-Parameter Logistic (Rasch) model
- `rasch_map`: Rasch model with MAP estimation
- `rasch_2pl`: 2-Parameter Logistic model
- `rasch_2pl_map`: 2PL model with MAP estimation
- `rasch_3pl`: 3-Parameter Logistic model (with guessing)
- `rasch_3pl_map`: 3PL model with MAP estimation
- `rasch_mml`: Rasch with Marginal Maximum Likelihood
- `rasch_mml_credible`: Rasch MML credible lower bound (conservative)
- `dynamic_irt`: Longitudinal IRT

**Graph-based:**
- `pagerank`: PageRank on comparison graph
- `spectral`: Spectral ranking (eigenvector)
- `rank_centrality`: Rank Centrality (Markov chain)
- `alpharank`: AlphaRank evolutionary dynamics
- `nash`: Nash equilibrium-based ranking

**Seriation-based:**
- `serial_rank`: SerialRank spectral seriation

**Hodge-theoretic:**
- `hodge_rank`: HodgeRank (least-squares on graph)

**Listwise / setwise choice models (Luce family):**
- `plackett_luce`: PL Maximum Likelihood via MM algorithm (pairwise reduction; BT-equivalent here)
- `plackett_luce_map`: PL Maximum A Posteriori with priors
- `davidson_luce`: Davidson-Luce ML with tie handling (setwise ties)
- `davidson_luce_map`: Davidson-Luce MAP with priors
- `bradley_terry_luce`: Bradley--Terry--Luce setwise-choice model (ML)
- `bradley_terry_luce_map`: Bradley--Terry--Luce setwise-choice model (MAP)

Examples
--------
>>> import numpy as np
>>> from scorio import rank
>>>
>>> # Generate sample data: 3 models, 5 questions, 4 trials
>>> R = np.random.randint(0, 2, size=(3, 5, 4))
>>>
>>> # Get rankings using different methods
>>> ranks_elo = rank.elo(R)
>>> ranks_bt = rank.bradley_terry(R)
>>> ranks_bayes = rank.bayes(R, w=np.array([0.0, 1.0]))
>>>
>>> # Get raw scores/ratings when needed
>>> _, scores_elo = rank.elo(R, return_scores=True)
>>>
>>> # Evaluation metric-based ranking
>>> ranks_pass_at_k = rank.pass_at_k(R, k=2)
>>>
>>> # Convert raw scores to alternative rank variants if needed
>>> from scorio.utils import rank_scores
>>> ranks_dense = rank_scores(scores_elo)["dense"]
"""

# Bayesian methods
from .bayesian import bayesian_mcmc, thompson

# Paired-comparison probabilistic models
from .bradley_terry import (
    bradley_terry,
    bradley_terry_davidson,
    bradley_terry_davidson_map,
    bradley_terry_map,
    rao_kupper,
    rao_kupper_map,
)

# Evaluation metric-based ranking methods
from .eval_ranking import (
    bayes,
    g_pass_at_k_tau,
    mean,
    mg_pass_at_k,
    pass_at_k,
    pass_hat_k,
)

# Graph-based methods
from .graph import alpharank, nash, pagerank, spectral

# Hodge-theoretic
from .hodge_rank import hodge_rank

# Item Response Theory
from .irt import (
    dynamic_irt,
    rasch,
    rasch_2pl,
    rasch_2pl_map,
    rasch_3pl,
    rasch_3pl_map,
    rasch_map,
    rasch_mml,
    rasch_mml_credible,
)

# Listwise / setwise choice models
from .listwise import (
    bradley_terry_luce,
    bradley_terry_luce_map,
    davidson_luce,
    davidson_luce_map,
    plackett_luce,
    plackett_luce_map,
)

# Pairwise rating systems
from .pairwise import elo, glicko, trueskill

# Pointwise methods
from .pointwise import inverse_difficulty

# Prior classes for MAP estimation
from .priors import (
    CauchyPrior,
    CustomPrior,
    EmpiricalPrior,
    GaussianPrior,
    LaplacePrior,
    Prior,
    UniformPrior,
)
from .rank_centrality import rank_centrality

# Seriation-based
from .serial_rank import serial_rank

# Voting methods
from .voting import (
    baldwin,
    borda,
    copeland,
    kemeny_young,
    majority_judgment,
    minimax,
    nanson,
    ranked_pairs,
    schulze,
    win_rate,
)

__all__ = [
    # Eval-based
    "mean",
    "pass_at_k",
    "pass_hat_k",
    "g_pass_at_k_tau",
    "mg_pass_at_k",
    "bayes",
    # Accuracy
    "inverse_difficulty",
    # Pairwise
    "elo",
    "glicko",
    "trueskill",
    # Bradley-Terry
    "bradley_terry",
    "bradley_terry_map",
    "bradley_terry_davidson",
    "bradley_terry_davidson_map",
    # Rao-Kupper
    "rao_kupper",
    "rao_kupper_map",
    # Prior classes
    "Prior",
    "GaussianPrior",
    "LaplacePrior",
    "CauchyPrior",
    "UniformPrior",
    "CustomPrior",
    "EmpiricalPrior",
    # Bayesian
    "thompson",
    "bayesian_mcmc",
    # Voting
    "borda",
    "copeland",
    "win_rate",
    "minimax",
    "schulze",
    "ranked_pairs",
    "kemeny_young",
    "nanson",
    "baldwin",
    "majority_judgment",
    # IRT
    "rasch",
    "rasch_map",
    "rasch_2pl",
    "rasch_2pl_map",
    "rasch_3pl",
    "rasch_3pl_map",
    "rasch_mml",
    "rasch_mml_credible",
    "dynamic_irt",
    # Graph
    "pagerank",
    "spectral",
    "rank_centrality",
    "alpharank",
    "nash",
    # Seriation-based
    "serial_rank",
    # Hodge-theoretic
    "hodge_rank",
    # Plackett-Luce
    "plackett_luce",
    "plackett_luce_map",
    "davidson_luce",
    "davidson_luce_map",
    "bradley_terry_luce",
    "bradley_terry_luce_map",
]
