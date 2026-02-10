"""Ranking methods for comparing multiple models.

This module provides a collection of ranking methods for evaluating and comparing
LLMs used in "Ranking Reasoning LLMs under Test-Time Scaling".

All ranking methods expect a 3D numpy array :math:`R` of shape :math:`(L, M, N)` where:

- :math:`L` = number of models to rank
- :math:`M` = number of questions
- :math:`N` = number of independent trials per question (e.g., using top-p sampling)

For longitudinal variants of :func:`scorio.rank.dynamic_irt` (``variant="growth"``
or ``variant="state_space"``), axis :math:`N` is interpreted as ordered
measurement occasions instead of i.i.d. trials.

The entry :math:`R_{lmn} = 1` if model :math:`l` answered question :math:`m` correctly
on trial :math:`n`, else :math:`R_{lmn} = 0`. For single-trial scenarios (:math:`N=1`),
you can pass a 2D array of shape :math:`(L, M)` which will be automatically converted
to :math:`(L, M, 1)`.

Ranking methods compute raw model scores (e.g., estimated strengths :math:`\\pi_l`,
posterior means, or metric-based scores) and derive rankings from those scores. By
default, methods return a 1D numpy array of shape :math:`(L,)` containing the ranking
(using the 'competition' ranking from `scorio.utils.rank_scores`). Set
`return_scores=True` to return a tuple `(ranking, scores)` where `ranking` is a 1D
array of ranks and `scores` is the corresponding raw scores. To convert raw scores to
different ranking variants, use `scorio.utils.rank_scores()` which returns a mapping
of ranking variants (e.g., 'competition', 'dense', 'ordinal').

Available Methods
-----------------

Evaluation metric-based: `mean`, `bayes`, `pass_at_k`, `pass_hat_k`,
`g_pass_at_k_tau`, `mg_pass_at_k`.

Paired-comparison probabilistic models: `bradley_terry`, `bradley_terry_map`,
`bradley_terry_davidson`, `bradley_terry_davidson_map`, `rao_kupper`, `rao_kupper_map`.

Accuracy-based: `inverse_difficulty`.

Pairwise rating systems: `elo`, `glicko`, `trueskill`.

Bayesian methods: `thompson`, `bayesian_mcmc`.

Voting methods: `borda`, `copeland`, `win_rate`, `minimax`, `schulze`,
`ranked_pairs`, `kemeny_young`, `nanson`, `baldwin`, `majority_judgment`.

Item Response Theory: `rasch`, `rasch_map`, `rasch_2pl`, `rasch_2pl_map`,
`rasch_3pl`, `rasch_3pl_map`, `rasch_mml`, `rasch_mml_credible`, `dynamic_irt`.

Graph-based: `pagerank`, `spectral`, `rank_centrality`, `alpharank`, `nash`.

Seriation-based: `serial_rank`.

Hodge-theoretic: `hodge_rank`.

Listwise/setwise choice models (Luce family): `plackett_luce`, `plackett_luce_map`,
`davidson_luce`, `davidson_luce_map`, `bradley_terry_luce`, `bradley_terry_luce_map`.

Examples
--------

>>> import numpy as np
>>> from scorio import rank
>>> # Generate sample data: 3 models, 5 questions, 4 trials
>>> R = np.random.randint(0, 2, size=(3, 5, 4))
>>> # Get rankings using different methods
>>> ranks_bayes = rank.bayes(R, w=np.array([0.0, 1.0]))
>>> ranks_pass_at_k = rank.pass_at_k(R, k=2)
>>> # Get raw scores when needed
>>> ranking, scores = rank.mean(R, return_scores=True)
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

# # Prior classes for MAP estimation
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
    # Priors
    "Prior",
    "GaussianPrior",
    "LaplacePrior",
    "CauchyPrior",
    "UniformPrior",
    "CustomPrior",
    "EmpiricalPrior",
    # Eval-based
    "mean",
    "bayes",
    "pass_at_k",
    "pass_hat_k",
    "g_pass_at_k_tau",
    "mg_pass_at_k",
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
    "alpharank",
    "nash",
    "rank_centrality",
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
