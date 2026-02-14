"""Scalar evaluation metrics and uncertainty estimators.

This module implements evaluation methods for model response tensors
 (``R``) used in ``scorio`` evaluation workflows.

Notation
--------
Let :math:`R \\in \\{0,\\ldots,C\\}^{M \\times N}` be an outcome matrix.

- :math:`M` is the number of questions.
- :math:`N` is the number of trials per question.
- :math:`R_{\\alpha i}` is the outcome for question :math:`\\alpha` on trial
  :math:`i`.

Binary metrics use :math:`R \\in \\{0,1\\}^{M \\times N}`.

Return Pattern
---------------------
Point estimators return a scalar score. Companion ``*_ci`` functions return
``(mu, sigma, lo, hi)``, where ``mu`` is the estimated score, ``sigma`` is the
posterior standard deviation under the method assumptions, and ``lo`` and
``hi`` define a normal-approximation credible interval.

Available Families
------------------
- Bayes family: ``bayes`` and ``bayes_ci``.
- Average family: ``avg`` and ``avg_ci``.
- Pass family: ``pass_at_k``, ``pass_hat_k``, ``g_pass_at_k``,
  ``g_pass_at_k_tau``, ``mg_pass_at_k``, and their ``*_ci`` variants.
"""

from .avg import avg, avg_ci
from .bayes import bayes, bayes_ci
from .pass_at_k import (
    g_pass_at_k,
    g_pass_at_k_ci,
    g_pass_at_k_tau,
    g_pass_at_k_tau_ci,
    mg_pass_at_k,
    mg_pass_at_k_ci,
    pass_at_k,
    pass_at_k_ci,
    pass_hat_k,
    pass_hat_k_ci,
)

__all__ = [
    # Bayes@N
    "bayes",
    "bayes_ci",
    # Avg@N with Bayesian uncertainty
    "avg",
    "avg_ci",
    # Pass-family point metrics
    "pass_at_k",
    "pass_hat_k",
    "g_pass_at_k",
    "g_pass_at_k_tau",
    "mg_pass_at_k",
    # Pass-family Bayesian uncertainty
    "pass_at_k_ci",
    "pass_hat_k_ci",
    "g_pass_at_k_ci",
    "g_pass_at_k_tau_ci",
    "mg_pass_at_k_ci",
]
