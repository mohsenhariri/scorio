scorio.eval
===========

Evaluation metrics for outcome matrices. Import the public API with
``from scorio import eval`` and call functions as ``eval.bayes(...)``,
``eval.pass_at_k(...)``, and so on.

Shared conventions
------------------

``R`` is an :math:`M \times N` matrix, where rows are questions and columns
are sampled trials. Binary metrics require entries in :math:`\{0,1\}`.
Categorical metrics use entries in :math:`\{0,\ldots,C\}` together with a
weight or reward vector ``w`` of length :math:`C+1`.

Point estimators return a scalar score. Companion ``*_ci`` functions return
``(mu, sigma, lo, hi)``, where ``mu`` is the posterior mean or point estimate,
``sigma`` is the posterior standard deviation under the metric's uncertainty
model, and ``lo`` and ``hi`` define a normal-approximation credible interval.
TailPass returns a profile object; its utility methods supply scalar summaries.
Its interval methods also support equal-tailed posterior Monte Carlo intervals.

.. currentmodule:: scorio.eval

TailPass profiles and utilities (Python)
------------------------------------------------------------

TailPass reports the probability of reaching each required average score in a
fresh bank of ``k`` attempts. For binary outcomes, its coordinates require
1, ..., k successes. The observed number of trials supplies evidence; the
reporting budget ``k`` may be larger or smaller.

Choose a utility explicitly to produce a scalar. Convex threshold weights
give ``mean = weights @ profile.mean`` and
``variance = weights @ profile.covariance @ weights``. Uniform binary weights
recover Bayes@N with the same prior; endpoint weights recover posterior
discovery and stability. ``moment(lam)`` values the average bank score raised
to ``lam``. It integrates that payoff over future outcomes and then over the
posterior, rather than raising a posterior mean to that power.

.. code-block:: python

   import numpy as np
   from scorio import eval
   from scorio.eval.tailpass_weights import moment_weights

   R = np.array([[0, 1, 1, 0], [1, 1, 1, 1]])
   profile = eval.tailpass(R, k=8)
   weights = moment_weights(k=8, lam=2)
   mu, sigma = profile.linear(weights)
   assert np.allclose((mu, sigma), profile.moment(lam=2))

   # One posterior sample supports multiple utilities and their differences.
   draws = profile.sample(n_draws=2000, rng=42)
   moment_summary = draws.summary(draws.moment(2))
   qrs_summary = draws.summary(draws.qrs())
   difference_summary = draws.summary(draws.moment(2) - draws.moment(4))
   other_budget = draws.at_k(16)  # preserves the same latent draws

``tailpass`` uses ``prior + counts(R) + eta * counts(R0)``. The default
concentration is one per original category. ``prior`` can supply a scalar,
category vector, or question-by-category matrix. Binary concentrations are
ordered as ``[failure, success]``. The transfer weight ``eta`` is in [0, 1].
Keep this evidence model fixed when comparing utility preferences.

For categorical outcomes, pass rubric scores ``w`` in [0, 1]. ``weights``
always refers to utility weights on the reported thresholds. Categorical
``moment`` integrates the full attainable score distribution: partial credit
between reporting thresholds is retained. For example, a deterministic score
of 0.5 at ``k=1`` has first-moment utility 0.5 even though it never reaches the
default reporting threshold 1. ``discovery()`` measures strictly positive
credit, including credit below the first grid threshold; ``stability()``
requires every attempt to receive score 1.

All nonlinear summaries operate on each posterior draw. ``power_mean`` and
``qrs`` average question-level summaries by default. Specify
``aggregation="profile"`` to transform the already averaged profile instead.
``rollout(m)`` keeps the replication-probability scale, ``harmonic(alpha)``
balances discovery and stability, and ``shortfall(target)`` measures violations
of a target profile (lower is better). Spectrum weights refer to the chosen
discrete reporting grid, including for categorical rubrics.

``ci``, ``linear_ci``, and ``moment_ci`` default to ``method="mc"``: all four
returned statistics are Monte Carlo estimates. ``method="normal"`` uses exact
moments and clipped Gaussian intervals. Reusing ``sample`` avoids resampling
for each utility. Intervals concern the latent expected performance on the
observed questions. They are coordinatewise credible intervals, not
simultaneous bands or predictive intervals for newly realized banks.

The explicitly empirical ``tailpass_empirical`` samples without replacement
from R and requires ``k <= N``. It returns an array. Dotting this binary
profile with uniform weights gives observed accuracy; its first and last
coordinates match finite-bank ``pass_at_k`` and ``pass_hat_k``.

Exact general categorical calculations enumerate count states. Requests above
20,000 states raise ``ValueError`` before allocation; exact covariance is also
limited to 4,000,000 count pairs or output entries. Shared draws provide a
covariance alternative when the count grid fits, but still require that grid
for general profiles. Categorical moments 1, 2, and 4 use a separate low-degree
expansion; their draws, and discovery/stability, avoid the full count grid.

.. autofunction:: tailpass

.. autofunction:: tailpass_empirical

.. autoclass:: scorio.eval.tailpass.TailPassProfile
   :members:

.. autoclass:: scorio.eval.tailpass.TailPassDraws
   :members:

.. automodule:: scorio.eval.tailpass_weights
   :members:

.. currentmodule:: scorio.eval


Bayes@N
-------

.. autofunction:: bayes

.. autofunction:: bayes_ci


avg@N
-----

.. autofunction:: avg

.. autofunction:: avg_ci


Pass@k
------

``unanimous_at_k`` and ``unanimous_at_k_ci`` are aliases for
``pass_hat_k`` and ``pass_hat_k_ci``.

.. autofunction:: pass_at_k

.. autofunction:: pass_hat_k

.. autofunction:: pass_at_k_ci

.. autofunction:: pass_hat_k_ci


AUC@K
------

.. autofunction:: auc_at_k

.. autofunction:: auc_at_k_ci


Majority
--------

.. autofunction:: maj_at_k

.. autofunction:: maj_at_k_ci


Generalized Pass@k
------------------

.. autofunction:: g_pass_at_k

.. autofunction:: g_pass_at_k_tau

.. autofunction:: mg_pass_at_k

.. autofunction:: g_pass_at_k_ci

.. autofunction:: g_pass_at_k_tau_ci

.. autofunction:: mg_pass_at_k_ci

GeoSpectrum
------------------

.. autofunction:: geom_at_k

.. autofunction:: geom_at_k_ci

.. autofunction:: geom_ds_at_k

.. autofunction:: geom_ds_at_k_ci

.. autofunction:: geo_spectrum_at_k

.. autofunction:: geo_spectrum_at_k_ci

.. autofunction:: geo_spectrum_star_at_k

.. autofunction:: geo_spectrum_star_at_k_ci

.. autofunction:: threshold_spectrum_at_k

.. autofunction:: threshold_spectrum_at_k_ci


Max-Reward
----------

.. autofunction:: max_at_k

.. autofunction:: max_at_k_ci
