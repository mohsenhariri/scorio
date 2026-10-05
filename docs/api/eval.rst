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
TailPass returns a profile object; its utility methods return scalar summaries.
Its interval methods also support equal-tailed posterior Monte Carlo intervals.

.. currentmodule:: scorio.eval

.. _tailpass-profiles-and-utilities-python:

TailPass profiles and utilities
-------------------------------

TailPass estimates the probability that the average score over ``k`` future
attempts meets each threshold. With the default thresholds, binary profiles
report the probabilities of at least 1, ..., k successes. ``k`` is the number
of future attempts; the observed trial count ``N`` supplies evidence and may
differ from ``k``.

Use a utility method to turn the profile into a scalar score. For nonnegative
threshold weights summing to one, ``linear(weights)`` gives
``mean = weights @ profile.mean`` and
``variance = weights @ profile.covariance @ weights``. On the default binary
grid, uniform weights recover Bayes@N with the same prior. Weighting only the
first or last threshold gives the posterior probability of at least one
success (discovery) or all ``k`` successes (stability).

``moment(lam)`` computes the expected payoff ``(average score)**lam``. The
power is applied to each possible future score before averaging over future
outcomes and the posterior.

.. code-block:: python

   import numpy as np
   from scorio import eval
   from scorio.eval.tailpass_weights import moment_weights

   R = np.array([[0, 1, 1, 0], [1, 1, 1, 1]])
   profile = eval.tailpass(R, k=8)
   weights = moment_weights(k=8, lam=2)
   mu, sigma = profile.linear(weights)
   assert np.allclose((mu, sigma), profile.moment(lam=2))

   # Reuse the draws to compare utilities.
   draws = profile.sample(n_draws=2000, rng=42)
   moment_summary = draws.summary(draws.moment(2))
   qrs_summary = draws.summary(draws.qrs())
   difference_summary = draws.summary(draws.moment(2) - draws.moment(4))
   other_budget = draws.at_k(16)  # reuse the draws for 16 future attempts

The posterior concentrations are ``prior + counts(R) + eta * counts(R0)``.
The default prior concentration is one per original category. ``prior`` accepts
a scalar, a category vector, or a question-by-category matrix. Binary
concentrations are ordered as ``[failure, success]``. ``eta`` sets the fraction
of auxiliary counts included, from zero to one. Use the same evidence and prior
when comparing utilities.

For categorical outcomes, pass rubric scores ``w`` in [0, 1]. ``weights``
refers to utility weights on the reported thresholds. ``moment`` accounts for
every attainable average score, including partial credit between thresholds.
For example, a deterministic score of 0.5 at ``k=1`` has first-moment utility
0.5 even though it never reaches the default threshold 1. ``discovery()``
measures the probability of strictly positive credit, including credit below
the first default threshold; ``stability()`` requires every attempt to receive
score 1.

Nonlinear utilities are evaluated for each posterior draw. By default,
``power_mean`` and ``qrs`` apply the transform to each question's profile, then
average across questions. Use ``aggregation="profile"`` to average the profiles
before applying the transform. ``rollout(m)`` reports the probability that
``m`` independent banks meet a common threshold, averaged using threshold
weights. ``harmonic(alpha)`` computes a weighted harmonic mean of discovery
and stability probabilities. ``shortfall(target)`` measures how far profiles
fall below a target (lower is better). Spectrum weights apply to the selected
discrete threshold grid, including for categorical rubrics.

``ci``, ``linear_ci``, and ``moment_ci`` default to ``method="mc"``. All four
returned statistics are Monte Carlo estimates. ``method="normal"`` uses exact
moments and Gaussian intervals clipped to [0, 1]. Call ``sample`` once to reuse
draws across utilities. Profile intervals describe uncertainty in expected
performance on the observed questions, separately for each threshold. They
are not simultaneous bands or predictive intervals for future banks.

``tailpass_empirical`` samples without replacement from ``R`` and requires
``k <= N``. It returns an array of threshold probabilities. For binary outcomes
on the default grid, averaging the entries gives observed accuracy. The first
and last entries match finite-bank ``pass_at_k`` and ``pass_hat_k``.

General categorical profiles are computed by enumerating category-count states.
Requests above 20,000 states raise ``ValueError`` before allocation. Exact
covariance is limited to 4,000,000 count pairs or output entries. Use posterior
draws when exact covariance exceeds this limit; general profiles still require
a count grid within the state limit. Categorical moments with ``lam`` equal to
1, 2, or 4 use a low-degree polynomial expansion. These moments and
discovery/stability utilities avoid the full count grid in both exact and
Monte Carlo calculations.

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
