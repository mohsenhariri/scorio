"""TailPass posterior profiles and utilities for repeated sampling.

TailPass describes the probability of reaching each required bank score under
fresh, conditionally independent attempts. A profile has no default scalar
score: call ``linear(weights)`` or select a named utility such as ``moment``.

Reference:
    Hariri, Hinczewski, Ayday, and Chaudhary (2026), *Success Has a Shape:
    TailPass@k for Repeated Sampling Evaluation*, Sections 2–3 and Appendix C.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp
from scipy.stats import hypergeom, multivariate_hypergeom

from ._categorical import _readonly_float64, prepare_categorical_bank
from ._count_score import CountScore, _real_vector
from ._inputs import (
    _finite_positive_scalar,
    _integral_scalar,
    validate_finite_k,
    validate_latent_k,
)
from ._posterior import _endpoint_kind, _state_moments
from ._tailpass import (
    MAX_COUNT_STATES,
    MAX_COVARIANCE_PAIRS,
    conditional_integer_moment,
    conditional_probabilities,
    count_covariance,
    count_grid,
    endpoint_moments,
    log_coefficients,
    moment_polynomial,
    nonnegative_variance,
    polynomial_moments,
    predictive_probabilities,
)
from .tailpass_weights import _validate_weights, uniform_weights
from .utils import _z_value, normal_credible_interval

FloatArray = NDArray[np.float64]
RNG = int | np.random.Generator | None
Summary = (
    tuple[float, float, float, float]
    | tuple[FloatArray, FloatArray, FloatArray, FloatArray]
)


def _unit_interval(value: float, name: str) -> float:
    array = _real_vector(value, name=name)
    if array.ndim != 0 or not 0.0 <= float(array) <= 1.0:
        raise ValueError(f"{name} must be a scalar in [0, 1]")
    return float(array)


def _thresholds(k: int, thresholds: ArrayLike | None) -> FloatArray:
    if thresholds is None:
        if k > MAX_COUNT_STATES:
            raise ValueError(
                "Default threshold grid is too large; supply explicit thresholds"
            )
        return _readonly_float64(np.arange(1, k + 1, dtype=float) / k)
    values = _real_vector(thresholds, name="thresholds")
    if (
        values.ndim != 1
        or values.size == 0
        or np.any((values < 0) | (values > 1))
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(
            "thresholds must be a non-empty, strictly increasing vector in [0, 1]"
        )
    return _readonly_float64(values)


def _bank_scores(counts: NDArray[np.int64], scores: FloatArray, k: int) -> FloatArray:
    # Round once to float64, including when a rubric score is itself a fraction.
    return np.asarray(
        counts.astype(np.longdouble) @ scores.astype(np.longdouble) / k, dtype=float
    )


def _normal_summary(
    mu: float, sigma: float, confidence: float
) -> tuple[float, float, float, float]:
    lo, hi = normal_credible_interval(
        mu, sigma, credibility=confidence, bounds=(0.0, 1.0)
    )
    return mu, sigma, lo, hi


def tailpass(
    R: ArrayLike,
    k: int,
    w: ArrayLike | None = None,
    R0: ArrayLike | None = None,
    *,
    eta: float = 1.0,
    prior: ArrayLike = 1.0,
    thresholds: ArrayLike | None = None,
) -> TailPassProfile:
    """Prepare a reusable posterior TailPass profile.

    Args:
        R: Question-by-trial categorical outcomes; 1D means one question.
        k: Positive reporting budget, independent of the observed trial count.
        w: Category scores in [0, 1]. Defaults to [0, 1] for binary R.
        R0: Optional auxiliary outcomes for the same questions, as in ``bayes``.
        eta: Auxiliary-count transfer weight in [0, 1].
        prior: Positive Dirichlet concentrations: a scalar, a length-(C+1)
            vector, or an M-by-(C+1) matrix for categories 0, ..., C.
            Entries follow category order; binary [failure, success] = [1, 1]
            is the default Beta(1, 1) prior.
        thresholds: Required average rubric scores in [0, 1], in increasing
            order. Defaults to 1/k, ..., 1. Threshold zero is always met;
            discovery means strictly positive credit instead.

    Returns:
        A profile with exact moments, scalar utilities, and posterior sampling.

    Notes:
        The posterior is ``prior + counts(R) + eta * counts(R0)``. Questions
        are independent under this model. R0 should be disjoint evidence.
        Equal-score categories are merged by adding their concentrations,
        preserving the original Dirichlet prior mass.

    Examples:
        >>> from scorio import eval
        >>> profile = eval.tailpass([[0, 1, 1], [1, 1, 1]], k=4)
        >>> profile.mean.shape
        (4,)
        >>> mu, sigma = profile.moment(lam=1)
        >>> round(mu, 3)
        0.7
    """
    k = validate_latent_k(k)
    grid = _thresholds(k, thresholds)
    bank = prepare_categorical_bank(R, w=w, R0=R0)
    if np.any((bank.weights < 0) | (bank.weights > 1)):
        raise ValueError("TailPass rubric scores w must lie in [0, 1]")
    eta = _unit_interval(eta, "eta")
    base = _real_vector(prior, name="prior")
    if base.ndim > 2 or np.any(base <= 0):
        raise ValueError("prior must contain positive Dirichlet concentrations")
    try:
        base = np.broadcast_to(base, bank.counts.shape)
    except ValueError as exc:
        raise ValueError(
            "prior must be a scalar, category vector, or question-by-category matrix"
        ) from exc
    with np.errstate(over="ignore"):
        parameters = base + bank.counts + eta * bank.prior_counts
        totals = parameters.sum(axis=1)
    if not np.all(np.isfinite(totals)):
        raise ValueError("Posterior concentration totals must be finite")
    scores, inverse = np.unique(bank.weights, return_inverse=True)
    grouped = np.zeros((bank.question_count, len(scores)))
    for category, group in enumerate(inverse):
        grouped[:, group] += parameters[:, category]
    return TailPassProfile(
        k, grid, _readonly_float64(grouped), _readonly_float64(scores)
    )


@dataclass(frozen=True, eq=False)
class TailPassProfile:
    """Posterior-backed profile constructed by :func:`tailpass`.

    ``mean`` and ``std`` have one entry per threshold; ``covariance`` describes
    their joint uncertainty. ``question_mean`` retains the question axis.
    Expensive arrays are computed on first access and stored read-only.
    """

    k: int
    thresholds: FloatArray
    _parameters: FloatArray = field(repr=False)
    _scores: FloatArray = field(repr=False)

    @property
    def question_count(self) -> int:
        """Number of independently modeled questions."""
        return len(self._parameters)

    @cached_property
    def _states(self) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.int64]]:
        return np.unique(
            self._parameters, axis=0, return_inverse=True, return_counts=True
        )

    @cached_property
    def _grid(self) -> tuple[NDArray[np.int64], FloatArray, FloatArray]:
        counts = count_grid(self.k, len(self._scores))
        return (
            counts,
            _bank_scores(counts, self._scores, self.k),
            log_coefficients(counts),
        )

    @cached_property
    def _events(self) -> FloatArray:
        return (self._grid[1][:, None] >= self.thresholds).astype(float)

    @cached_property
    def _tail_map(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        order = np.argsort(self._grid[1], kind="stable")
        return order, np.searchsorted(self._grid[1][order], self.thresholds)

    def _tails(self, pmf: FloatArray) -> FloatArray:
        order, indices = self._tail_map
        tails = np.cumsum(pmf[..., order][..., ::-1], axis=-1)[..., ::-1]
        padded = np.concatenate((tails, np.zeros((*tails.shape[:-1], 1))), axis=-1)
        return padded[..., indices]

    @cached_property
    def _state_means(self) -> FloatArray:
        counts, _, log_coeff = self._grid
        return np.array(
            [
                self._tails(predictive_probabilities(counts, alpha, log_coeff))
                for alpha in self._states[0]
            ]
        )

    @cached_property
    def question_mean(self) -> FloatArray:
        """Exact per-question mean profiles, shape (M, number of thresholds)."""
        return _readonly_float64(self._state_means[self._states[1]])

    @cached_property
    def mean(self) -> FloatArray:
        """Exact posterior mean of the dataset profile."""
        return _readonly_float64(
            self._states[2] @ self._state_means / self.question_count
        )

    @cached_property
    def covariance(self) -> FloatArray:
        """Exact joint posterior covariance of the dataset profile.

        Independent question covariances are summed and divided by M squared.
        Exact enumeration has size limits; posterior draws provide an explicit
        Monte Carlo alternative when the covariance calculation is too large.
        """
        counts, _, log_coeff = self._grid
        if max(len(counts), len(self.thresholds)) ** 2 > MAX_COVARIANCE_PAIRS:
            raise ValueError(
                "Exact covariance is too large; use shared posterior draws"
            )
        result = np.zeros((len(self.thresholds), len(self.thresholds)))
        for alpha, frequency in zip(self._states[0], self._states[2], strict=True):
            result += frequency * count_covariance(
                counts, alpha, self._events, log_coeff
            )
        result /= self.question_count**2
        for index in range(len(result)):
            result[index, index] = nonnegative_variance(float(result[index, index]))
        return _readonly_float64(result)

    @cached_property
    def std(self) -> FloatArray:
        """Exact marginal posterior standard deviations."""
        return _readonly_float64(np.sqrt(np.diag(self.covariance)))

    def _payoff_moments(self, values: FloatArray) -> tuple[float, float]:
        counts, _, log_coeff = self._grid
        parameters, _, frequencies = self._states
        if len(self._scores) == 2:
            score = CountScore(self.k, values)
            endpoint = _endpoint_kind(score)
            if endpoint is not None:
                means, variances = endpoint_moments(
                    self.k, parameters[:, 1], parameters[:, 0], endpoint
                )
            else:
                means, variances = _state_moments(
                    score, parameters[:, 1], parameters[:, 0], None
                )
        else:
            means = np.array(
                [
                    predictive_probabilities(counts, a, log_coeff) @ values
                    for a in parameters
                ]
            )
            variances = np.array(
                [
                    count_covariance(counts, a, values[:, None], log_coeff)[0, 0]
                    for a in parameters
                ]
            )
        return self._aggregate(means, variances, frequencies)

    def _aggregate(
        self, means: FloatArray, variances: FloatArray, frequencies: NDArray[np.int64]
    ) -> tuple[float, float]:
        mean = float(frequencies @ means / self.question_count)
        variance = nonnegative_variance(
            float(frequencies @ variances / self.question_count**2)
        )
        return mean, math.sqrt(variance)

    def linear(self, weights: ArrayLike) -> tuple[float, float]:
        """Exact mean/std of a convex combination of the reported thresholds.

        Args:
            weights: One finite nonnegative weight per threshold, summing to one.

        The variance includes all cross-threshold covariances. This method can
        evaluate a single binary payoff without constructing the full covariance.
        """
        weights = _validate_weights(weights, len(self.thresholds))
        cumulative = np.r_[0.0, np.cumsum(weights)]
        payoff = cumulative[
            np.searchsorted(self.thresholds, self._grid[1], side="right")
        ]
        return self._payoff_moments(payoff)

    def moment(self, lam: float) -> tuple[float, float]:
        """Exact mean/std of E[(average future rubric score)**lam].

        Args:
            lam: Finite positive exponent. One gives posterior mean rubric
                credit (Bayes@N with matching priors); larger exponents value
                high bank scores more heavily.

        This integrates the full attainable score distribution, independently
        of the reporting grid. First, second, and fourth categorical moments
        have fast paths that avoid enumerating all k-attempt category counts.
        """
        lam = _finite_positive_scalar(lam, name="lam")
        parameters, _, frequencies = self._states
        if lam == 1.0:
            totals = parameters.sum(axis=1)
            probabilities = parameters / totals[:, None]
            means = probabilities @ self._scores
            variances = np.sum(
                probabilities * (self._scores - means[:, None]) ** 2, axis=1
            ) / (totals + 1)
            return self._aggregate(means, variances, frequencies)
        if lam in (2.0, 4.0) and (len(self._scores) != 2 or self.k >= MAX_COUNT_STATES):
            powers, coefficients = moment_polynomial(self._scores, self.k, int(lam))
            moments = np.array(
                [polynomial_moments(powers, coefficients, a) for a in parameters]
            )
            return self._aggregate(moments[:, 0], moments[:, 1], frequencies)
        # Binary payoffs are cumulative moment_weights(k, lam). Evaluating the
        # payoff directly also retains categorical credit between grid points.
        return self._payoff_moments(self._grid[1] ** lam)

    def _endpoint(self, full_credit: bool) -> tuple[float, float]:
        mask = self._scores == 1 if full_credit else self._scores > 0
        if np.all(mask) or not np.any(mask):
            return float(np.all(mask)), 0.0
        parameters, _, frequencies = self._states
        means, variances = endpoint_moments(
            self.k,
            parameters[:, mask].sum(axis=1),
            parameters[:, ~mask].sum(axis=1),
            "unanimous" if full_credit else "pass",
        )
        return self._aggregate(means, variances, frequencies)

    def discovery(self) -> tuple[float, float]:
        """Exact mean/std of positive bank credit (the moment limit at 0+).

        For categorical rubrics this can differ from the first reported grid
        coordinate: even credit below 1/k counts as discovery.
        """
        return self._endpoint(False)

    def stability(self) -> tuple[float, float]:
        """Exact mean/std of every attempt receiving full rubric credit 1."""
        return self._endpoint(True)

    def at_k(self, k: int, *, thresholds: ArrayLike | None = None) -> TailPassProfile:
        """Reuse the fitted posterior at another positive reporting budget."""
        k = validate_latent_k(k)
        return TailPassProfile(
            k, _thresholds(k, thresholds), self._parameters, self._scores
        )

    def sample(self, n_draws: int = 4000, *, rng: RNG = None) -> TailPassDraws:
        """Draw independent question probabilities, shared across all utilities.

        Args:
            n_draws: At least two posterior draws.
            rng: NumPy Generator, integer seed, or None.

        Stores latent probabilities (draws by questions by distinct scores),
        rather than a potentially much larger tensor of question profiles.
        """
        n_draws = _integral_scalar(n_draws, name="n_draws")
        if n_draws < 2:
            raise ValueError("n_draws must be at least two")
        generator = np.random.default_rng(rng)
        values = np.empty((n_draws, *self._parameters.shape))
        for question, alpha in enumerate(self._parameters):
            values[:, question] = (
                1.0 if len(alpha) == 1 else generator.dirichlet(alpha, size=n_draws)
            )
        values.setflags(write=False)
        return TailPassDraws(self, values)

    def ci(
        self,
        confidence: float = 0.95,
        *,
        method: Literal["mc", "normal"] = "mc",
        n_draws: int = 4000,
        rng: RNG = None,
    ) -> Summary:
        """Return profile (mean, std, lo, hi), one entry per threshold.

        ``mc`` uses shared draws for all four summaries and equal-tailed
        pointwise credible intervals. ``normal`` uses exact moments and clipped
        Gaussian intervals. Neither is a simultaneous profile credible band.
        """
        z = _z_value(confidence)
        if method == "mc":
            return self.sample(n_draws, rng=rng).summary(confidence=confidence)
        if method != "normal":
            raise ValueError("method must be 'mc' or 'normal'")
        return (
            self.mean,
            self.std,
            np.clip(self.mean - z * self.std, 0, 1),
            np.clip(self.mean + z * self.std, 0, 1),
        )

    def linear_ci(
        self,
        weights: ArrayLike,
        confidence: float = 0.95,
        *,
        method: Literal["mc", "normal"] = "mc",
        n_draws: int = 4000,
        rng: RNG = None,
    ) -> Summary:
        """Linear-utility summary; interval methods follow :meth:`ci`."""
        weights = _validate_weights(weights, len(self.thresholds))
        _z_value(confidence)
        if method == "normal":
            return _normal_summary(*self.linear(weights), confidence)
        if method != "mc":
            raise ValueError("method must be 'mc' or 'normal'")
        draws = self.sample(n_draws, rng=rng)
        return draws.summary(draws.linear(weights), confidence=confidence)

    def moment_ci(
        self,
        lam: float,
        confidence: float = 0.95,
        *,
        method: Literal["mc", "normal"] = "mc",
        n_draws: int = 4000,
        rng: RNG = None,
    ) -> Summary:
        """Moment-utility summary; interval methods follow :meth:`ci`."""
        lam = _finite_positive_scalar(lam, name="lam")
        _z_value(confidence)
        if method == "normal":
            return _normal_summary(*self.moment(lam), confidence)
        if method != "mc":
            raise ValueError("method must be 'mc' or 'normal'")
        draws = self.sample(n_draws, rng=rng)
        return draws.summary(draws.moment(lam), confidence=confidence)


@dataclass(frozen=True, eq=False)
class TailPassDraws:
    """Shared posterior draws created by :meth:`TailPassProfile.sample`.

    Utility methods return one scalar per draw. Use ``summary(values)`` for
    the Monte Carlo mean, standard deviation, and equal-tailed interval. Draws
    describe latent expected performance; they do not simulate future banks.
    """

    _source: TailPassProfile = field(repr=False)
    _probabilities: FloatArray = field(repr=False)

    def _chunks(self) -> Iterator[tuple[int, int, FloatArray]]:
        counts, _, log_coeff = self._source._grid
        batch = max(1, 1_000_000 // max(len(counts), len(self._source.thresholds)))
        for question in range(self._source.question_count):
            for start in range(0, len(self._probabilities), batch):
                stop = start + batch
                pmf = conditional_probabilities(
                    counts, self._probabilities[start:stop, question], log_coeff
                )
                yield start, start + len(pmf), pmf

    @cached_property
    def profile(self) -> FloatArray:
        """Dataset profile samples, shape (draws, thresholds)."""
        values = np.zeros((len(self._probabilities), len(self._source.thresholds)))
        for start, stop, pmf in self._chunks():
            values[start:stop] += self._source._tails(pmf)
        return _readonly_float64(values / self._source.question_count)

    def linear(self, weights: ArrayLike) -> FloatArray:
        """Apply a convex combination to every shared profile draw."""
        weights = _validate_weights(weights, len(self._source.thresholds))
        return self.profile @ weights

    def moment(self, lam: float) -> FloatArray:
        """Draws of the full-score moment utility, independent of the grid."""
        lam = _finite_positive_scalar(lam, name="lam")
        values = np.zeros(len(self._probabilities))
        if lam in (1.0, 2.0, 4.0):
            for question in range(self._source.question_count):
                values += conditional_integer_moment(
                    self._probabilities[:, question],
                    self._source._scores,
                    self._source.k,
                    int(lam),
                )
        else:
            payoff = self._source._grid[1] ** lam
            for start, stop, pmf in self._chunks():
                values[start:stop] += pmf @ payoff
        return values / self._source.question_count

    def _apply(
        self,
        transform: Callable[[FloatArray], FloatArray],
        aggregation: str = "question",
    ) -> FloatArray:
        if aggregation == "profile":
            return transform(self.profile)
        if aggregation != "question":
            raise ValueError("aggregation must be 'question' or 'profile'")
        values = np.zeros(len(self._probabilities))
        for start, stop, pmf in self._chunks():
            values[start:stop] += transform(self._source._tails(pmf))
        return values / self._source.question_count

    def power_mean(
        self,
        q: float,
        weights: ArrayLike | None = None,
        *,
        aggregation: Literal["question", "profile"] = "question",
    ) -> FloatArray:
        """Weighted spectrum power means, computed question-first by default.

        Args:
            q: Positive power-mean exponent (distinct from moment utility lam).
            weights: Convex weights on the reported thresholds; default uniform.
            aggregation: Apply the nonlinear transform before (question) or
                after (profile) averaging questions. These differ for q != 1.
        """
        q = _finite_positive_scalar(q, name="q")
        size = len(self._source.thresholds)
        weights = _validate_weights(
            uniform_weights(size) if weights is None else weights, size
        )

        def transform(values: FloatArray) -> FloatArray:
            if q == 1.0:
                return values @ weights
            active = weights > 0
            selected = values[:, active]
            scale = selected.max(axis=1)
            normalized = np.divide(
                selected,
                scale[:, None],
                out=np.zeros_like(selected),
                where=scale[:, None] > 0,
            )
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                logs = q * np.log(normalized)
                delta = np.expm1(logs) @ weights[active]
                # log1p retains the geometric-mean limit as q approaches zero.
                log_mean = logsumexp(logs + np.log(weights[active]), axis=1)
                small = np.abs(delta) < 0.25
                log_mean[small] = np.log1p(delta[small])
                result = scale * np.exp(log_mean / q)
            return np.where(scale == 0, 0.0, result)

        return self._apply(transform, aggregation)

    def qrs(
        self,
        weights: ArrayLike | None = None,
        *,
        aggregation: Literal["question", "profile"] = "question",
    ) -> FloatArray:
        """Quadratic RMS spectrum draws; uniform binary weights give L2@k."""
        return self.power_mean(2.0, weights, aggregation=aggregation)

    def rollout(self, m: int, weights: ArrayLike | None = None) -> FloatArray:
        """Common-threshold m-rollout utility, on its probability scale.

        Args:
            m: Integer number of independent k-attempt rollouts, at least two.
            weights: Convex threshold weights; default uniform.
        """
        m = _integral_scalar(m, name="m")
        if m < 2:
            raise ValueError("m must be at least two")
        size = len(self._source.thresholds)
        weights = _validate_weights(
            uniform_weights(size) if weights is None else weights, size
        )
        return self._apply(lambda values: values**m @ weights)

    def harmonic(self, alpha: float = 0.5) -> FloatArray:
        """Question-first harmonic balance of discovery and full credit.

        ``alpha`` in [0, 1] weights discovery; ``1-alpha`` weights stability.
        Discovery means positive bank credit and stability means bank credit 1,
        including for categorical rubrics. This is independent of the grid.
        """
        alpha = _unit_interval(alpha, "alpha")
        values = np.zeros(len(self._probabilities))
        for question in range(self._source.question_count):
            probabilities = self._probabilities[:, question]
            zero = np.clip(
                probabilities[:, self._source._scores == 0].sum(axis=1), 0, 1
            )
            one = np.clip(probabilities[:, self._source._scores == 1].sum(axis=1), 0, 1)
            with np.errstate(divide="ignore"):
                discovery = -np.expm1(self._source.k * np.log(zero))
            stability = one**self._source.k
            if alpha == 1:
                values += discovery
            elif alpha == 0:
                values += stability
            else:
                positive = (discovery > 0) & (stability > 0)
                with np.errstate(over="ignore"):
                    values[positive] += 1.0 / (
                        alpha / discovery[positive] + (1 - alpha) / stability[positive]
                    )
        return values / self._source.question_count

    def discovery(self) -> FloatArray:
        """Shared draws of the probability of positive bank credit."""
        return self.harmonic(1.0)

    def stability(self) -> FloatArray:
        """Shared draws of the probability of full bank credit."""
        return self.harmonic(0.0)

    def shortfall(
        self,
        target: ArrayLike,
        weights: ArrayLike | None = None,
        *,
        epsilon: float = 0.01,
    ) -> FloatArray:
        """Question-first reference-profile shortfall; lower values are better.

        Args:
            target: Nonincreasing target profile in [0, 1].
            weights: Strictly positive preference/normalization weights,
                default all ones. These need not sum to one.
            epsilon: Positive multiplier for the total weighted shortfall.

        Returns max(weighted shortfall) + epsilon * sum(weighted shortfall)
        within each question, then averages questions on each posterior draw.
        """
        size = len(self._source.thresholds)
        target = _real_vector(target, name="target")
        if (
            target.shape != (size,)
            or np.any((target < 0) | (target > 1))
            or np.any(np.diff(target) > 0)
        ):
            raise ValueError(
                "target must be a nonincreasing [0, 1] vector matching thresholds"
            )
        weights = _real_vector(
            np.ones(size) if weights is None else weights, name="weights"
        )
        if weights.shape != (size,) or np.any(weights <= 0):
            raise ValueError("shortfall weights must be positive and match thresholds")
        epsilon = _finite_positive_scalar(epsilon, name="epsilon")

        def transform(values: FloatArray) -> FloatArray:
            gaps = np.maximum(target - values, 0.0) * weights
            return gaps.max(axis=1) + epsilon * gaps.sum(axis=1)

        return self._apply(transform)

    def at_k(self, k: int, *, thresholds: ArrayLike | None = None) -> TailPassDraws:
        """Reuse these same latent draws at a different reporting budget."""
        return TailPassDraws(
            self._source.at_k(k, thresholds=thresholds), self._probabilities
        )

    def summary(
        self, values: ArrayLike | None = None, *, confidence: float = 0.95
    ) -> Summary:
        """Monte Carlo mean/std and equal-tailed interval along the draw axis.

        Args:
            values: Utility draws, or None for the dataset profile draws.
                Differences between utilities from this draw set are supported.
            confidence: Posterior mass strictly between zero and one.

        Intervals are coordinatewise. Values are not clipped, allowing utility
        differences and shortfalls whose ranges need not be [0, 1].
        """
        _z_value(confidence)
        samples = (
            self.profile if values is None else _real_vector(values, name="values")
        )
        if samples.ndim not in (1, 2) or samples.shape[0] != len(self._probabilities):
            raise ValueError(
                "values must have this draw set's sample count on the first axis"
            )
        lo, hi = np.quantile(
            samples, [(1 - confidence) / 2, (1 + confidence) / 2], axis=0
        )
        result = (samples.mean(axis=0), samples.std(axis=0, ddof=1), lo, hi)
        if samples.ndim == 1:
            return tuple(float(value) for value in result)  # type: ignore[return-value]
        return result


def tailpass_empirical(
    R: ArrayLike,
    k: int,
    w: ArrayLike | None = None,
    *,
    thresholds: ArrayLike | None = None,
) -> FloatArray:
    """Finite-bank profile from sampling k observed trials without replacement.

    Args:
        R: Question-by-trial outcomes, or a 1D single-question bank.
        k: Integer satisfying 1 <= k <= N.
        w: Category scores in [0, 1]; defaults to binary [0, 1].
        thresholds: Increasing required bank scores, default 1/k, ..., 1.

    Returns:
        Dataset mean threshold probabilities, without posterior uncertainty.
        Binary endpoints recover finite-bank Pass@k and unanimity. This has
        no R0 or prior arguments: the observed bank defines the distribution.
    """
    bank = prepare_categorical_bank(R, w=w)
    k = validate_finite_k(bank.trial_count, k)
    grid = _thresholds(k, thresholds)
    if np.any((bank.weights < 0) | (bank.weights > 1)):
        raise ValueError("TailPass rubric scores w must lie in [0, 1]")
    observed, scores = bank.grouped_observed_counts()
    counts = count_grid(k, len(scores))
    bank_scores = _bank_scores(counts, scores, k)
    order = np.argsort(bank_scores, kind="stable")
    cutoffs = np.searchsorted(bank_scores[order], grid)
    states, frequencies = np.unique(observed, axis=0, return_counts=True)
    output = np.zeros(len(grid))
    for state, frequency in zip(states, frequencies, strict=True):
        if len(scores) == 2:
            output += frequency * hypergeom.sf(
                cutoffs - 1, bank.trial_count, state[1], k
            )
        else:
            pmf = multivariate_hypergeom.pmf(counts, state, k)
            pmf /= pmf.sum()
            tails = np.r_[np.cumsum(pmf[order][::-1])[::-1], 0.0]
            output += frequency * tails[cutoffs]
    return _readonly_float64(output / bank.question_count)


__all__ = ["tailpass", "tailpass_empirical", "TailPassProfile", "TailPassDraws"]
