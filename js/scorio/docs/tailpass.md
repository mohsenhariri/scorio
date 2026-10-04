---
title: TailPass profiles
---

TailPass returns a vector of threshold probabilities. It has no default scalar
score. For binary outcomes at budget `k`, the default thresholds are
`1/k, 2/k, …, 1`: the first component describes discovering at least one
success, and the last describes succeeding on every trial. The intermediate
components show how reliably the model meets stricter thresholds. The
{@link eval.TailPassProfile | profile reference} lists its properties and methods.

## Fit and inspect a profile

```ts
import { tailpass } from "scorio/eval";

const R = [[0, 1, 1], [1, 1, 1]];
const profile = tailpass(R, 8);

console.log(profile.thresholds);
console.log(profile.mean);         // one posterior mean per threshold
console.log(profile.questionMean); // one threshold profile per question
console.log(profile.std);          // posterior standard deviations
```

The posterior concentrations are `prior + counts(R) + eta × counts(R0)`.
`prior` defaults to `1` and can be a positive scalar, a category vector, or a
question-by-category matrix. The binary category order is failure, success.
`eta` defaults to `1` and lies in `[0, 1]`.

The fitted posterior supports budgets larger than the observed trial count.
That is why the example uses `k = 8` with only three observed trials per
question. `tailpassEmpirical`, the counterpart that samples the observed bank
without replacement, requires `k <= N`.

## Choose a scalar utility

```ts
import { tailpass, tailpassWeights } from "scorio/eval";

const profile = tailpass([[0, 1, 1], [1, 1, 1]], 8);
const weights = tailpassWeights.momentWeights(8, 2);
const [mean, std] = profile.linear(weights);
const [discoveryMean] = profile.discovery();
const [stabilityMean] = profile.stability();

console.log(mean, std, discoveryMean, stabilityMean);
```

`linear(weights)` returns the exact mean and standard deviation of a weighted
threshold utility, including covariance between thresholds. Weights must be
nonnegative and sum to one. The `tailpassWeights` namespace includes uniform,
single-threshold, moment, Beta, maximum-entropy, and payoff weights.

For binary outcomes, `discovery()` is the posterior Pass@k endpoint and
`stability()` is the all-success endpoint. These are scalar summaries of the
profile. The empirical `passAtK` estimate in the
[evaluation guide](evaluation.md) uses a different sampling model.

For categorical data, provide rubric scores in `[0, 1]`. Thresholds then apply
to the mean rubric score of a sampled bank. Use `moment(lam)` for a moment
utility: it integrates the full attainable score distribution, including
scores between reporting thresholds. `discovery()` counts at least one
positive-score trial, and `stability()` requires every trial to have score `1`.

## Intervals and posterior draws

```ts
import { tailpass } from "scorio/eval";

const profile = tailpass([[0, 1, 1], [1, 1, 1]], 8);
const draws = profile.sample(4000, { rng: 42 });
const [mean, std, lo, hi] = draws.summary(draws.qrs());
console.log(mean, std, lo, hi);

const [means, stds, lower, upper] = profile.ci(0.95, {
  method: "normal",
});
console.log([means.length, stds.length, lower.length, upper.length]); // => [8, 8, 8, 8]
```

`sample` shares posterior draws across thresholds and utilities. Draw objects
offer `linear`, `moment`, `powerMean`, `qrs`, `rollout`, `harmonic`, and
`shortfall`. Nonlinear utilities transform each question before averaging by
default; `powerMean` and `qrs` also accept `{ aggregation: "profile" }`.

`summary` returns a mean, sample standard deviation, and equal-tailed
interval. The profile methods `ci`, `linearCi`, and `momentCi` use Monte Carlo
by default; pass `{ method: "normal" }` for exact moments with clipped normal
intervals. These describe uncertainty in latent expected performance.

`atK(k)` reuses the posterior at a different budget. Seeded draws are repeatable
within JavaScript, but their random streams differ from Python and Julia.
For large categorical state spaces, exact calculations can reach the source's
enumeration limits; use posterior draws or an explicit threshold grid when the
error message calls for them.
