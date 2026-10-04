---
title: Evaluation and uncertainty
---

Evaluation starts with repeated outcomes for one model. The functions average
over questions, so each question contributes equally. The
[data guide](data.md) describes binary and categorical inputs. The
{@link eval | API reference} lists the exported metrics and their signatures.

## Bayes@N and the observed average

`bayes(R, w?, R0?)` returns `[mean, std]`. It adds one prior count per category,
then adds the observed counts from `R` and any prior outcomes from `R0`.
`std` is the posterior standard deviation of the mean performance across
questions.

`avg(R, w?)` also returns two numbers. Its first value is the observed weighted
average; its second is a Bayesian uncertainty estimate rescaled to the average
scale. It does not accept `R0`.

```ts
import { avg, bayes } from "scorio/eval";

const R = [[0, 1, 1, 0, 1], [1, 1, 0, 1, 1]];
const [observed, observedStd] = avg(R);
const [posterior, posteriorStd] = bayes(R);

console.log(observed);  // => 0.7
console.log(posterior); // about 0.642857
console.log(observedStd, posteriorStd);
```

The difference between the two means is expected. Bayes@N includes the prior,
which pulls estimates away from the extremes when few trials are available.

## Sampling metrics

The scalar metrics below answer different questions about a bank of completions.
For the binary pass and majority families, `k` is an integer from `1` to the
observed trial count `N`.

| Function | Quantity |
| --- | --- |
| `passAtK(R, k)` | Probability at least one of `k` selected trials succeeds |
| `passHatK(R, k)`, `unanimousAtK(R, k)`, `gPassAtK(R, k)` | Probability all `k` selected trials succeed |
| `gPassAtKTau(R, k, tau)` | Probability of at least `max(1, ceil(tau × k))` successes |
| `mgPassAtK(R, k)` | Average of generalized pass thresholds over the upper half of the threshold range |
| `majAtK(R, k)` | Probability a strict majority of the selected trials succeeds |
| `aucAtK(R, k)` | Normalized trapezoidal area under Pass@1 through Pass@k |
| `maxAtK(R, k, w?)` | Expected maximum rubric score in `k` selected trials |

```ts
import { passAtK, passHatK, majAtK } from "scorio/eval";

const R = [[0, 1, 1, 0, 1], [1, 1, 0, 1, 1]];
console.log(passAtK(R, 2));  // => 0.95
console.log(passHatK(R, 2)); // => 0.45
console.log(majAtK(R, 3));   // => 0.85
```

`majAtK` counts correct outcomes. Actual voting over answer strings is handled
by `majorityVote` in [aggregation](aggregation.md).

The API also includes `geomAtK`, `geomDsAtK`, `geoSpectrumAtK`,
`geoSpectrumStarAtK`, and `thresholdSpectrumAtK` for geometric and threshold
blends. Their parameters and definitions are in the API reference.

## Credible intervals

The `*Ci` functions return `[mean, std, lo, hi]`. The usual confidence default
is `0.95`, and the interval uses a normal approximation. Pass-family intervals
clip to `[0, 1]` by default; `bayesCi` and `avgCi` accept optional bounds and
do not clip unless you pass them.

```ts
import { bayesCi, passAtK, passAtKCi } from "scorio/eval";

const R = [[0, 1, 1, 0, 1], [1, 1, 0, 1, 1]];
const [mean, std, lo, hi] = bayesCi(R, undefined, undefined, 0.95, [0, 1]);
console.log(mean, std, lo, hi);

console.log(passAtK(R, 2));       // empirical estimate: 0.95
console.log(passAtKCi(R, 2)[0]); // posterior mean: about 0.839286
```

The interval companion's mean can differ from the scalar point estimate.
For `passAtK`, the point estimator samples the observed bank without
replacement. `passAtKCi` instead summarizes the i.i.d. success quantity under a
Beta posterior, with success and failure prior concentrations both `1` by
default. Report which estimate you use.

TailPass has its own profile and interval API. Read the
[TailPass guide](tailpass.md) when you need several success thresholds or want
to choose a utility explicitly.
