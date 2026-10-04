---
title: Sequential inference
---

The `scorio/sinf` helpers take posterior means and standard deviations and
return stopping or allocation decisions. Your code runs the trials and updates
the estimates. The helpers use normal approximations, so their decisions
depend on that approximation and the uncertainty estimates you supply. See the
{@link sinf | API reference} for the available functions.

## Stop when an interval is narrow enough

```ts
import { bayes } from "scorio/eval";
import { ciFromMuSigma, shouldStop } from "scorio/sinf";

const [mean, std] = bayes([[0, 1, 1, 0, 1], [1, 1, 0, 1, 1]]);
console.log(ciFromMuSigma(mean, std, { confidence: 0.95, clip: [0, 1] }));
console.log(shouldStop(std, { confidence: 0.95, maxHalfWidth: 0.05 }));
```

Pass exactly one of `maxHalfWidth` or `maxCiWidth` to `shouldStop`. The latter
is the full interval width. Both use the requested confidence level, which
defaults to `0.95`.

## Check the leading model

```ts
import { shouldStopTop1, suggestNextAllocation } from "scorio/sinf";

const means = [0.81, 0.79, 0.55];
const stds = [0.03, 0.03, 0.02];
const decision = shouldStopTop1(means, stds, { confidence: 0.95 });
console.log(decision); // => {"stop": false, "leader": 0, "ambiguous": [1]}

console.log(suggestNextAllocation(means, stds)); // => {"leader": 0, "competitor": 1}
```

`leader` and the entries in `ambiguous` are zero-based model indices.
The default `"ci_overlap"` method stops when the leader's lower interval
endpoint exceeds every competitor's upper endpoint. With
`{ method: "zscore" }`, it instead compares pairwise ordering probabilities
to the confidence threshold.

`suggestNextAllocation` returns the leader and the competitor with the smallest
separation under the same method. It requires at least two models and does
not assign a trial budget or start sampling.

`rankingConfidence` returns `{ rho, z }` for two means and standard
deviations. The calculation combines their variances as independent
estimates; it does not take a cross-model covariance.
