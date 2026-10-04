---
title: Answer aggregation
---

Aggregation selects an answer from sampled candidates. It does not need their
ground-truth correctness labels. Normalize equivalent answers before passing
them to a selector; the selectors group labels by JavaScript equality. The
{@link aggregate | API reference} includes each selector's options and return type.

## Votes and rewards

```ts
import { majorityVote, bestOfN } from "scorio/aggregate";

const answers = ["A", "A", "B", "C"];
const scores = [0.3, 0.4, 0.9, 0.2];

console.log(majorityVote(answers)); // => "A"
console.log(bestOfN(answers, scores)); // => "B"
console.log(bestOfN(answers, scores, {
  returnIndex: true,
  returnScore: true,
})); // => ["B", 2, 0.9]
```

`majorityVote` selects the most frequent label. `bestOfN` selects the valid
candidate with the largest score. Ties in these two rules use the earliest
candidate. Missing answers (`null`, `undefined`, `""`, and `NaN`) are ignored.

One candidate row returns a scalar answer. A batch returns one answer per
question. With `returnIndex` and `returnScore`, the tuple order is answer,
index, score; for a batch, each tuple component is an array. A row with no
valid answer returns `null`, with index `-1` and score `NaN` when requested.

## Score-aware voting

`weightedMajorityVote` combines scores within each answer group.
`softmaxWeightedVote`, `rankWeightedVote`, and `logitWeightedVote` convert
candidate scores to weights before grouping them. `filteredVote` retains a
chosen count or fraction of the highest-scoring candidates, then votes.

```ts
import { filteredVote } from "scorio/aggregate";

const answers = ["A", "A", "B", "C"];
const scores = [0.3, 0.4, 0.9, 0.2];
console.log(filteredVote(answers, scores, { keep: 1 })); // => "B"
console.log(filteredVote(answers, scores, {
  keep: { fraction: 1 },
  weighted: false,
})); // => "A"
```

JavaScript treats `1` and `1.0` as the same number. Numeric `keep: 1` therefore
means one candidate. Use `{ fraction: 1 }` to retain the whole pool. Explicit
`{ count: n }` and `{ fraction: f }` settings avoid that ambiguity.
Voting is score-weighted by default; `weighted: false` counts each retained
candidate once.

`majorityOfTheBests` (also `mob`) computes the mode of the bootstrapped
Best-of-N distribution. `bestOfMajority` scores answer groups.
For calibrated weights, the API provides `KDEVoteCalibration`,
`fitKdeVoteCalibration`, and `kdeWeightedVote`.

## Confidence from traces

Confidence functions turn token scores into a scalar or a per-token signal.
`meanLogprob` and `sequenceLogprob` use the chosen tokens' log probabilities.
Other functions use the supplied top-k log-probability rows, which can be
ragged across token positions.

```ts
import { meanLogprob, prmAggregate, bestOfN } from "scorio/aggregate";

const answers = ["A", "B"];
const tokenLogprobs = [[-0.1, -0.2, -0.1], [-0.8, -0.4, -0.6]];
const scores = tokenLogprobs.map(meanLogprob);
console.log(bestOfN(answers, scores)); // => "A"

console.log(prmAggregate([0.9, 0.8, 0.7], { method: "min" })); // => 0.7
```

The confidence API includes `perplexity`, `selfCertainty`, `tokenEntropy`,
`varentropy`, `maxSoftmaxProbability`, `logprobMargin`, `picsar`,
`tokenConfidence`, and `deepconfConfidence`. Check the score direction:
perplexity and entropy are lower when confidence is higher, while Best-of-N
selects the largest score. Transform those signals before using them as a
reward.

`prmAggregate` reduces step rewards with `last`, `min`, `mean`, `prod`, or
`max`. Select the reduction that matches how your process reward model was
trained.

## Stop sampling candidates

```ts
import { adaptiveConsistencyStop } from "scorio/aggregate";

const answers = ["A", "A", "A", "A", "A", "B"];
const [stop, probability] = adaptiveConsistencyStop(answers, {
  threshold: 0.95,
  returnProb: true,
});
console.log(stop, probability);
```

`adaptiveConsistencyStop` compares the two largest answer counts under a
Bayesian model. Dirichlet and finite-horizon CRP variants are available as
`adaptiveConsistencyDirichletStop` and `adaptiveConsistencyCrpStop`.
`escStop` checks a fixed answer window, while the DeepConf helpers use token
confidence. `cgesVote` and `cgesStop` provide confidence-guided selection and
stopping.

These functions report a decision; your application controls generation.
For stopping an evaluation campaign based on performance uncertainty, use
[sequential inference](sequential-inference.md).
