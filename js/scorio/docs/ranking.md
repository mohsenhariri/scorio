---
title: Model ranking
---

Ranking functions compare models on shared questions. Pass a response tensor
with shape `L × M × N`, or a matrix `L × M` for one trial per question. Most
methods require binary outcomes. `rank.bayes` also accepts categorical outcomes
with a rubric. See [data shapes](data.md) for the layouts and the
{@link rank | API reference} for method signatures and options.

## Rank by an evaluation metric

```ts
import { avg, bayes } from "scorio/rank";

const R = [
  [[1, 1, 0], [1, 0, 1]],
  [[0, 1, 0], [0, 0, 1]],
];

console.log(avg(R).ranking); // => [1, 2]
console.log(bayes(R, { quantile: 0.05 }).ranking); // => [1, 2]
```

Every estimator returns at least `{ ranking, scores }`. The entry
`ranking[l]` is the rank assigned to model `l`, with `1` best. The entry
`scores[l]` is the underlying method score, with larger values ranked higher.
Some methods add uncertainty, item parameters, or diagnostics.

For `bayes`, `quantile` selects the normal-approximation posterior quantile
used as the ranking score. A value of `0.05` ranks by a lower posterior
quantile; omitting it ranks by the posterior mean. Categorical weights and
prior outcomes are passed through the `w` and `R0` options.

## Ties

The `method` option selects how equal scores become ranks:

| Method | Ranks for scores `[0.9, 0.8, 0.8, 0.5]` |
| --- | --- |
| `"competition"` (default) | `[1, 2, 2, 4]` |
| `"competition_max"` | `[1, 3, 3, 4]` |
| `"dense"` | `[1, 2, 2, 3]` |
| `"avg"` | `[1, 2.5, 2.5, 4]` |

```ts
import { avg } from "scorio/rank";

const R = [[1, 1], [1, 0], [0, 1], [0, 0]];
console.log(avg(R, { method: "dense" }).ranking); // => [1, 2, 2, 3]
```

## Other ranking models

The exported methods include the following families. Each consumes responses
in the same model/question/trial layout and constructs the comparisons it
needs.

| Family | Functions |
| --- | --- |
| Evaluation metrics | `avg`, `bayes`, `passAtK`, `passHatK`, `gPassAtKTau`, `mgPassAtK` |
| Difficulty weighting | `inverseDifficulty` |
| Pairwise ratings | `elo`, `glicko`, `trueskill` |
| Bradley–Terry and tie models | `bradleyTerry`, `bradleyTerryDavidson`, `raoKupper`, and their `Map` variants |
| Posterior sampling | `thompson`, `bayesianMcmc` |
| Voting | `borda`, `copeland`, `winRate`, `minimax`, `schulze`, `rankedPairs`, `kemenyYoung`, `nanson`, `baldwin`, `majorityJudgment` |
| Item response models | `rasch`, `rasch2pl`, `rasch3pl`, their `Map` variants, `raschMml`, `raschMmlCredible`, `dynamicIrt`, `mirt` |
| Graph methods | `pagerank`, `spectral`, `alpharank`, `nash`, `rankCentrality` |
| Seriation and decomposition | `serialRank`, `hodgeRank` |
| Listwise choice models | `plackettLuce`, `davidsonLuce`, `bradleyTerryLuce`, and their `Map` variants |

## Priors and fit failures

Maximum-likelihood methods can throw when no finite estimate exists. For
example, Bradley–Terry checks whether the directed win graph is strongly
connected. A model that always beats another can violate that condition.
Use a MAP estimator with a stated prior when a regularized fit suits the
analysis:

```ts
import { bradleyTerryMap, GaussianPrior } from "scorio/rank";

const R = [
  [[1, 1], [1, 1]],
  [[0, 0], [0, 0]],
];
const result = bradleyTerryMap(R, {
  prior: new GaussianPrior(0, 1),
  maxIter: 500,
});
console.log(result.ranking); // => [1, 2]
```

A numeric `prior` is interpreted as the variance of a zero-mean Gaussian.
The API also exports `LaplacePrior`, `CauchyPrior`, `UniformPrior`,
`CustomPrior`, and `EmpiricalPrior`. A uniform prior does not remove
maximum-likelihood existence problems.

Optimization failures and unproven exact solutions are reported as errors.
`kemenyYoung`, for example, enumerates permutations and throws if a supplied
time limit expires before it proves an optimum. Increasing an iteration or
time limit only helps when the underlying fit exists and computation is the
limiting factor.

Monte Carlo methods accept seeds. Repeated calls with the same data and seed
are reproducible within JavaScript; Python's and Julia's streams differ.
For comparing the resulting rankings, see [ranking utilities](utilities.md).
