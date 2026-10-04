# scorio

[JavaScript / TypeScript documentation](https://mohsenhariri.github.io/scorio/javascript/)

Bayesian evaluation toolkit for stochastic models — a TypeScript/JavaScript port of [Scorio](https://github.com/mohsenhariri/scorio).

It provides these main API families:

- **`scorio/eval`** — point estimates **and** Bayesian uncertainty for metrics used to evaluate LLMs and other stochastic models under repeated sampling: Bayes@N, Avg@N, Pass@k / Pass^k, G-Pass@k, Maj@k, AUC@K, Max@k, and the geometric/spectrum blends.
- **`scorio/rank`** — 40+ ranking estimators that order multiple models from a binary (or categorical) response tensor: eval-metric, voting, pairwise-rating (Elo/Glicko/TrueSkill), Bradley-Terry / Plackett-Luce / Rao-Kupper, IRT (Rasch/2PL/3PL/MML and multidimensional MIRT), graph (PageRank/spectral/α-Rank/Nash), seriation, and Hodge-theoretic methods.
- **`scorio/aggregate`** — test-time-scaling confidence signals, PRM reward reduction, Best-of-N and voting rules, and online early stopping over candidate answer pools.
- **`scorio/utils`** — score-to-rank conversion, ranking correlation statistics, and collision-free hashes for permutations and rankings with ties.
- `scorio/sinf` provides stopping and allocation decisions from posterior means and standard deviations.

- **Zero runtime dependencies** — pure TypeScript (special functions, linear algebra, optimization, and an LP solver reimplemented from `scipy`/`numpy`).
- **Dual ESM + CommonJS** builds with full type declarations.
- **Numerically faithful** to the Python reference (verified against generated ground-truth fixtures).
- **Two naming styles**: idiomatic camelCase (`passAtK`, `bradleyTerry`) and snake_case aliases matching the Python/Julia API (`pass_at_k`, `bradley_terry`).

## Install

```sh
npm install scorio
```

## Usage

The outcome matrix `R` has shape `M × N` (M questions, N trials per question) with integer category entries in `{0,…,C}`. Binary metrics use entries in `{0,1}`. A 1-D array is treated as a single row.

```ts
import { eval as scorio } from "scorio";
// or: import { bayes, passAtK } from "scorio/eval";

// Multi-category outcomes with a rubric weight vector (length C+1)
const R = [
  [0, 1, 2, 2, 1],
  [1, 1, 0, 2, 2],
];
const w = [0.0, 0.5, 1.0]; // 0=incorrect, 1=partial, 2=correct
const R0 = [               // optional prior outcomes (M × D)
  [0, 2],
  [1, 2],
];

const [mu, sigma] = scorio.bayes(R, w, R0);
// mu ≈ 0.575, sigma ≈ 0.084275

const [a, sa] = scorio.avg(R, w);
// weighted average with Bayesian uncertainty

// Binary metrics
const B = [
  [0, 1, 1, 0, 1],
  [1, 1, 0, 1, 1],
];
scorio.passAtK(B, 2);   // 0.95
scorio.passHatK(B, 2);  // 0.45  (a.k.a. unanimousAtK / g_pass@k)
scorio.passAtKCi(B, 2); // [mu, sigma, lo, hi]
```

### Point estimators vs. credible intervals

`bayes` and `avg` return `[mu, sigma]`; the other metrics in the table below return a scalar. Their `*Ci` functions (and `*_ci` aliases) return `[mu, sigma, lo, hi]` with a normal-approximation credible interval. For pass-family metrics, the interval companion uses a posterior mean that can differ from the empirical point estimate. TailPass returns a profile with a separate utility and interval API.

## API

| Family | Point estimator | Credible interval |
| --- | --- | --- |
| Bayes@N | `bayes` | `bayesCi` |
| Avg@N | `avg` | `avgCi` |
| Pass@k | `passAtK` | `passAtKCi` |
| Pass^k / unanimous | `passHatK`, `unanimousAtK` | `passHatKCi`, `unanimousAtKCi` |
| G-Pass@k | `gPassAtK`, `gPassAtKTau`, `mgPassAtK` | `gPassAtKCi`, `gPassAtKTauCi`, `mgPassAtKCi` |
| Majority | `majAtK` | `majAtKCi` |
| AUC@K | `aucAtK` | `aucAtKCi` |
| Max@k | `maxAtK` | `maxAtKCi` |
| Geometric / spectrum | `geomAtK`, `geomDsAtK`, `geoSpectrumAtK`, `geoSpectrumStarAtK`, `thresholdSpectrumAtK` | each with a `*Ci` variant |

Each camelCase name has a snake_case alias (`pass_at_k`, `g_pass_at_k_tau`, `geo_spectrum_at_k`, …) for parity with the Python and Julia packages.

## Aggregation (`scorio/aggregate`)

Aggregation methods consume one candidate row `(N,)` or a batch `(M, N)`. Invalid answers (`null`, `undefined`, `""`, and `NaN`) are ignored. Score-aware selectors can also return the representative candidate's index and raw score.

```ts
import { aggregate as agg } from "scorio";
// or: import { bestOfN, majorityVote } from "scorio/aggregate";

const answers = ["A", "A", "B", "C"];
const scores = [0.3, 0.4, 0.9, 0.2];

agg.majorityVote(answers); // "A"
agg.bestOfN(answers, scores); // "B"
agg.bestOfN(answers, scores, { returnIndex: true, returnScore: true });
// ["B", 2, 0.9]
```

| Family | Methods |
| --- | --- |
| Confidence | `meanLogprob`, `sequenceLogprob`, `perplexity`, `picsar`, `selfCertainty`, `tokenConfidence`, `deepconfConfidence`, `tokenEntropy`, `varentropy`, `maxSoftmaxProbability`, `logprobMargin` |
| PRM reduction | `prmAggregate` (`last`, `min`, `mean`, `prod`, `max`) |
| Reward selection | `bestOfN`, `majorityOfTheBests` / `mob`, `bestOfMajority` |
| Voting | `majorityVote`, `weightedMajorityVote`, `softmaxWeightedVote`, `rankWeightedVote`, `logitWeightedVote`, `filteredVote` |
| Calibrated voting | `KDEVoteCalibration`, `fitKdeVoteCalibration`, `kdeWeightedVote` |
| Confidence-guided aggregation | `CGES_OTHER`, `cgesVote`, `cgesStop` |
| Online stopping | `adaptiveConsistencyStop`, `adaptiveConsistencyDirichletStop`, `adaptiveConsistencyCrpStop`, `escStop`, `deepconfStopThreshold`, `deepconfOnlineStop` |

Every camelCase function also has a snake_case alias matching Python/Julia. Python distinguishes `filtered_vote(..., keep=1)` (one candidate) from `keep=1.0` (all candidates), but JavaScript has only one numeric `1`; numeric `1` therefore means a count, while `{ keep: { fraction: 1 } }` explicitly means the full fraction.

`adaptiveConsistencyCrpStop` implements the same finite-horizon CRP model,
defaults, tie rule, and seeded reproducibility contract as Python. Its deterministic
JavaScript random generator is not NumPy's PCG64, however, so a fixed seed produces
the same statistical procedure but not a bit-for-bit identical Monte Carlo stream or
probability estimate.

## Ranking (`scorio/rank`)

Ranking estimators take a response tensor `R` of shape `(L, M, N)` — `L` models, `M` questions, `N` trials — with binary entries (a 2-D `(L, M)` matrix is treated as `N = 1`). Each method returns `{ ranking, scores }`: `ranking[l]` is model `l`'s rank (1 = best) and `scores[l]` the raw method score (larger is better). The optional `method` selects the tie convention (`"competition"` by default; also `"competition_max"`, `"dense"`, `"avg"`).

```ts
import { rank } from "scorio";
// or: import { borda, bradleyTerry } from "scorio/rank";

// 2 models, 2 questions, 2 trials
const R = [
  [[1, 1], [1, 1]],
  [[0, 0], [0, 0]],
];

rank.borda(R).ranking;          // [1, 2]
rank.elo(R).scores;             // final Elo ratings
rank.bradleyTerryMap(R, { prior: 1, maxIter: 500 }).ranking;
rank.bayes(R, { quantile: 0.05 });   // conservative, uncertainty-aware
rank.raschMap(R, { prior: 1.0 });    // MAP IRT with a Gaussian prior

// snake_case aliases mirror the Python API
rank.pass_at_k(R, 2);
rank.rank_centrality(R);
```

| Family | Methods |
| --- | --- |
| Eval-metric | `avg`, `bayes`, `passAtK`, `passHatK`, `gPassAtKTau`, `mgPassAtK` |
| Pointwise | `inverseDifficulty` |
| Pairwise ratings | `elo`, `glicko`, `trueskill` |
| Bradley-Terry | `bradleyTerry`(`Map`), `bradleyTerryDavidson`(`Map`), `raoKupper`(`Map`) |
| Bayesian | `thompson`, `bayesianMcmc` |
| Voting | `borda`, `copeland`, `winRate`, `minimax`, `schulze`, `rankedPairs`, `kemenyYoung`, `nanson`, `baldwin`, `majorityJudgment` |
| IRT | `rasch`(`Map`), `rasch2pl`(`Map`), `rasch3pl`(`Map`), `raschMml`, `raschMmlCredible`, `dynamicIrt`, `mirt` (multidimensional) |
| Graph | `pagerank`, `spectral`, `alpharank`, `nash`, `rankCentrality` |
| Seriation / Hodge | `serialRank`, `hodgeRank` |
| Plackett-Luce | `plackettLuce`(`Map`), `davidsonLuce`(`Map`), `bradleyTerryLuce`(`Map`) |
| Priors (for MAP) | abstract runtime `Prior`, `GaussianPrior`, `LaplacePrior`, `CauchyPrior`, `UniformPrior`, `CustomPrior`, `EmpiricalPrior` |

The MAP estimators accept a `prior` option — either a variance (interpreted as a zero-mean `GaussianPrior`) or a `Prior` instance. The Monte-Carlo methods (`thompson`, `bayesianMcmc`) are seeded and reproducible but, since they use a different RNG, are not bit-identical to the Python reference.

## Ranking utilities (`scorio/utils`)

```ts
import { compareRankings, rankScores, rankingHash, unhashRanking } from "scorio/utils";

const ranks = rankScores([0.95, 0.8, 0.8, 0.5]);
// ranks.competition = [1, 2, 2, 4]

compareRankings([1, 2, 3], [1, 3, 2]);
rankingHash([1, 2, 2]);
unhashRanking(2, 3); // [1, 2, 2]
```

The module provides camelCase names and exact Python-style aliases:
`rank_scores`, `compare_rankings`, `lehmer_hash`, `lehmer_unhash`,
`ranking_hash`, and `unhash_ranking`. Public combinatorial helpers from the
Python module are included as well. Hashes that exceed JavaScript's safe
integer range are returned as `bigint`, preserving the Python implementation's
collision-free behavior.

## Development

```sh
npm install
npm test          # vitest golden tests (parity with the Python reference)
npm run build     # tsup -> dist/ (ESM + CJS + d.ts)
npm run typecheck
npm run docs      # TypeDoc site from the current source
npm run docs:check # check guide examples and generated links
```

## License

MIT © Mohsen Hariri. See the repository root `LICENSE` and `CITATION.cff`.

## TailPass profiles and utilities

TailPass returns a posterior threshold profile. Choose a scalar utility explicitly:

```ts
import { tailpass, tailpassWeights } from "scorio/eval";

const profile = tailpass([[0, 1, 1], [1, 1, 1]], 8);
const [mu, sigma] = profile.linear(tailpassWeights.momentWeights(8, 2));
const draws = profile.sample(4000, { rng: 42 });
const [qrsMean, qrsStd, lo, hi] = draws.summary(draws.qrs());
```

`tailpass(R, k, w?, R0?, { eta, prior, thresholds })` uses the posterior
`prior + counts(R) + eta * counts(R0)`. Rubric scores must lie in [0, 1].
`prior` is a positive scalar, category vector, or question-by-category matrix;
binary order is failure, success. `k` may exceed the observed trial count.

Profiles expose `mean`, `questionMean`, `covariance`, and `std`, and provide
`linear(weights)`, `moment(lam)`, `discovery()`, `stability()`, and `atK(k)`.
For categorical rubrics, `moment(lam)` integrates the full attainable score
distribution, including scores between reporting thresholds. The empirical
counterpart `tailpassEmpirical(R, k, w?, { thresholds })` samples observed trials
without replacement and requires `k <= N`.

Draws provide `linear`, `moment`, `powerMean`, `qrs`, `rollout`, `harmonic`,
`shortfall`, and `atK`. Nonlinear utilities transform each question before
averaging by default. `powerMean` and `qrs` also accept `{ aggregation: "profile" }`.
`summary(values?, confidence?)` returns mean, sample standard deviation, and
pointwise equal-tailed intervals. `ci`, `linearCi`, and `momentCi` on a profile
accept `{ method: "normal" }` for exact moments and clipped Gaussian intervals,
or `{ method: "mc", nDraws, rng }` (the default) for shared posterior draws.
These intervals describe latent expected performance, not future sampled banks.
Seeds are reproducible within JavaScript; random streams differ across languages.

Weight constructors in `tailpassWeights` include `uniformWeights`,
`thresholdWeights`, `discoveryWeights`, `stabilityWeights`, `momentWeights`,
`betaWeights`, `maxentWeights`, and `payoffWeights`. Snake_case aliases and the
`tailpass_weights` namespace are also available.

Exact count enumeration is limited to 20,000 states, and profile covariance to
4,000,000 count/threshold pairs. For large categorical budgets, moments 1, 2,
and 4 and the discovery/stability utilities avoid count enumeration. Supply an
explicit threshold grid when `k > 20,000`. Posterior draws provide a covariance
alternative when the mean profile is feasible but exact covariance is too large.

## Sequential inference

The `scorio/sinf` entry point exposes `rankingConfidence`, `ciFromMuSigma`,
`shouldStop`, `shouldStopTop1`, and `suggestNextAllocation`, with snake_case
aliases. These operate on posterior means and standard deviations, as in Python.
