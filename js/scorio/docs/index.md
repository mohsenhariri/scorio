# Scorio for JavaScript and TypeScript

Scorio evaluates models from repeated outcomes, ranks them on shared questions,
and selects answers from sampled completions. The npm package includes Bayesian
uncertainty estimates and runs in Node.js or a browser, with no runtime
dependencies. Type declarations are included.

These pages describe the current TypeScript source in the
[Scorio repository](https://github.com/mohsenhariri/scorio/tree/main/js/scorio).
The published npm package may lag behind this checkout. See
[installation](installation.md) to use the repository version before the next
release.

## Evaluate repeated outcomes

Install the package with `npm install scorio`. In this example, each row is a
question and each column is a trial; `1` means correct and `0` means incorrect.

```ts
import { bayes, passAtK } from "scorio/eval";

const R = [
  [0, 1, 1, 0, 1],
  [1, 1, 0, 1, 1],
];

const [mean, std] = bayes(R);
console.log(mean); // => 0.6428571428571428
console.log(std);  // posterior standard deviation
console.log(passAtK(R, 2)); // => 0.95
```

`bayes` returns a posterior mean and standard deviation. `passAtK` estimates the
probability that at least one of `k` samples is correct. The
[evaluation guide](evaluation.md) explains their return values and interval
estimates.

## Choose an API

| Import | Use it to | Guide |
| --- | --- | --- |
| {@link eval | scorio/eval} | Estimate performance and uncertainty from repeated outcomes | [Evaluation](evaluation.md), [TailPass](tailpass.md) |
| {@link rank | scorio/rank} | Rank models evaluated on the same questions | [Model ranking](ranking.md) |
| {@link aggregate | scorio/aggregate} | Select an answer using votes, rewards, or token confidence | [Answer aggregation](aggregation.md) |
| {@link sinf | scorio/sinf} | Check stopping criteria and suggest which models to sample next | [Sequential inference](sequential-inference.md) |
| {@link utils | scorio/utils} | Convert scores to ranks, compare rankings, and encode rankings | [Ranking utilities](utilities.md) |

You can also import namespaces from the package root:

```ts
import { eval as metrics, rank, aggregate, sinf, utils } from "scorio";

const [mean] = metrics.bayes([0, 1, 1]);
const result = rank.avg([[1, 1], [0, 1]]);
const answer = aggregate.majorityVote(["A", "A", "B"]);
const stop = sinf.shouldStop(0.01, { maxHalfWidth: 0.05 });
const ranks = utils.rankScores([mean, 0.5]);

console.log(result.ranking, answer, stop, ranks.competition);
```

The root export `agg` is an alias for `aggregate`. Most camelCase functions also
have snake_case aliases for code shared with Python and Julia. The API reference
lists the exported names, signatures, defaults, and option types directly from
the source.

Read [data shapes](data.md) before loading your own results. Evaluation outcomes,
model responses, and candidate answers have different layouts.
