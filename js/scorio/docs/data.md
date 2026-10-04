---
title: Data shapes
---

## Outcomes for one model

Evaluation functions take an outcome matrix `R` with shape `M × N`: one row per
question and one column per trial. All rows must have the same length. Entries
are finite integer category labels; binary metrics use `0` and `1`.

```ts
import { bayes } from "scorio/eval";

// Two questions, three trials each.
const R = [
  [0, 1, 1],
  [1, 0, 1],
];
console.log(bayes(R));

// A flat array is one question with three trials.
console.log(bayes([0, 1, 1]));
```

For a categorical rubric, pass `w`, a vector that maps category `c` to score
`w[c]`. Its length defines the allowed categories `0` through `C`.

```ts
import { bayes } from "scorio/eval";

const R = [[0, 1, 2, 2, 1], [1, 1, 0, 2, 2]];
const w = [0, 0.5, 1]; // incorrect, partial, correct
const R0 = [[0, 2], [1, 2]];

console.log(bayes(R, w, R0));
```

`R0` holds prior outcomes for the same `M` questions. It can have a different
trial count `D`, but its rows must be rectangular and its category labels must
use the same rubric as `R`. Passing a nested `M × D` matrix makes the question
alignment explicit.

`bayes` and `avg` accept finite rubric scores beyond `[0, 1]`. TailPass requires
scores in `[0, 1]`. Check a metric's signature before applying a categorical
rubric; many metrics accept binary outcomes only.

## Responses for several models

Ranking methods take `R` with shape `L × M × N`: model, question, trial. Keep
the same question order for every model.

```ts
import { avg } from "scorio/rank";

const R = [
  [[1, 1, 0], [1, 0, 1]], // model 0
  [[0, 1, 0], [0, 0, 1]], // model 1
];
console.log(avg(R).ranking); // => [1, 2]
```

A two-dimensional `L × M` matrix means one trial per question. Ranking
functions construct comparisons from these outcomes. You do not pass a
pairwise win matrix to `bradleyTerry`, `borda`, or the other ranking estimators.

## Answers for selection

Aggregation uses answer labels rather than correctness categories. Pass one
array of `N` candidates, or an `M × N` batch. When scores are needed, their
shape must match the answers.

```ts
import { bestOfN } from "scorio/aggregate";

const answers = [["A", "B", "A"], ["C", "D", "D"]];
const scores = [[0.3, 0.9, 0.4], [0.8, 0.2, 0.6]];
console.log(bestOfN(answers, scores)); // => ["B", "C"]
```

Normalize equivalent answers to the same label before voting. For example,
`"1/2"` and `"0.5"` are distinct strings to the selector. `null`, `undefined`,
empty strings, and numeric `NaN` are ignored as invalid answers; a row with no
valid answer selects `null`. See [answer aggregation](aggregation.md) for return
shapes and tie handling.
