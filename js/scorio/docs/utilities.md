---
title: Ranking utilities
---

`scorio/utils` converts scores to ranks and compares existing rankings.
It also encodes permutations and rankings as collision-free integer hashes.
The {@link utils | API reference} lists the functions and option types.

## Convert and compare

```ts
import { rankScores, compareRankings } from "scorio/utils";

const ranks = rankScores([0.95, 0.8, 0.8, 0.5]);
console.log(ranks.competition); // => [1, 2, 2, 4]
console.log(ranks.dense);       // => [1, 2, 2, 3]

const comparison = compareRankings([1, 2, 3], [1, 3, 2]);
console.log(comparison.spearmanr[0]); // => 0.5
```

`rankScores` returns all four tie conventions described in the
[ranking guide](ranking.md). Supplying `sigmas` adds interval-aware tie results,
using `confidence` and `ciTieMethod`. The options-object form is convenient in
TypeScript; a positional form is available for Python-style calls.

`compareRankings` defaults to `"all"`, returning Kendall, Spearman, and weighted
Kendall statistics together with the fraction of mismatched ranks and maximum
displacement. To request one statistic, use `"kendall"`, `"spearman"`, or
`"weighted_kendall"` as the third argument. Each correlation result is a
`[statistic, pvalue]` tuple.

## Encode rankings

```ts
import { rankingHash, unhashRanking, lehmerHash, lehmerUnhash } from "scorio/utils";

const hash = rankingHash([1, 2, 2]);
console.log(unhashRanking(hash, 3)); // => [1, 2, 2]

const permutation = [2, 0, 1];
console.log(lehmerUnhash(lehmerHash(permutation), 3)); // => [2, 0, 1]
```

Ranking hashes preserve ordered tie blocks. Decoding returns competition ranks,
so the numeric rank convention can change while the ordering and ties remain
the same. Lehmer hashes encode permutations of `0` through `n - 1`.

Hashes return a `number` when it is safe and a `bigint` for larger values.
Keep the integer type intact when storing or decoding a hash. JSON has no
`bigint` value type, so store large hashes as decimal strings and restore them
with `BigInt` before decoding.

The module also exports `orderedBell`, `combRankLex`, `combUnrankLex`, and
`blocksFromRankList` for combinatorial calculations.
