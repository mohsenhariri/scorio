/**
 * Max@k family — the continuous-reward generalization of Pass@k. Instead of
 * asking whether at least one sampled response is correct, Max@k scores the
 * best response among `k` sampled traces under a user-specified reward scale.
 * Port of `scorio/eval/max_reward.py`.
 *
 * The point estimator matches the appendix evaluation formula in Bagirov et
 * al. (2025), "The Best of N Worlds: Aligning Reinforcement Learning with
 * Best-of-N Sampling via max@k Optimization" (Appendix C.1 / Listing 1,
 * arXiv:2510.23393). The companion `*_ci` function is a `scorio` Bayesian
 * extension using the same grouped-Dirichlet posterior as `bayes`.
 */

import {
  logBetaPower,
  positiveLogDifference,
  scaledRewards,
  variance,
  integer,
  realVector,
  dot,
  sum,
} from "./internal/tailpass.js";
import { hypergeomAtLeastOne } from "./internal/math.js";
import { normalCredibleInterval, type Bounds } from "./internal/ci.js";
import {
  asMatrix,
  asPriorMatrix,
  validateMatrixRange,
  type Matrix,
} from "./internal/validate.js";
import { bayesCi } from "./bayes.js";

/** Per-row counts of the values `0..length-1` over a possibly-empty matrix. */
function rowBincountWide(
  A: readonly (readonly number[])[],
  length: number,
): number[][] {
  return A.map((row) => {
    const counts = new Array<number>(length).fill(0);
    for (const v of row) counts[v]! += 1;
    return counts;
  });
}

/** Normalized `R`, weight vector `w`, and prior matrix `R0`. */
interface CategoricalInput {
  Rm: number[][];
  wv: number[];
  R0m: number[][];
}

/** Normalize R, w, and R0 for weighted categorical metrics. */
function prepareCategoricalInput(
  R: Matrix,
  w?: readonly number[] | null,
  R0?: Matrix | null,
): CategoricalInput {
  const Rm = asMatrix(R);

  let wv: number[];
  if (w == null) {
    const seen = new Set<number>();
    for (const row of Rm) for (const v of row) seen.add(v);
    const isBinary =
      seen.size <= 2 && [...seen].every((v) => v === 0 || v === 1);
    if (!isBinary) {
      const vals = [...seen].sort((a, b) => a - b).join(", ");
      throw new Error(
        `R contains more than 2 unique values (${vals}), so weight vector 'w' must be provided. ` +
          `Please specify a weight vector of length ${seen.size} to map each category to a score.`,
      );
    }
    wv = [0.0, 1.0];
  } else {
    wv = realVector(w, "w");
    if (!wv.length) throw new Error("w must be nonempty");
  }

  const M = Rm.length;
  const C = wv.length - 1;
  validateMatrixRange(Rm, 0, C, "R");

  let R0m: number[][];
  if (R0 == null) {
    R0m = Rm.map(() => []);
  } else {
    R0m = asPriorMatrix(R0, M);
    if (R0m.length !== M) {
      throw new Error("R0 must have the same number of rows (M) as R.");
    }
    validateMatrixRange(R0m, 0, C, "R0");
  }

  return { Rm, wv, R0m };
}

function validateK(N: number, k: number): void {
  if (!(k >= 1 && k <= N) || !Number.isInteger(k)) {
    throw new Error(`k must satisfy 1 <= k <= N (N=${N}); got k=${k}`);
  }
}

/** Sorted unique values of `w`, with each entry's index into that sorted list. */
function uniqueLevels(wv: readonly number[]): {
  levels: number[];
  inverse: number[];
} {
  const sorted = [...new Set(wv)].sort((a, b) => a - b);
  const index = new Map<number, number>();
  sorted.forEach((v, i) => index.set(v, i));
  return { levels: sorted, inverse: wv.map((v) => index.get(v)!) };
}

/** Grouped Dirichlet posterior parameters and the unique reward levels. */
function groupedPosteriorParams(
  R: Matrix,
  w?: readonly number[] | null,
  R0?: Matrix | null,
): { gamma: number[][]; levels: number[] } {
  const { Rm, wv, R0m } = prepareCategoricalInput(R, w, R0);
  const C = wv.length - 1;

  const { levels, inverse } = uniqueLevels(wv);
  const L = levels.length;

  const nCounts = rowBincountWide(Rm, C + 1);
  const n0Counts = rowBincountWide(R0m, C + 1).map((row) =>
    row.map((c) => c + 1),
  );

  const gamma = Rm.map((_, row) => {
    const g = new Array<number>(L).fill(0);
    for (let cat = 0; cat <= C; cat++) {
      g[inverse[cat]!]! += nCounts[row]![cat]! + n0Counts[row]![cat]!;
    }
    return g;
  });

  return { gamma, levels };
}

/**
 * Max@k: expected best reward among `k` sampled traces.
 *
 * When `w = [0, 1]`, Max@k reduces exactly to Pass@k. More generally, the
 * reward vector `w` maps categorical outcomes to arbitrary real-valued scores,
 * and Max@k averages the best score obtainable from a subset of size `k`.
 *
 * The finite-sample estimator matches Bagirov et al. (2025), Appendix C.1 /
 * Listing 1 (arXiv:2510.23393).
 *
 * @param R `M x N` categorical outcome matrix with integer entries in
 *          `{0,...,C}`.
 * @param k Number of selected samples, with `1 <= k <= N`.
 * @param w Optional reward vector of shape `(C+1,)`. If omitted, `R` must be
 *          binary and `[0, 1]` is used.
 * @returns Average Max@k score across prompts.
 */
export function maxAtK(
  R: Matrix,
  k: number,
  w?: readonly number[] | null,
): number {
  const { Rm, wv } = prepareCategoricalInput(R, w),
    N = Rm[0]!.length;
  validateK(N, k);
  const { levels } = uniqueLevels(wv),
    [offset, scale, normalized] = scaledRewards(levels);
  if (scale === 0) return offset;
  const gaps = normalized.slice(1).map((v, j) => v - normalized[j]!);
  const means = Rm.map(
    (row) =>
      normalized[0]! +
      sum(
        gaps.map(
          (gap, j) =>
            gap *
            hypergeomAtLeastOne(
              N,
              row.filter((c) => wv[c]! > levels[j]!).length,
              k,
            ),
        ),
      ),
  );
  return offset + scale * (sum(means) / Rm.length);
}

/** Posterior mean/std for Max@k under a grouped Dirichlet posterior. */
function maxAtKBayes(
  R: Matrix,
  k: number,
  w?: readonly number[] | null,
  R0?: Matrix | null,
): { mu: number; sigma: number; levels: number[] } {
  integer(k);
  const { gamma, levels } = groupedPosteriorParams(R, w, R0);
  const [offset, scale, normalized] = scaledRewards(levels);
  if (scale === 0) return { mu: offset, sigma: 0, levels };
  const gaps = normalized.slice(1).map((v, j) => v - normalized[j]!);
  const moments = gamma.map((row) => {
    const total = sum(row),
      cum: number[] = [];
    let running = 0;
    for (const a of row.slice(0, -1)) {
      running += a;
      cum.push(running);
    }
    const logs = cum.map((a) => logBetaPower(a, total - a, k));
    const mean =
      normalized[0]! +
      dot(
        gaps,
        logs.map((v) => -Math.expm1(v)),
      );
    let v = 0;
    for (let i = 0; i < gaps.length; i++)
      for (let j = 0; j < gaps.length; j++) {
        const lower = Math.min(i, j),
          upper = Math.max(i, j);
        const cross =
          logBetaPower(cum[upper]!, total - cum[upper]!, 2 * k) +
          (i === j
            ? 0
            : logBetaPower(cum[lower]!, cum[upper]! - cum[lower]!, k));
        v +=
          gaps[i]! *
          gaps[j]! *
          positiveLogDifference(cross, logs[i]! + logs[j]!);
      }
    return [mean, Math.sqrt(variance(v))];
  });
  return {
    mu: offset + (scale * sum(moments.map((v) => v[0]!))) / gamma.length,
    sigma: (scale * Math.hypot(...moments.map((v) => v[1]!))) / gamma.length,
    levels,
  };
}

/**
 * Bayesian posterior summary for {@link maxAtK}, returning `[mu, sigma, lo, hi]`.
 *
 * The posterior uses the same Dirichlet-plus-one construction as `bayes`. When
 * `k = 1`, Max@1 reduces to the single-draw expected score, so this function
 * agrees with `bayesCi`. This uncertainty model is a `scorio` extension and is
 * not part of Bagirov et al. (2025).
 *
 * @param R `M x N` categorical outcome matrix with integer entries in
 *          `{0,...,C}`.
 * @param k Selection count; defined for any integer `k >= 1`. `k = 1` matches
 *          `bayesCi`.
 * @param w Optional reward vector of shape `(C+1,)`. If omitted, `R` must be
 *          binary and `[0, 1]` is used.
 * @param R0 Optional `M x D` matrix of prior outcomes.
 * @param confidence Credibility level for the normal-approximation interval.
 * @param bounds Optional `[lo, hi]` clipping bounds. If omitted, the interval
 *          is clipped to the minimum and maximum reward levels in `w`.
 * @returns `[mu, sigma, lo, hi]`.
 */
export function maxAtKCi(
  R: Matrix,
  k: number,
  w?: readonly number[] | null,
  R0?: Matrix | null,
  confidence = 0.95,
  bounds?: Bounds | null,
): [number, number, number, number] {
  integer(k);
  if (k === 1) {
    return bayesCi(R, w, R0, confidence, bounds);
  }

  const { mu, sigma, levels } = maxAtKBayes(R, k, w, R0);
  const effectiveBounds: Bounds =
    bounds == null ? [Math.min(...levels), Math.max(...levels)] : bounds;
  const [lo, hi] = normalCredibleInterval(
    mu,
    sigma,
    confidence,
    true,
    effectiveBounds,
  );
  return [mu, sigma, lo, hi];
}
