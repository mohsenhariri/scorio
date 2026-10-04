/**
 * Stopping and allocation helpers, imported from `scorio/sinf`.
 *
 * Functions consume posterior means and standard deviations from evaluation.
 * They return decisions or model indices; the caller runs any further trials.
 *
 * @module sinf
 */
import { ndtri } from "../eval/internal/math.js";
import { normCdf } from "../rank/internal/special.js";

export interface StopOptions {
  /** Desired confidence level. Default: 0.95. */
  confidence?: number;
  /** Compare normal intervals or pairwise ordering probabilities. */
  method?: "ci_overlap" | "zscore";
}
/** Normal-approximation confidence in the pairwise ordering. */
export function rankingConfidence(
  muA: number,
  sigmaA: number,
  muB: number,
  sigmaB: number,
): { rho: number; z: number } {
  const denominator = Math.hypot(sigmaA, sigmaB);
  if (denominator === 0) return { rho: muA === muB ? 0.5 : 1, z: Infinity };
  const z = Math.abs(muA - muB) / denominator;
  return { rho: normCdf(z), z };
}
/** Build a two-sided normal interval, optionally clipped to a score range. */
export function ciFromMuSigma(
  mu: number,
  sigma: number,
  options: { confidence?: number; clip?: readonly [number, number] } = {},
): { lo: number; hi: number } {
  const confidence = options.confidence ?? 0.95;
  if (!(confidence > 0 && confidence < 1) || sigma < 0)
    throw new Error("confidence must be in (0, 1) and sigma must be >= 0");
  const half = ndtri(0.5 + confidence / 2) * sigma;
  return {
    lo: options.clip ? Math.max(options.clip[0], mu - half) : mu - half,
    hi: options.clip ? Math.min(options.clip[1], mu + half) : mu + half,
  };
}
/**
 * Whether a normal interval is narrow enough at the chosen confidence level.
 * Supply exactly one of `maxCiWidth` (full width) or `maxHalfWidth`.
 */
export function shouldStop(
  sigma: number,
  options: {
    confidence?: number;
    maxCiWidth?: number;
    maxHalfWidth?: number;
  } = {},
): boolean {
  if ((options.maxCiWidth == null) === (options.maxHalfWidth == null))
    throw new Error("Provide exactly one of maxCiWidth or maxHalfWidth");
  const half = ndtri(0.5 + (options.confidence ?? 0.95) / 2) * sigma;
  return options.maxHalfWidth != null
    ? half <= options.maxHalfWidth
    : 2 * half <= options.maxCiWidth!;
}
function leaderOf(
  mus: readonly number[],
  sigmas: readonly number[],
  minimum: number,
): number {
  if (mus.length !== sigmas.length || mus.length < minimum)
    throw new Error(
      `mus and sigmas must have the same length, at least ${minimum}`,
    );
  return mus.reduce((best, value, j) => (value > mus[best]! ? j : best), 0);
}
/**
 * Whether the model with the largest posterior mean is separated from all others.
 * Returns its zero-based index and the indices of remaining ambiguous models.
 */
export function shouldStopTop1(
  mus: readonly number[],
  sigmas: readonly number[],
  options: StopOptions = {},
): { stop: boolean; leader: number; ambiguous: number[] } {
  const leader = leaderOf(mus, sigmas, 1),
    confidence = options.confidence ?? 0.95,
    method = options.method ?? "ci_overlap";
  if (method !== "ci_overlap" && method !== "zscore")
    throw new Error("method must be 'ci_overlap' or 'zscore'");
  const z = ndtri(0.5 + confidence / 2);
  const ambiguous = mus
    .map((_, i) => i)
    .filter(
      (i) =>
        i !== leader &&
        (method === "zscore"
          ? rankingConfidence(
              mus[leader]!,
              sigmas[leader]!,
              mus[i]!,
              sigmas[i]!,
            ).rho < confidence
          : mus[leader]! - z * sigmas[leader]! <= mus[i]! + z * sigmas[i]!),
    );
  return { stop: ambiguous.length === 0, leader, ambiguous };
}
/**
 * Suggest the current leader and its least-separated competitor for more trials.
 * Requires at least two models. Both returned indices are zero-based.
 */
export function suggestNextAllocation(
  mus: readonly number[],
  sigmas: readonly number[],
  options: StopOptions = {},
): { leader: number; competitor: number } {
  const leader = leaderOf(mus, sigmas, 2),
    confidence = options.confidence ?? 0.95,
    method = options.method ?? "ci_overlap";
  if (method !== "ci_overlap" && method !== "zscore")
    throw new Error("method must be 'ci_overlap' or 'zscore'");
  const z = ndtri(0.5 + confidence / 2);
  const candidates = mus.map((_, i) => i).filter((i) => i !== leader);
  const separation = (i: number) =>
    method === "zscore"
      ? rankingConfidence(mus[leader]!, sigmas[leader]!, mus[i]!, sigmas[i]!).z
      : mus[leader]! - z * sigmas[leader]! - (mus[i]! + z * sigmas[i]!);
  const competitor = candidates.reduce((best, i) =>
    separation(i) < separation(best) ? i : best,
  );
  return { leader, competitor };
}
export {
  rankingConfidence as ranking_confidence,
  ciFromMuSigma as ci_from_mu_sigma,
  shouldStop as should_stop,
  shouldStopTop1 as should_stop_top1,
  suggestNextAllocation as suggest_next_allocation,
};
