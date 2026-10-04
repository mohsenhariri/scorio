/** Convex threshold weights. Categorical moments use profile.moment(lam). */
import { betainc } from "../aggregate/internal/math.js";
import { integer, positive, realVector, sum } from "./internal/tailpass.js";

export function uniformWeights(k: number): number[] {
  return new Array<number>(integer(k)).fill(1 / k);
}
export function thresholdWeights(k: number, threshold: number): number[] {
  integer(k);
  integer(threshold, "threshold");
  if (threshold > k) throw new Error("threshold must be <= k");
  return Array.from({ length: k }, (_, i) => +(i + 1 === threshold));
}
export function discoveryWeights(k: number): number[] {
  return thresholdWeights(k, 1);
}
export function stabilityWeights(k: number): number[] {
  return thresholdWeights(k, k);
}
export function momentWeights(k: number, lam: number): number[] {
  integer(k);
  positive(lam, "lam");
  const w = Array.from(
    { length: k },
    (_, i) =>
      Math.exp(lam * Math.log((i + 1) / k)) *
      -Math.expm1(lam * Math.log1p(-1 / (i + 1))),
  );
  const total = sum(w);
  return w.map((v) => v / total);
}
export function betaWeights(k: number, theta: number, kappa: number): number[] {
  integer(k);
  positive(theta, "theta");
  positive(kappa, "kappa");
  if (theta >= 1) throw new Error("theta must be in (0, 1)");
  const a = theta * kappa,
    b = (1 - theta) * kappa;
  positive(a, "Beta alpha");
  positive(b, "Beta beta");
  const cdf = Array.from({ length: k + 1 }, (_, i) => betainc(a, b, i / k));
  const sf = Array.from({ length: k + 1 }, (_, i) =>
    betainc(b, a, (k - i) / k),
  );
  const w = Array.from({ length: k }, (_, i) =>
    cdf[i + 1]! <= 0.5 ? cdf[i + 1]! - cdf[i]! : sf[i]! - sf[i + 1]!,
  );
  if (w.some((x) => !Number.isFinite(x) || x < 0))
    throw new Error("Could not compute finite Beta weights");
  const total = sum(w);
  return w.map((v) => v / total);
}
/** Integrate the continuous maximum-entropy density on fractional thresholds. */
export function maxentWeights(k: number, meanThreshold: number): number[] {
  integer(k);
  positive(meanThreshold, "meanThreshold");
  if (meanThreshold >= 1) throw new Error("meanThreshold must be in (0, 1)");
  if (meanThreshold === 0.5) return uniformWeights(k);
  const distance = Math.min(meanThreshold, 1 - meanThreshold);
  let hi = 1 / distance,
    lo = 0;
  if (!Number.isFinite(hi))
    return meanThreshold < 0.5 ? discoveryWeights(k) : stabilityWeights(k);
  for (let iteration = 0; iteration < 160; iteration++) {
    const tau = lo + (hi - lo) / 2;
    const mean =
      tau < 1e-3
        ? 0.5 - tau / 12 + tau ** 3 / 720
        : 1 / tau - Math.exp(-tau) / -Math.expm1(-tau);
    if (mean > distance) lo = tau;
    else hi = tau;
    if (hi - lo <= Number.EPSILON * Math.max(1, hi)) break;
  }
  const tau = (lo + hi) / 2;
  if (tau === 0) return uniformWeights(k);
  let w = Array.from(
    { length: k },
    (_, i) =>
      (Math.exp(-tau * (i / k)) * -Math.expm1(-tau / k)) / -Math.expm1(-tau),
  );
  const total = sum(w);
  w = w.map((x) => x / total);
  return meanThreshold < 0.5 ? w : w.reverse();
}
export function payoffWeights(payoff: readonly number[]): number[] {
  const v = realVector(payoff, "payoff");
  if (
    v.length < 2 ||
    v[0] !== 0 ||
    v[v.length - 1] !== 1 ||
    v.some((x, i) => i > 0 && x < v[i - 1]!)
  )
    throw new Error(
      "payoff must be nondecreasing, start at zero, and end at one",
    );
  return v.slice(1).map((x, i) => x - v[i]!);
}
export {
  uniformWeights as uniform_weights,
  thresholdWeights as threshold_weights,
  discoveryWeights as discovery_weights,
  stabilityWeights as stability_weights,
  momentWeights as moment_weights,
  betaWeights as beta_weights,
  maxentWeights as maxent_weights,
  payoffWeights as payoff_weights,
};
