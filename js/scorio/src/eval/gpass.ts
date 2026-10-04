/**
 * Generalized pass-family metrics for binary outcomes — G-Pass@k, the
 * thresholded G-Pass@k_τ (at least `ceil(τk)` successes), and mG-Pass@k
 * (the mean over thresholds `τ ∈ [0.5, 1.0]`), with Beta-posterior credible
 * intervals. Port of `scorio/eval/gpass.py`.
 *
 * References: Liu et al. (2024), arXiv:2412.13147; Yao et al. (2024),
 * arXiv:2406.12045.
 */

import {
  positive,
  endpointMoments,
  binaryPayoffMoments,
  sum,
} from "./internal/tailpass.js";
import { hypergeomPmf, hypergeomSf } from "./internal/math.js";
import { normalCredibleInterval, type Bounds } from "./internal/ci.js";
import {
  asMatrix,
  rowSums,
  validateBinary,
  type Matrix,
} from "./internal/validate.js";
import { passAtK, passHatK, passHatKCi } from "./passAtK.js";

function checkK(k: number, N: number): void {
  if (!Number.isSafeInteger(k) || !(k >= 1 && k <= N)) {
    throw new Error(`k must satisfy 1 <= k <= N (N=${N}); got k=${k}`);
  }
}

function checkTau(tau: number): void {
  if (!(tau >= 0.0 && tau <= 1.0)) {
    throw new Error(`tau must be in [0, 1]; got ${tau}`);
  }
}

/**
 * Success threshold `max(1, ceil(τk))` using the representable `j / k`
 * boundaries. Comparing `tau` with the boundary avoids multiplication
 * rounding at grid points and preserves even the next float above them,
 * without a fixed tolerance.
 */
function tauThreshold(tau: number, k: number): number {
  let threshold = Math.floor(tau * k);
  if (tau > threshold / k) threshold += 1;
  return Math.max(1, threshold);
}

/** Per-row Beta posterior parameters `[alpha, beta]` for binary outcomes. */
function binaryBetaPosterior(
  Rm: readonly (readonly number[])[],
  alpha0: number,
  beta0: number,
): { alpha: number[]; beta: number[]; N: number } {
  positive(alpha0, "alpha0");
  positive(beta0, "beta0");
  validateBinary(Rm);
  const N = Rm[0]!.length;
  const c = rowSums(Rm);
  return {
    alpha: c.map((ci) => alpha0 + ci),
    beta: c.map((ci) => beta0 + (N - ci)),
    N,
  };
}

/**
 * G-Pass@k: the all-success (`τ = 1`) threshold, an alias for Pass^k.
 *
 * Included for literature using the G-Pass@k naming convention.
 */
export function gPassAtK(R: Matrix, k: number): number {
  return passHatK(R, k);
}

/**
 * G-Pass@k_τ: average probability of at least `ceil(τk)` successes among `k`
 * selected samples (unbiased hypergeometric estimator). `τ = 0` reduces to
 * Pass@k and `τ = 1` to Pass^k.
 */
export function gPassAtKTau(R: Matrix, k: number, tau: number): number {
  const Rm = asMatrix(R);
  validateBinary(Rm);
  const N = Rm[0]!.length;
  checkTau(tau);
  checkK(k, N);
  if (!Number.isInteger(k)) return NaN;

  if (tau <= 0.0) {
    return passAtK(Rm, k);
  }

  const nu = rowSums(Rm);
  const j0 = tauThreshold(tau, k);
  const M = Rm.length;
  const vals = nu.map((v) => hypergeomSf(N, v, k, j0));
  return vals.reduce((s, v) => s + v, 0) / M;
}

/**
 * mG-Pass@k: the mean generalized pass metric, `2 ∫_{0.5}^{1} G-Pass@k_τ dτ`.
 *
 * The integral over `τ ∈ [0.5, 1.0]` collapses to the closed form
 * `(2/k) · Σ_{j=m+1}^{k} (j - m) · P(X = j)`, where `m = ceil(k/2)` is the
 * majority threshold and `X ~ Hypergeometric(N, ν, k)`.
 */
export function mgPassAtK(R: Matrix, k: number): number {
  const Rm = asMatrix(R);
  validateBinary(Rm);
  const N = Rm[0]!.length;
  checkK(k, N);
  if (!Number.isInteger(k)) {
    throw new TypeError("'number' object cannot be interpreted as an integer");
  }

  const nu = rowSums(Rm);

  const majority = Math.ceil(0.5 * k);
  if (majority >= k) {
    return 0.0;
  }

  const M = Rm.length;
  const vals = new Array<number>(M).fill(0);
  for (let j = majority + 1; j <= k; j++) {
    for (let i = 0; i < M; i++) {
      const v = nu[i]!;
      const pmf = hypergeomPmf(N, v, k, j);
      vals[i]! += (j - majority) * pmf;
    }
  }
  return vals.reduce((s, v) => s + (v * 2.0) / k, 0) / M;
}

/** Posterior mean/std for the i.i.d. G-Pass@k_τ quantity. */
function gPassAtKTauBayes(
  R: Matrix,
  k: number,
  tau: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const Rm = asMatrix(R),
    { alpha, beta, N } = binaryBetaPosterior(Rm, alpha0, beta0);
  checkK(k, N);
  checkTau(tau);
  if (tau <= 0) return passAtKBayes(Rm, k, alpha0, beta0);
  if (tau >= 1) return passHatKBayes(Rm, k, alpha0, beta0);
  const cutoff = tauThreshold(tau, k),
    values = Array.from({ length: k + 1 }, (_, j) => [+(j >= cutoff)]);
  const moments = alpha.map((a, i) =>
    binaryPayoffMoments(k, a, beta[i]!, values),
  );
  return [
    sum(moments.map((v) => v.mean[0]!)) / Rm.length,
    Math.sqrt(sum(moments.map((v) => v.covariance[0]![0]!))) / Rm.length,
  ];
}

/** Posterior mean/std for the i.i.d. mG-Pass@k quantity. */
function mgPassAtKBayes(
  R: Matrix,
  k: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const Rm = asMatrix(R),
    { alpha, beta, N } = binaryBetaPosterior(Rm, alpha0, beta0);
  checkK(k, N);
  const majority = Math.ceil(k / 2),
    values = Array.from({ length: k + 1 }, (_, j) => [
      (2 / k) * Math.max(j - majority, 0),
    ]);
  const moments = alpha.map((a, i) =>
    binaryPayoffMoments(k, a, beta[i]!, values),
  );
  return [
    sum(moments.map((v) => v.mean[0]!)) / Rm.length,
    Math.sqrt(sum(moments.map((v) => v.covariance[0]![0]!))) / Rm.length,
  ];
}

/** Posterior mean/std for the i.i.d. Pass@k quantity `1 - (1-p)^k`. */
function passAtKBayes(
  R: Matrix,
  k: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const Rm = asMatrix(R),
    { alpha, beta, N } = binaryBetaPosterior(Rm, alpha0, beta0);
  checkK(k, N);
  const moments = alpha.map((a, i) => endpointMoments(k, a, beta[i]!, true));
  return [
    sum(moments.map((v) => v[0])) / Rm.length,
    Math.sqrt(sum(moments.map((v) => v[1]))) / Rm.length,
  ];
}

/** Posterior mean/std for the i.i.d. Pass^k quantity `p^k`. */
function passHatKBayes(
  R: Matrix,
  k: number,
  alpha0: number,
  beta0: number,
): [number, number] {
  const Rm = asMatrix(R),
    { alpha, beta, N } = binaryBetaPosterior(Rm, alpha0, beta0);
  checkK(k, N);
  const moments = alpha.map((a, i) => endpointMoments(k, a, beta[i]!, false));
  return [
    sum(moments.map((v) => v[0])) / Rm.length,
    Math.sqrt(sum(moments.map((v) => v[1]))) / Rm.length,
  ];
}

/**
 * Bayesian `[mu, sigma, lo, hi]` for G-Pass@k (alias for Pass^k posterior).
 */
export function gPassAtKCi(
  R: Matrix,
  k: number,
  confidence = 0.95,
  bounds: Bounds | null = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  return passHatKCi(R, k, confidence, bounds, alpha0, beta0);
}

/** Bayesian `[mu, sigma, lo, hi]` for thresholded G-Pass@k_τ. */
export function gPassAtKTauCi(
  R: Matrix,
  k: number,
  tau: number,
  confidence = 0.95,
  bounds: Bounds | null = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  const [mu, sigma] = gPassAtKTauBayes(R, k, tau, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}

/** Bayesian `[mu, sigma, lo, hi]` for mG-Pass@k. */
export function mgPassAtKCi(
  R: Matrix,
  k: number,
  confidence = 0.95,
  bounds: Bounds | null = [0.0, 1.0],
  alpha0 = 1.0,
  beta0 = 1.0,
): [number, number, number, number] {
  const [mu, sigma] = mgPassAtKBayes(R, k, alpha0, beta0);
  const [lo, hi] = normalCredibleInterval(mu, sigma, confidence, true, bounds);
  return [mu, sigma, lo, hi];
}
