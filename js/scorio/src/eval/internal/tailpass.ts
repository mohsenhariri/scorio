/** Count distributions and posterior moments shared by TailPass utilities. */
import { gammaln } from "./math.js";

export const MAX_STATES = 20_000;
export const MAX_PAIRS = 4_000_000;
export const sum = (x: readonly number[]): number =>
  x.reduce((a, b) => a + b, 0);
export const dot = (a: readonly number[], b: readonly number[]): number =>
  a.reduce((v, x, i) => v + x * b[i]!, 0);
export const clamp = (x: number): number => Math.max(0, Math.min(1, x));
export function positive(x: number, name: string): number {
  if (typeof x !== "number" || !Number.isFinite(x) || x <= 0)
    throw new Error(`${name} must be finite and positive`);
  return x;
}
export function integer(x: number, name = "k", minimum = 1): number {
  if (!Number.isSafeInteger(x) || x < minimum)
    throw new Error(`${name} must be an integer >= ${minimum}`);
  return x;
}
export function unit(x: number, name: string): number {
  if (typeof x !== "number" || !Number.isFinite(x) || x < 0 || x > 1)
    throw new Error(`${name} must be in [0, 1]`);
  return x;
}
export function realVector(x: readonly number[], name: string): number[] {
  if (
    !Array.isArray(x) ||
    !x.every((v) => typeof v === "number" && Number.isFinite(v))
  )
    throw new Error(`${name} must be a finite numeric vector`);
  return [...x];
}
export function variance(x: number): number {
  if (!Number.isFinite(x) || x < -1e-12)
    throw new Error(`Invalid posterior variance: ${x}`);
  return Math.max(0, x);
}
export function normalizeLogs(logs: readonly number[]): number[] {
  const peak = logs.reduce((a, b) => Math.max(a, b), -Infinity);
  const values = logs.map((v) => Math.exp(v - peak));
  const total = sum(values);
  if (!(total > 0) || !Number.isFinite(total))
    throw new Error("Invalid probability normalization");
  return values.map((v) => v / total);
}
export function countGrid(k: number, categories: number): number[][] {
  let size = 1;
  for (let i = 1; i < categories; i++) {
    size *= (k + i) / i;
    if (size > MAX_STATES)
      throw new Error(
        "Exact profile exceeds 20000 count states; use a smaller k/rubric or moment(1, 2, 4)",
      );
  }
  if (k + 1 > MAX_STATES && categories > 1)
    throw new Error("Exact profile exceeds 20000 count states");
  const output: number[][] = [];
  function visit(prefix: number[], remaining: number, slots: number): void {
    if (slots === 1) {
      output.push([...prefix, remaining]);
      return;
    }
    // Descending failure counts gives ascending successes in the binary case.
    for (let n = remaining; n >= 0; n--)
      visit([...prefix, n], remaining - n, slots - 1);
  }
  visit([], k, categories);
  return output;
}
export function logCoefficients(counts: number[][]): number[] {
  return counts.map(
    (row) => gammaln(sum(row) + 1) - sum(row.map((n) => gammaln(n + 1))),
  );
}
export function betaBinomial(k: number, a: number, b: number): number[] {
  const logs = [0];
  for (let j = 0; j < k; j++)
    logs.push(
      logs[j]! +
        Math.log(k - j) -
        Math.log(j + 1) +
        Math.log(a + j) -
        Math.log(b + k - j - 1),
    );
  return normalizeLogs(logs);
}
export function predictive(
  counts: number[][],
  alpha: readonly number[],
  coeff: number[],
): number[] {
  const k = sum(counts[0]!);
  if (alpha.length === 2) return betaBinomial(k, alpha[1]!, alpha[0]!);
  const rising = alpha.map((a) => {
    const values = [0];
    for (let j = 0; j < k; j++) values.push(values[j]! + Math.log(a + j));
    return values;
  });
  return normalizeLogs(
    counts.map((row, i) => coeff[i]! + sum(row.map((n, j) => rising[j]![n]!))),
  );
}
export function binomial(k: number, p: number): number[] {
  if (p <= 0 || p >= 1)
    return Array.from({ length: k + 1 }, (_, j) => +(j === (p <= 0 ? 0 : k)));
  const logs = [0],
    odds = Math.log(p) - Math.log1p(-p);
  for (let j = 0; j < k; j++)
    logs.push(logs[j]! + Math.log(k - j) - Math.log(j + 1) + odds);
  return normalizeLogs(logs);
}
export function conditional(
  counts: number[][],
  p: readonly number[],
  coeff: number[],
): number[] {
  if (p.length === 2) return binomial(sum(counts[0]!), p[1]!);
  return normalizeLogs(
    counts.map(
      (row, i) =>
        coeff[i]! + sum(row.map((n, j) => (n === 0 ? 0 : n * Math.log(p[j]!)))),
    ),
  );
}

/** Normalized Gauss-Jacobi rule via the shifted recurrence and implicit QL.
 * Only the first eigenvector row is needed for the probability weights. */
export function betaQuadrature(
  k: number,
  a: number,
  b: number,
): [number[], number[]] {
  const total = a + b,
    n = k + 1;
  const d = [a / total],
    e: number[] = [Math.sqrt(((a / total) * (b / total)) / (total + 1))];
  for (let j = 1; j <= k; j++)
    d.push(
      ((j + a) / (2 * j + total)) * ((j + total - 1) / (2 * j + total - 1)) +
        (j / (2 * j + total - 1)) * ((j + b - 1) / (2 * j + total - 2)),
    );
  for (let j = 2; j <= k; j++)
    e.push(
      Math.sqrt(
        (j / (2 * j + total - 2)) *
          ((j + a - 1) / (2 * j + total - 2)) *
          ((j + b - 1) / (2 * j + total - 1)) *
          ((j + total - 2) / (2 * j + total - 3)),
      ),
    );
  e[k] = 0;
  const z = new Array<number>(n).fill(0);
  z[0] = 1;
  for (let l = 0; l < n; l++) {
    let iterations = 0;
    for (;;) {
      let m = l;
      while (
        m < n - 1 &&
        Math.abs(e[m]!) >
          Number.EPSILON * (Math.abs(d[m]!) + Math.abs(d[m + 1]!))
      )
        m++;
      if (m === l) break;
      if (++iterations > 100)
        throw new Error("Beta quadrature did not converge");
      let g = (d[l + 1]! - d[l]!) / (2 * e[l]!);
      let r = Math.hypot(g, 1);
      g = d[m]! - d[l]! + e[l]! / (g + (g >= 0 ? r : -r));
      let s = 1,
        c = 1,
        p = 0;
      for (let i = m - 1; i >= l; i--) {
        const f = s * e[i]!,
          b0 = c * e[i]!;
        r = Math.hypot(f, g);
        e[i + 1] = r;
        if (r === 0) {
          d[i + 1]! -= p;
          e[m] = 0;
          break;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1]! - p;
        r = (d[i]! - g) * s + 2 * c * b0;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b0;
        const zi = z[i + 1]!;
        z[i + 1] = s * z[i]! + c * zi;
        z[i] = c * z[i]! - s * zi;
      }
      d[l]! -= p;
      e[l] = g;
      e[m] = 0;
    }
  }
  const weights = z.map((v) => v * v),
    norm = sum(weights);
  return [d.map(clamp), weights.map((v) => v / norm)];
}
export function countCovariance(
  counts: number[][],
  alpha: readonly number[],
  values: number[][],
  coeff: number[],
): number[][] {
  const size = counts.length,
    t = values[0]!.length;
  if (alpha.length !== 2 && Math.max(size, t) ** 2 > MAX_PAIRS)
    throw new Error(
      "Exact covariance exceeds 4000000 count pairs; use shared posterior draws",
    );
  const cov = Array.from({ length: t }, () => new Array<number>(t).fill(0));
  if (alpha.length === 2) {
    const reflect = alpha[1]! > alpha[0]!,
      k = sum(counts[0]!);
    const [nodes, weights] = betaQuadrature(
      k,
      alpha[reflect ? 0 : 1]!,
      alpha[reflect ? 1 : 0]!,
    );
    const v = reflect ? [...values].reverse() : values;
    // Removing a constant before evaluation preserves tiny endpoint uncertainty.
    const evaluated = nodes.map((p) => {
      const pmf = binomial(k, p);
      return Array.from({ length: t }, (_, j) =>
        sum(pmf.map((prob, i) => prob * (v[i]![j]! - v[0]![j]!))),
      );
    });
    const means = Array.from({ length: t }, (_, j) =>
      sum(evaluated.map((v, i) => weights[i]! * v[j]!)),
    );
    for (let i = 0; i < nodes.length; i++)
      for (let j = 0; j < t; j++)
        for (let l = 0; l < t; l++)
          cov[j]![l]! +=
            weights[i]! *
            (evaluated[i]![j]! - means[j]!) *
            (evaluated[i]![l]! - means[l]!);
    return cov;
  }
  const pmf = predictive(counts, alpha, coeff);
  const means = Array.from({ length: t }, (_, j) =>
    sum(values.map((v, i) => pmf[i]! * v[j]!)),
  );
  const centered = values.map((v) => v.map((x, j) => x - means[j]!));
  for (let i = 0; i < size; i++) {
    const next = predictive(
      counts,
      alpha.map((a, j) => a + counts[i]![j]!),
      coeff,
    );
    const expected = Array.from({ length: t }, (_, j) =>
      sum(centered.map((v, l) => next[l]! * v[j]!)),
    );
    for (let j = 0; j < t; j++)
      for (let l = 0; l < t; l++)
        cov[j]![l]! += pmf[i]! * centered[i]![j]! * expected[l]!;
  }
  return cov.map((row, j) => row.map((v, l) => (v + cov[l]![j]!) / 2));
}
export function endpointMoments(
  k: number,
  a: number,
  b: number,
  discovery = false,
): [number, number] {
  if (discovery) [a, b] = [b, a];
  let logMean = 0,
    logSecond = 0,
    logRatio = 0;
  for (let j = 0; j < 2 * k; j++) {
    const complement = b / (a + b + j);
    const term =
      complement <= 0.5
        ? Math.log1p(-complement)
        : Math.log(a + j) - Math.log(a + b + j);
    // log1p loses a when b dominates the denominator completely.
    const stable = Number.isFinite(term)
      ? term
      : Math.log(a + j) - Math.log(a + b + j);
    logSecond += stable;
    if (j < k) {
      logMean += stable;
      logRatio += Math.log1p((k / (a + j)) * (b / (a + b + k + j)));
    }
  }
  return [
    discovery ? -Math.expm1(logMean) : Math.exp(logMean),
    Math.exp(logSecond) * -Math.expm1(-logRatio),
  ];
}
export function integerTerms(k: number, power: number): [number, number[]][] {
  if (power === 1) return [[1, [1]]];
  if (power === 2)
    return [
      [1 / k, [2]],
      [(k - 1) / k, [1, 1]],
    ];
  const terms: [number, number[]][] = [[1 / k ** 3, [4]]];
  if (k >= 2)
    terms.push(
      [(4 * (k - 1)) / k ** 3, [3, 1]],
      [(3 * (k - 1)) / k ** 3, [2, 2]],
    );
  if (k >= 3) terms.push([(6 * ((k - 1) / k) * ((k - 2) / k)) / k, [2, 1, 1]]);
  if (k >= 4)
    terms.push([((k - 1) / k) * ((k - 2) / k) * ((k - 3) / k), [1, 1, 1, 1]]);
  return terms;
}
export function conditionalMoment(
  p: readonly number[],
  scores: readonly number[],
  k: number,
  power: number,
): number {
  return sum(
    integerTerms(k, power).map(([c, factors]) =>
      factors.reduce(
        (v, f) =>
          v *
          dot(
            p,
            scores.map((s) => s ** f),
          ),
        c,
      ),
    ),
  );
}
export function polynomialMoments(
  scores: readonly number[],
  k: number,
  power: number,
  alpha: readonly number[],
): [number, number] {
  let size = 1;
  for (let i = 1; i <= power; i++) size *= (scores.length + i) / i;
  if (size > MAX_STATES)
    throw new Error("Too many rubric categories for exact moment expansion");
  const terms = new Map<string, number>();
  for (const [coefficient, factors] of integerTerms(k, power)) {
    let current = new Map([
      [new Array(scores.length).fill(0).join(","), coefficient],
    ]);
    for (const factor of factors) {
      const next = new Map<string, number>();
      for (const [key, value] of current)
        for (let j = 0; j < scores.length; j++)
          if (scores[j] !== 0) {
            const powers = key.split(",").map(Number);
            powers[j]!++;
            const key2 = powers.join(",");
            next.set(
              key2,
              (next.get(key2) ?? 0) + value * scores[j]! ** factor,
            );
          }
      current = next;
    }
    for (const [key, value] of current)
      terms.set(key, (terms.get(key) ?? 0) + value);
  }
  if (terms.size ** 2 > MAX_PAIRS)
    throw new Error(
      "Too many moment terms for exact variance; use posterior draws",
    );
  const powers = [...terms.keys()].map((key) => key.split(",").map(Number)),
    coefficients = [...terms.values()];
  const total = sum(alpha);
  function monomial(v: number[]): number {
    let result = 1,
      used = 0;
    for (let j = 0; j < v.length; j++)
      for (let n = 0; n < v[j]!; n++)
        result *= (alpha[j]! + n) / (total + used++);
    return result;
  }
  const first = powers.map(monomial),
    mean = dot(coefficients, first);
  let v = 0;
  for (let i = 0; i < powers.length; i++)
    for (let j = 0; j < powers.length; j++)
      v +=
        coefficients[i]! *
        coefficients[j]! *
        (monomial(powers[i]!.map((x, l) => x + powers[j]![l]!)) -
          first[i]! * first[j]!);
  return [mean, variance(v)];
}

/** Exact mean and covariance of binary Bernstein payoffs, without coefficient products. */
export function binaryPayoffMoments(
  k: number,
  a: number,
  b: number,
  values: number[][],
): { mean: number[]; covariance: number[][] } {
  const pmf = betaBinomial(k, a, b),
    counts = Array.from({ length: k + 1 }, (_, j) => [k - j, j]);
  const mean = values[0]!.map((_, j) =>
    sum(values.map((v, i) => pmf[i]! * v[j]!)),
  );
  return { mean, covariance: countCovariance(counts, [b, a], values, []) };
}

/** Center/scale finite rewards without overflowing differences or variances. */
export function scaledRewards(
  values: readonly number[],
): [number, number, number[]] {
  let offset = values[0]!,
    centered = values.map((v) => v - offset);
  if (centered.some((v) => !Number.isFinite(v))) {
    offset = 0;
    centered = [...values];
  }
  const scale = Math.max(...centered.map(Math.abs));
  return [offset, scale, centered.map((v) => (scale === 0 ? 0 : v / scale))];
}
export function logBetaPower(a: number, b: number, k: number): number {
  let value = 0;
  for (let j = 0; j < k; j++) {
    const denominator = a + b + j,
      complement = b / denominator;
    value +=
      complement <= 0.5
        ? Math.log1p(-complement)
        : Math.log(a + j) - Math.log(denominator);
  }
  return value;
}
export function positiveLogDifference(
  logLarge: number,
  logSmall: number,
): number {
  const difference = logSmall - logLarge;
  if (
    difference >
    64 * Number.EPSILON * Math.max(1, Math.abs(logLarge), Math.abs(logSmall))
  )
    throw new Error("Posterior covariance is materially negative");
  return Math.exp(logLarge) * -Math.expm1(Math.min(difference, 0));
}

export function validateWeights(
  weights: readonly number[],
  size: number,
): number[] {
  const v = realVector(weights, "weights");
  if (v.length !== size || v.some((x) => x < 0) || Math.abs(sum(v) - 1) > 1e-12)
    throw new Error(
      "weights must match thresholds, be nonnegative, and sum to one",
    );
  return v;
}

/** Round the average rubric score once, so strict threshold boundaries agree
 * with the reference's extended-precision accumulation. */
export function bankScore(
  counts: readonly number[],
  scores: readonly number[],
  k: number,
): number {
  function product(a: number, b: number): [number, number] {
    const value = a * b,
      split = 134217729;
    const ac = split * a,
      bc = split * b;
    const ah = ac - (ac - a),
      bh = bc - (bc - b);
    const al = a - ah,
      bl = b - bh;
    return [value, ah * bh - value + ah * bl + al * bh + al * bl];
  }
  let value = 0,
    error = 0;
  for (let j = 0; j < counts.length; j++) {
    const [term, residual] = product(counts[j]!, scores[j]!);
    const next = value + term;
    error +=
      (Math.abs(value) >= Math.abs(term)
        ? value - next + term
        : term - next + value) + residual;
    value = next;
  }
  const quotient = value / k,
    [back, residual] = product(quotient, k);
  return quotient + (value - back + (error - residual)) / k;
}
