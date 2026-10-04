/** TailPass profiles are posterior objects, with no default scalar utility.
 * Hariri et al. (2026), Success Has a Shape, Sections 2–3 and Appendix C.
 */
import {
  asMatrix,
  asPriorMatrix,
  rowBincount,
  validateMatrixRange,
  type Matrix,
} from "./internal/validate.js";
import { normalCredibleInterval, zValue } from "./internal/ci.js";
import { logComb } from "./internal/math.js";
import { SeededRng } from "../rank/internal/rng.js";
import { uniformWeights } from "./tailpassWeights.js";
import {
  bankScore,
  validateWeights,
  MAX_STATES,
  MAX_PAIRS,
  clamp,
  conditional,
  conditionalMoment,
  countCovariance,
  countGrid,
  dot,
  endpointMoments,
  integer,
  logCoefficients,
  normalizeLogs,
  polynomialMoments,
  positive,
  predictive,
  realVector,
  sum,
  unit,
  variance,
} from "./internal/tailpass.js";

export interface TailPassOptions {
  eta?: number;
  /** Scalar, category vector, or question-by-category concentrations. */
  prior?: number | readonly number[] | readonly (readonly number[])[];
  thresholds?: readonly number[];
}
export interface SamplingOptions {
  rng?: number | { random(): number };
}
export interface TailPassCIOptions extends SamplingOptions {
  method?: "mc" | "normal";
  nDraws?: number;
}
export type ScalarSummary = [number, number, number, number];
export type ProfileSummary = [number[], number[], number[], number[]];
export type Aggregation = "question" | "profile";
function thresholdsFor(k: number, thresholds?: readonly number[]): number[] {
  integer(k);
  if (thresholds === undefined) {
    if (k > MAX_STATES)
      throw new Error(
        "Default threshold grid is too large; supply explicit thresholds",
      );
    return Array.from({ length: k }, (_, i) => (i + 1) / k);
  }
  const grid = realVector(thresholds, "thresholds");
  if (
    !grid.length ||
    grid.some((v, i) => v < 0 || v > 1 || (i > 0 && v <= grid[i - 1]!))
  )
    throw new Error(
      "thresholds must be nonempty and strictly increasing in [0, 1]",
    );
  return grid;
}
function prepare(R: Matrix, w?: readonly number[] | null, R0?: Matrix | null) {
  const rows = asMatrix(R),
    scores = realVector(w ?? [0, 1], "w");
  if (
    !rows.length ||
    !rows[0]!.length ||
    !scores.length ||
    scores.some((v) => v < 0 || v > 1)
  )
    throw new Error("R must be nonempty and w must contain scores in [0, 1]");
  validateMatrixRange(rows, 0, scores.length - 1, "R");
  const aux = R0 == null ? rows.map(() => []) : asPriorMatrix(R0, rows.length);
  if (aux.length !== rows.length)
    throw new Error("R0 must have the same question count as R");
  validateMatrixRange(aux, 0, scores.length - 1, "R0");
  return {
    counts: rowBincount(rows, scores.length),
    priorCounts: rowBincount(aux, scores.length),
    scores,
    N: rows[0]!.length,
  };
}
function grouped(
  parameters: number[][],
  scores: number[],
): [number[][], number[]] {
  const levels = [...new Set(scores)].sort((a, b) => a - b);
  return [
    parameters.map((row) =>
      levels.map((s) => sum(row.filter((_, j) => scores[j] === s))),
    ),
    levels,
  ];
}
/** Fit prior + counts(R) + eta*counts(R0); binary prior order is failure, success. */
export function tailpass(
  R: Matrix,
  k: number,
  w?: readonly number[] | null,
  R0?: Matrix | null,
  options: TailPassOptions = {},
): TailPassProfile {
  const grid = thresholdsFor(k, options.thresholds),
    bank = prepare(R, w, R0);
  const eta = unit(options.eta ?? 1, "eta"),
    prior = options.prior ?? 1;
  const parameters = bank.counts.map((row, i) =>
    row.map((n, j) => {
      let value: number;
      if (typeof prior === "number") value = prior;
      else if (Array.isArray(prior[0])) {
        const matrix = prior as readonly (readonly number[])[];
        if (matrix.length !== bank.counts.length && matrix.length !== 1)
          throw new Error("prior must match the question count");
        const r = matrix[matrix.length === 1 ? 0 : i]!;
        if (r.length !== row.length && r.length !== 1)
          throw new Error("prior must match the category count");
        value = r[r.length === 1 ? 0 : j]!;
      } else {
        const vector = prior as readonly number[];
        if (vector.length !== row.length && vector.length !== 1)
          throw new Error("prior must match the category count");
        value = vector[vector.length === 1 ? 0 : j]!;
      }
      return positive(value, "prior") + n + eta * bank.priorCounts[i]![j]!;
    }),
  );
  if (parameters.some((row) => !Number.isFinite(sum(row))))
    throw new Error("Posterior concentration totals must be finite");
  const [params, scores] = grouped(parameters, bank.scores);
  return new TailPassProfile(k, grid, params, scores);
}

/** A posterior profile. Arrays are defensive, frozen copies. */
export class TailPassProfile {
  readonly thresholds: readonly number[];
  readonly parameters: readonly (readonly number[])[];
  readonly scores: readonly number[];
  private cachedGrid?: {
    counts: number[][];
    scores: number[];
    coeff: number[];
    order: number[];
  };
  private cachedMean?: number[][];
  private cachedCov?: number[][];
  constructor(
    readonly k: number,
    thresholds: readonly number[],
    parameters: readonly (readonly number[])[],
    scores: readonly number[],
  ) {
    this.thresholds = Object.freeze([...thresholds]);
    this.parameters = Object.freeze(
      parameters.map((v) => Object.freeze([...v])),
    );
    this.scores = Object.freeze([...scores]);
  }
  get questionCount(): number {
    return this.parameters.length;
  }
  get question_count(): number {
    return this.questionCount;
  }
  /** @internal */
  get grid() {
    if (!this.cachedGrid) {
      const counts = countGrid(this.k, this.scores.length);
      const scores = counts.map((row) => bankScore(row, this.scores, this.k));
      const order = scores
        .map((_, i) => i)
        .sort((a, b) => scores[a]! - scores[b]!);
      this.cachedGrid = {
        counts,
        scores,
        coeff: logCoefficients(counts),
        order,
      };
    }
    return this.cachedGrid;
  }
  /** @internal */
  tails(pmf: readonly number[]): number[] {
    const { order, scores } = this.grid,
      tails = new Array<number>(order.length + 1).fill(0);
    for (let i = order.length - 1; i >= 0; i--)
      tails[i] = tails[i + 1]! + pmf[order[i]!]!;
    return this.thresholds.map((t) => {
      let lo = 0,
        hi = order.length;
      while (lo < hi) {
        const m = (lo + hi) >>> 1;
        if (scores[order[m]!]! < t) lo = m + 1;
        else hi = m;
      }
      return clamp(tails[lo]!);
    });
  }
  get questionMean(): number[][] {
    if (!this.cachedMean) {
      const g = this.grid;
      this.cachedMean = this.parameters.map((a) =>
        this.tails(predictive(g.counts, a, g.coeff)),
      );
    }
    return this.cachedMean.map((row) => [...row]);
  }
  get question_mean(): number[][] {
    return this.questionMean;
  }
  get mean(): number[] {
    const rows = this.questionMean;
    return this.thresholds.map(
      (_, j) => sum(rows.map((row) => row[j]!)) / this.questionCount,
    );
  }
  get covariance(): number[][] {
    if (!this.cachedCov) {
      const g = this.grid;
      if (Math.max(g.counts.length, this.thresholds.length) ** 2 > MAX_PAIRS)
        throw new Error(
          "Exact covariance is too large; use shared posterior draws",
        );
      const values = g.scores.map((s) => this.thresholds.map((t) => +(s >= t)));
      const cov = this.thresholds.map(() => this.thresholds.map(() => 0));
      for (const a of this.parameters) {
        const c = countCovariance(g.counts, a, values, g.coeff);
        for (let j = 0; j < cov.length; j++)
          for (let l = 0; l < cov.length; l++)
            cov[j]![l]! += c[j]![l]! / this.questionCount ** 2;
      }
      cov.forEach((row, j) => {
        row[j] = variance(row[j]!);
      });
      this.cachedCov = cov;
    }
    return this.cachedCov.map((row) => [...row]);
  }
  get std(): number[] {
    return this.covariance.map((row, j) => Math.sqrt(row[j]!));
  }
  private aggregate(moments: [number, number][]): [number, number] {
    return [
      sum(moments.map((v) => v[0])) / this.questionCount,
      Math.sqrt(
        variance(sum(moments.map((v) => v[1])) / this.questionCount ** 2),
      ),
    ];
  }
  private payoff(values: number[]): [number, number] {
    const g = this.grid;
    return this.aggregate(
      this.parameters.map((a) => [
        dot(predictive(g.counts, a, g.coeff), values),
        countCovariance(
          g.counts,
          a,
          values.map((v) => [v]),
          g.coeff,
        )[0]![0]!,
      ]),
    );
  }
  /** Exact mean/std, including cross-threshold covariance. */
  linear(weights: readonly number[]): [number, number] {
    const w = validateWeights(weights, this.thresholds.length);
    return this.payoff(
      this.grid.scores.map((s) =>
        sum(w.filter((_, j) => s >= this.thresholds[j]!)),
      ),
    );
  }
  /** Integrate the full attainable score distribution, independently of the grid. */
  moment(lam: number): [number, number] {
    positive(lam, "lam");
    if (lam === 1)
      return this.aggregate(
        this.parameters.map((a) => {
          const total = sum(a),
            p = a.map((v) => v / total),
            mean = dot(p, this.scores);
          return [
            mean,
            sum(p.map((v, j) => v * (this.scores[j]! - mean) ** 2)) /
              (total + 1),
          ];
        }),
      );
    if (
      (lam === 2 || lam === 4) &&
      (this.scores.length !== 2 || this.k >= MAX_STATES)
    )
      return this.aggregate(
        this.parameters.map((a) =>
          polynomialMoments(this.scores, this.k, lam, a),
        ),
      );
    return this.payoff(this.grid.scores.map((s) => s ** lam));
  }
  private endpoint(full: boolean): [number, number] {
    const mask = this.scores.map((s) => (full ? s === 1 : s > 0));
    if (mask.every(Boolean) || mask.every((v) => !v))
      return [+mask.every(Boolean), 0];
    return this.aggregate(
      this.parameters.map((a) =>
        endpointMoments(
          this.k,
          sum(a.filter((_, j) => mask[j])),
          sum(a.filter((_, j) => !mask[j])),
          !full,
        ),
      ),
    );
  }
  discovery(): [number, number] {
    return this.endpoint(false);
  }
  stability(): [number, number] {
    return this.endpoint(true);
  }
  atK(
    k: number,
    options: Pick<TailPassOptions, "thresholds"> = {},
  ): TailPassProfile {
    return new TailPassProfile(
      integer(k),
      thresholdsFor(k, options.thresholds),
      this.parameters,
      this.scores,
    );
  }
  at_k(
    k: number,
    options: Pick<TailPassOptions, "thresholds"> = {},
  ): TailPassProfile {
    return this.atK(k, options);
  }
  /** Seeded draws are reproducible within JS, not bit-identical to NumPy. */
  sample(nDraws = 4000, options: SamplingOptions = {}): TailPassDraws {
    integer(nDraws, "nDraws", 2);
    const rng =
      typeof options.rng === "number"
        ? new SeededRng(integer(options.rng, "rng", 0))
        : (options.rng ?? { random: Math.random });
    const uniform = () => Math.max(Number.MIN_VALUE, rng.random());
    function logGamma(shape: number): number {
      if (shape < 1) return logGamma(shape + 1) + Math.log(uniform()) / shape;
      const d = shape - 1 / 3,
        c = 1 / Math.sqrt(9 * d);
      for (;;) {
        const x =
            Math.sqrt(-2 * Math.log(uniform())) *
            Math.cos(2 * Math.PI * uniform()),
          base = 1 + c * x;
        if (base <= 0) continue;
        const v = base ** 3,
          u = uniform();
        if (
          u < 1 - 0.0331 * x ** 4 ||
          Math.log(u) < (x * x) / 2 + d * (1 - v + Math.log(v))
        )
          return Math.log(d) + 3 * Math.log(base);
      }
    }
    const probabilities = Array.from({ length: nDraws }, () =>
      this.parameters.map((a) =>
        a.length === 1 ? [1] : normalizeLogs(a.map(logGamma)),
      ),
    );
    return new TailPassDraws(this, probabilities);
  }
  ci(confidence = 0.95, options: TailPassCIOptions = {}): ProfileSummary {
    const z = zValue(confidence);
    if ((options.method ?? "mc") === "mc")
      return this.sample(options.nDraws, options).summary(
        undefined,
        confidence,
      );
    if (options.method !== "normal")
      throw new Error("method must be 'mc' or 'normal'");
    const mu = this.mean,
      std = this.std;
    return [
      mu,
      std,
      mu.map((m, i) => clamp(m - z * std[i]!)),
      mu.map((m, i) => clamp(m + z * std[i]!)),
    ];
  }
  linearCi(
    weights: readonly number[],
    confidence = 0.95,
    options: TailPassCIOptions = {},
  ): ScalarSummary {
    validateWeights(weights, this.thresholds.length);
    zValue(confidence);
    if (options.method === "normal") {
      const [mu, std] = this.linear(weights);
      return [
        mu,
        std,
        ...normalCredibleInterval(mu, std, confidence, true, [0, 1]),
      ];
    }
    if (options.method !== undefined && options.method !== "mc")
      throw new Error("method must be 'mc' or 'normal'");
    const d = this.sample(options.nDraws, options);
    return d.summary(d.linear(weights), confidence);
  }
  momentCi(
    lam: number,
    confidence = 0.95,
    options: TailPassCIOptions = {},
  ): ScalarSummary {
    positive(lam, "lam");
    zValue(confidence);
    if (options.method === "normal") {
      const [mu, std] = this.moment(lam);
      return [
        mu,
        std,
        ...normalCredibleInterval(mu, std, confidence, true, [0, 1]),
      ];
    }
    if (options.method !== undefined && options.method !== "mc")
      throw new Error("method must be 'mc' or 'normal'");
    const d = this.sample(options.nDraws, options);
    return d.summary(d.moment(lam), confidence);
  }
  linear_ci(
    weights: readonly number[],
    confidence = 0.95,
    options: TailPassCIOptions = {},
  ): ScalarSummary {
    return this.linearCi(weights, confidence, options);
  }
  moment_ci(
    lam: number,
    confidence = 0.95,
    options: TailPassCIOptions = {},
  ): ScalarSummary {
    return this.momentCi(lam, confidence, options);
  }
}

/** Shared latent posterior draws: transform each question before averaging by default. */
export class TailPassDraws {
  private cachedProfile?: number[][];
  constructor(
    private readonly source: TailPassProfile,
    private readonly probabilities: readonly (readonly (readonly number[])[])[],
  ) {}
  private apply(
    transform: (tails: number[]) => number,
    aggregation: Aggregation = "question",
  ): number[] {
    if (aggregation === "profile") return this.profile.map(transform);
    if (aggregation !== "question")
      throw new Error("aggregation must be 'question' or 'profile'");
    const g = this.source.grid;
    return this.probabilities.map(
      (draw) =>
        sum(
          draw.map((p) =>
            transform(this.source.tails(conditional(g.counts, p, g.coeff))),
          ),
        ) / this.source.questionCount,
    );
  }
  get profile(): number[][] {
    if (!this.cachedProfile) {
      const g = this.source.grid;
      this.cachedProfile = this.probabilities.map((draw) => {
        const result = this.source.thresholds.map(() => 0);
        for (const p of draw)
          this.source
            .tails(conditional(g.counts, p, g.coeff))
            .forEach((v, j) => {
              result[j]! += v / this.source.questionCount;
            });
        return result;
      });
    }
    return this.cachedProfile.map((row) => [...row]);
  }
  linear(weights: readonly number[]): number[] {
    const w = validateWeights(weights, this.source.thresholds.length);
    return this.profile.map((row) => dot(row, w));
  }
  moment(lam: number): number[] {
    positive(lam, "lam");
    if ([1, 2, 4].includes(lam))
      return this.probabilities.map(
        (draw) =>
          sum(
            draw.map((p) =>
              conditionalMoment(p, this.source.scores, this.source.k, lam),
            ),
          ) / this.source.questionCount,
      );
    const g = this.source.grid,
      payoff = g.scores.map((s) => s ** lam);
    return this.probabilities.map(
      (draw) =>
        sum(draw.map((p) => dot(conditional(g.counts, p, g.coeff), payoff))) /
        this.source.questionCount,
    );
  }
  powerMean(
    q: number,
    weights?: readonly number[],
    options: { aggregation?: Aggregation } = {},
  ): number[] {
    positive(q, "q");
    const w = validateWeights(
      weights ?? uniformWeights(this.source.thresholds.length),
      this.source.thresholds.length,
    );
    return this.apply((values) => {
      if (q === 1) return dot(values, w);
      const scale = Math.max(...values.filter((_, j) => w[j]! > 0));
      if (scale === 0) return 0;
      const logs = values.map((v, j) =>
        w[j] === 0 ? -Infinity : q * Math.log(v / scale),
      );
      const delta = sum(logs.map((v, j) => w[j]! * Math.expm1(v)));
      const logMean =
        Math.abs(delta) < 0.25
          ? Math.log1p(delta)
          : Math.log(sum(logs.map((v, j) => w[j]! * Math.exp(v))));
      return scale * Math.exp(logMean / q);
    }, options.aggregation);
  }
  power_mean(
    q: number,
    weights?: readonly number[],
    options: { aggregation?: Aggregation } = {},
  ): number[] {
    return this.powerMean(q, weights, options);
  }
  qrs(
    weights?: readonly number[],
    options: { aggregation?: Aggregation } = {},
  ): number[] {
    return this.powerMean(2, weights, options);
  }
  rollout(m: number, weights?: readonly number[]): number[] {
    integer(m, "m", 2);
    const w = validateWeights(
      weights ?? uniformWeights(this.source.thresholds.length),
      this.source.thresholds.length,
    );
    return this.apply((values) =>
      dot(
        values.map((v) => v ** m),
        w,
      ),
    );
  }
  harmonic(alpha = 0.5): number[] {
    unit(alpha, "alpha");
    return this.probabilities.map(
      (draw) =>
        sum(
          draw.map((p) => {
            const zero = clamp(
                sum(p.filter((_, j) => this.source.scores[j] === 0)),
              ),
              one = clamp(sum(p.filter((_, j) => this.source.scores[j] === 1)));
            const discovery = -Math.expm1(this.source.k * Math.log(zero)),
              stability = one ** this.source.k;
            if (alpha === 1) return discovery;
            if (alpha === 0) return stability;
            return discovery > 0 && stability > 0
              ? 1 / (alpha / discovery + (1 - alpha) / stability)
              : 0;
          }),
        ) / this.source.questionCount,
    );
  }
  discovery(): number[] {
    return this.harmonic(1);
  }
  stability(): number[] {
    return this.harmonic(0);
  }
  shortfall(
    target: readonly number[],
    weights?: readonly number[],
    options: { epsilon?: number } = {},
  ): number[] {
    const t = realVector(target, "target"),
      size = this.source.thresholds.length;
    if (
      t.length !== size ||
      t.some((v, j) => v < 0 || v > 1 || (j > 0 && v > t[j - 1]!))
    )
      throw new Error(
        "target must be nonincreasing in [0, 1] and match thresholds",
      );
    const w = realVector(weights ?? new Array<number>(size).fill(1), "weights");
    if (w.length !== size || w.some((v) => v <= 0))
      throw new Error(
        "shortfall weights must be positive and match thresholds",
      );
    const epsilon = positive(options.epsilon ?? 0.01, "epsilon");
    return this.apply((values) => {
      const gaps = values.map((v, j) => Math.max(t[j]! - v, 0) * w[j]!);
      return Math.max(...gaps) + epsilon * sum(gaps);
    });
  }
  atK(
    k: number,
    options: Pick<TailPassOptions, "thresholds"> = {},
  ): TailPassDraws {
    return new TailPassDraws(this.source.atK(k, options), this.probabilities);
  }
  at_k(
    k: number,
    options: Pick<TailPassOptions, "thresholds"> = {},
  ): TailPassDraws {
    return this.atK(k, options);
  }
  summary(values?: undefined, confidence?: number): ProfileSummary;
  summary(values: readonly number[], confidence?: number): ScalarSummary;
  summary(
    values: readonly (readonly number[])[],
    confidence?: number,
  ): ProfileSummary;
  summary(
    values?: readonly number[] | readonly (readonly number[])[],
    confidence = 0.95,
  ): ScalarSummary | ProfileSummary {
    zValue(confidence);
    const samples = values ?? this.profile;
    if (samples.length !== this.probabilities.length)
      throw new Error("values must match this draw set's sample count");
    const matrix = Array.isArray(samples[0]);
    const rows = matrix
      ? (samples as readonly (readonly number[])[])
      : (samples as readonly number[]).map((v) => [v]);
    const size = rows[0]!.length;
    if (
      !size ||
      rows.some(
        (row) =>
          row.length !== size || realVector(row, "values").length !== size,
      )
    )
      throw new Error("values must be rectangular");
    const output = Array.from({ length: 4 }, () => [] as number[]);
    for (let j = 0; j < size; j++) {
      const column = rows.map((row) => row[j]!),
        mean = sum(column) / column.length;
      const std = Math.sqrt(
        sum(column.map((v) => (v - mean) ** 2)) / (column.length - 1),
      );
      column.sort((a, b) => a - b);
      const quantile = (p: number) => {
        const at = p * (column.length - 1),
          i = Math.floor(at);
        return (
          column[i]! +
          (at - i) * (column[Math.min(i + 1, column.length - 1)]! - column[i]!)
        );
      };
      [
        mean,
        std,
        quantile((1 - confidence) / 2),
        quantile((1 + confidence) / 2),
      ].forEach((v, i) => output[i]!.push(v));
    }
    return matrix
      ? (output as ProfileSummary)
      : (output.map((v) => v[0]!) as ScalarSummary);
  }
}

/** Finite-bank sampling without replacement, requiring k <= N. */
export function tailpassEmpirical(
  R: Matrix,
  k: number,
  w?: readonly number[] | null,
  options: Pick<TailPassOptions, "thresholds"> = {},
): number[] {
  const bank = prepare(R, w),
    grid = thresholdsFor(k, options.thresholds);
  if (k > bank.N)
    throw new Error("Empirical k must not exceed the observed trial count");
  const [counts, scores] = grouped(bank.counts, bank.scores),
    profile = new TailPassProfile(k, grid, counts, scores),
    g = profile.grid;
  const values = counts.map((row) =>
    profile.tails(
      normalizeLogs(
        g.counts.map((state) => sum(state.map((v, j) => logComb(row[j]!, v)))),
      ),
    ),
  );
  return grid.map((_, j) => sum(values.map((row) => row[j]!)) / counts.length);
}
export { tailpassEmpirical as tailpass_empirical };
