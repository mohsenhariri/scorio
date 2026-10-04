import { describe, expect, it } from "vitest";
import {
  tailpass,
  tailpassEmpirical,
  TailPassDraws,
  tailpassWeights as w,
  type TailPassOptions,
} from "../src/eval/index.js";
import fixtures from "./fixtures/tailpass.json";

function close(actual: unknown, expected: unknown, tolerance = 2e-10): void {
  if (Array.isArray(expected)) {
    expect(Array.isArray(actual)).toBe(true);
    expect((actual as unknown[]).length).toBe(expected.length);
    expected.forEach((v, i) => close((actual as unknown[])[i], v, tolerance));
  } else
    expect(
      Math.abs((actual as number) - (expected as number)),
    ).toBeLessThanOrEqual(tolerance);
}
describe("TailPass Python reference parity", () => {
  fixtures.cases.forEach((c, index) => {
    it(`profile, empirical, and all utility families: case ${index}`, () => {
      const p = tailpass(c.R, c.k, c.w, c.R0, c.options as TailPassOptions);
      close(p.mean, c.mean);
      close(p.std, c.std);
      close(p.questionMean, c.question_mean);
      close(p.covariance, c.covariance);
      const weights = w.uniformWeights(p.thresholds.length);
      close(p.linear(weights), c.linear);
      close(p.discovery(), c.discovery);
      close(p.stability(), c.stability);
      for (const m of c.moments) close(p.moment(m.lam), m.result);
      close(
        tailpassEmpirical(c.R, Math.min(c.k, c.R[0]!.length), c.w, {
          thresholds: c.options.thresholds,
        }),
        c.empirical,
      );
      const d = new TailPassDraws(p, c.draws.probabilities);
      close(d.profile, c.draws.profile);
      close(d.linear(weights), c.draws.linear);
      close(d.moment(2), c.draws.moment);
      close(d.qrs(), c.draws.qrs);
      close(d.qrs(undefined, { aggregation: "profile" }), c.draws.qrs_profile);
      close(d.rollout(3), c.draws.rollout);
      close(d.harmonic(), c.draws.harmonic);
      close(d.summary(), c.draws.summary);
      const target = p.thresholds.map(
        (_, j) => 1 - (0.5 * j) / (p.thresholds.length - 1),
      );
      close(d.shortfall(target), c.draws.shortfall);
      close(d.atK(c.k + 1).moment(2), c.draws.at_k);
      close(p.ci(0.95, { method: "normal" })[0], c.mean);
      close(
        p.linearCi(weights, 0.95, { method: "normal" }).slice(0, 2),
        c.linear,
      );
      close(
        p.momentCi(2, 0.95, { method: "normal" }).slice(0, 2),
        c.moments.find((m) => m.lam === 2)!.result,
      );
    });
  });
  for (const c of fixtures.weights)
    it(c.name + JSON.stringify(c.args), () => {
      const fn = (
        w as unknown as Record<string, (...args: unknown[]) => number[]>
      )[c.name]!;
      close(fn(...c.args), c.expected);
    });
  it("posterior draws agree with exact joint moments and reuse latent draws", () => {
    const p = tailpass(
      [
        [0, 1, 1],
        [1, 1, 1],
      ],
      4,
    );
    const draws = p.sample(12000, { rng: 42 });
    const summary = draws.summary();
    close(summary[0], p.mean, 0.006);
    close(summary[1], p.std, 0.006);
    expect(p.sample(8, { rng: 42 }).profile).toEqual(
      p.sample(8, { rng: 42 }).profile,
    );
    close(draws.atK(6).moment(1), draws.moment(1));
    close(draws.powerMean(1), draws.linear(w.uniformWeights(4)));
    expect(
      draws
        .qrs()
        .some(
          (v, i) =>
            Math.abs(v - draws.qrs(undefined, { aggregation: "profile" })[i]!) >
            1e-3,
        ),
    ).toBe(true);
  });
  it("avoids enumeration for large categorical budgets and preserves endpoint uncertainty", () => {
    const p = tailpass([0, 1, 2], 100000, [0, 0.2, 1], null, {
      thresholds: [0.5],
    });
    for (const lam of [1, 2, 4])
      expect(p.moment(lam).every(Number.isFinite)).toBe(true);
    expect(() => p.mean).toThrow(/count states/);
    expect(
      () => tailpass([0, 1], 3000, null, null, { thresholds: [1] }).covariance,
    ).toThrow(/covariance/);
    for (const prior of [
      [1e20, 1],
      [1, 1e20],
    ]) {
      const q = tailpass([0, 1], 4, null, null, { prior });
      expect(q.std[prior[0]! > prior[1]! ? 0 : 3]).toBeGreaterThan(0);
      expect(q.discovery()[1]).toBeGreaterThan(0);
      expect(q.stability()[1]).toBeGreaterThan(0);
    }
  });
  it("rejects invalid inputs and returns defensive profile copies", () => {
    for (const k of [0, 1.5, NaN, Infinity])
      expect(() => tailpass([0, 1], k)).toThrow();
    for (const R of [[], [[0.5, 1]], [[NaN, 1]], [[0, 2]]])
      expect(() => tailpass(R, 2)).toThrow();
    for (const opts of [
      { eta: -1 },
      { prior: 0 },
      { prior: [1, -1] },
      { thresholds: [0.5, 0.5] },
      { thresholds: [1.1] },
    ])
      expect(() => tailpass([0, 1], 2, null, null, opts)).toThrow();
    const p = tailpass([0, 1], 2),
      mean = p.mean;
    mean[0] = -1;
    expect(p.mean[0]).toBeGreaterThan(0);
    for (const weights of [[1], [-1, 2], [0.2, 0.2], [NaN, 1]])
      expect(() => p.linear(weights)).toThrow();
    expect(() => p.sample(1)).toThrow();
    expect(() => p.moment(0)).toThrow();
    const d = p.sample(8, { rng: 1 });
    expect(() => d.rollout(1)).toThrow();
    expect(() => d.powerMean(0)).toThrow();
    expect(() => d.shortfall([0.1, 0.5])).toThrow();
    expect(() => d.summary([1, 2])).toThrow();
    expect(() => tailpassEmpirical([0, 1], 3)).toThrow();
  });
});
