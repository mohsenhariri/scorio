import { describe, expect, it } from "vitest";

import {
  gPassAtK,
  gPassAtKTau,
  mgPassAtK,
  gPassAtKCi,
  gPassAtKTauCi,
  mgPassAtKCi,
} from "../src/eval/gpass.js";

const round = (x: number, d: number) => Number(x.toFixed(d));

const R = [
  [0, 1, 1, 0, 1],
  [1, 1, 0, 1, 1],
];

describe("gPassAtK", () => {
  it("matches doctest values (alias of pass^k... actually pass_hat_k)", () => {
    // Doctest: round(g_pass_at_k(R, 1), 6) -> 0.7
    expect(round(gPassAtK(R, 1), 6)).toBe(0.7);
    // Doctest: round(g_pass_at_k(R, 2), 6) -> 0.45
    expect(round(gPassAtK(R, 2), 6)).toBe(0.45);
  });
});

describe("gPassAtKTau", () => {
  it("matches doctest values", () => {
    // Doctest: round(g_pass_at_k_tau(R, 2, 0.5), 6) -> 0.95
    expect(round(gPassAtKTau(R, 2, 0.5), 6)).toBe(0.95);
    // Doctest: round(g_pass_at_k_tau(R, 2, 1.0), 6) -> 0.45
    expect(round(gPassAtKTau(R, 2, 1.0), 6)).toBe(0.45);
  });

  it("matches Python reference for extra cases", () => {
    // python: eval.g_pass_at_k_tau(R, 3, 0.5) -> 0.85
    expect(round(gPassAtKTau(R, 3, 0.5), 6)).toBe(0.85);
  });
});

describe("mgPassAtK", () => {
  it("matches doctest values", () => {
    // Doctest: round(mg_pass_at_k(R, 2), 6) -> 0.45
    expect(round(mgPassAtK(R, 2), 6)).toBe(0.45);
    // Doctest: round(mg_pass_at_k(R, 3), 6) -> 0.166667
    expect(round(mgPassAtK(R, 3), 6)).toBe(0.166667);
  });
});

describe("gPassAtKCi", () => {
  it("matches Python reference values", () => {
    // python: eval.g_pass_at_k_ci(R, 2)
    const [mu, sigma, lo, hi] = gPassAtKCi(R, 2);
    expect(round(mu, 6)).toBe(0.446429);
    expect(round(sigma, 6)).toBe(0.146167);
    expect(round(lo, 6)).toBe(0.159946);
    expect(round(hi, 6)).toBe(0.732911);
  });
});

describe("gPassAtKTauCi", () => {
  it("matches Python reference values", () => {
    // python: eval.g_pass_at_k_tau_ci(R, 2, 0.5)
    const [mu, sigma, lo, hi] = gPassAtKTauCi(R, 2, 0.5);
    expect(round(mu, 6)).toBe(0.839286);
    expect(round(sigma, 6)).toBe(0.097263);
    expect(round(lo, 6)).toBe(0.648654);
    expect(round(hi, 6)).toBe(1.0);
  });

  it("matches Python reference for an interior tau (k=3)", () => {
    // python: eval.g_pass_at_k_tau_ci(R, 3, 0.5)
    const [mu, sigma, lo, hi] = gPassAtKTauCi(R, 3, 0.5);
    expect(round(mu, 6)).toBe(0.684524);
    expect(round(sigma, 6)).toBe(0.151958);
    expect(round(lo, 6)).toBe(0.386692);
    expect(round(hi, 6)).toBe(0.982356);
  });
});

describe("mgPassAtKCi", () => {
  it("matches Python reference values", () => {
    // python: eval.mg_pass_at_k_ci(R, 3)
    const [mu, sigma, lo, hi] = mgPassAtKCi(R, 3);
    expect(round(mu, 6)).toBe(0.218254);
    expect(round(sigma, 6)).toBe(0.098816);
    expect(round(lo, 6)).toBe(0.024578);
    expect(round(hi, 6)).toBe(0.41193);
  });
});

describe("tau threshold when tau * k is a whole number", () => {
  // (j, k) pairs where (j / k) * k rounds above j in IEEE doubles.
  const cases: [number, number][] = [
    [7, 25],
    [14, 25],
    [15, 29],
    [29, 35],
    [21, 38],
  ];
  const bank = (ones: number, n: number) => [
    [...Array(ones).fill(1), ...Array(n - ones).fill(0)],
  ];
  // Positive IEEE doubles have monotonically increasing bit representations.
  const adjacentFloat = (value: number, direction: -1 | 1): number => {
    const view = new DataView(new ArrayBuffer(8));
    view.setFloat64(0, value);
    view.setBigUint64(0, view.getBigUint64(0) + BigInt(direction));
    return view.getFloat64(0);
  };

  it.each(cases)("needs exactly %i of %i successes", (j, k) => {
    expect(gPassAtKTau(bank(j, k), k, j / k)).toBeCloseTo(1.0, 12);
    expect(gPassAtKTau(bank(j - 1, k), k, j / k)).toBeCloseTo(0.0, 12);
    const exact = gPassAtKTauCi(bank(j, k), k, j / k);
    const below = gPassAtKTauCi(bank(j, k), k, (j - 0.5) / k);
    exact.forEach((v, i) => expect(v).toBeCloseTo(below[i]!, 12));
  });

  it.each<[number, number]>([
    [1, 3],
    [7, 25],
    [15, 29],
    [7, 100],
    [29, 100],
    [24, 25],
  ])("preserves thresholds adjacent to %i / %i", (j, k) => {
    const R = bank(j, k);
    const boundary = j / k;
    const lowerCi = gPassAtKTauCi(R, k, (j - 0.5) / k);
    const upperCi = gPassAtKTauCi(R, k, (j + 0.5) / k);
    for (const tau of [adjacentFloat(boundary, -1), boundary]) {
      expect(gPassAtKTau(R, k, tau)).toBeCloseTo(1.0, 12);
      gPassAtKTauCi(R, k, tau).forEach((v, i) =>
        expect(v).toBeCloseTo(lowerCi[i]!, 12),
      );
    }
    for (const tau of [adjacentFloat(boundary, 1), boundary + 2e-11]) {
      expect(gPassAtKTau(R, k, tau)).toBeCloseTo(0.0, 12);
      gPassAtKTauCi(R, k, tau).forEach((v, i) =>
        expect(v).toBeCloseTo(upperCi[i]!, 12),
      );
    }
  });

  it("matches Python reference at tau = 0.28, k = 25", () => {
    // python: eval.g_pass_at_k_tau(R, 25, 0.28) -> 1.0 (7 of 25 correct)
    expect(gPassAtKTau(bank(7, 25), 25, 0.28)).toBeCloseTo(1.0, 12);
    // python: eval.g_pass_at_k_tau_ci(R, 25, 0.28)
    //   -> (0.5893754878280344, 0.2781303668356882, 0.0442499858..., 1.0)
    const [mu, sigma] = gPassAtKTauCi(bank(7, 25), 25, 0.28);
    expect(round(mu, 6)).toBe(0.589375);
    expect(round(sigma, 6)).toBe(0.27813);
  });
});
