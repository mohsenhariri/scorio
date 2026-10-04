import { describe, expect, it } from "vitest";

import { gammaln } from "../src/eval/internal/math.js";
import { asMatrix } from "../src/eval/internal/validate.js";
import { passAtK } from "../src/eval/passAtK.js";
import { bayes, avg } from "../src/eval/index.js";
import { normalCredibleInterval } from "../src/eval/internal/ci.js";
import { maxAtKCi } from "../src/eval/maxReward.js";

describe("review fixes", () => {
  it("rejects fractional and non-finite outcomes before conversion", () => {
    for (const value of [0.8, NaN, Infinity, -Infinity]) {
      expect(() => asMatrix([[value, 1]])).toThrow();
      expect(() => passAtK([[0, value, 1]], 1)).toThrow();
    }
  });

  it("toInt still accepts integer-valued floats", () => {
    expect(asMatrix([[0.0, 1.0]])).toEqual([[0, 1]]);
    // Integer-valued floats compute identically to plain integers.
    expect(passAtK([[0.0, 1.0, 1.0]], 1)).toBe(passAtK([[0, 1, 1]], 1));
  });

  it("gammaln of a negative non-integer matches scipy (not NaN)", () => {
    // scipy.special.gammaln(-0.5) = 1.2655121234846454
    expect(gammaln(-0.5)).toBeCloseTo(1.2655121234846454, 10);
    expect(Number.isNaN(gammaln(-0.5))).toBe(false);
  });

  it("rejects non-integer posterior budgets", () => {
    expect(() => maxAtKCi([[0, 1, 2]], 2.5, [0, 0.5, 1])).toThrow();
  });
  it("keeps categorical uncertainty finite under large scales and translations", () => {
    const R = [
        [0, 1, 2],
        [1, 2, 2],
      ],
      weights = [0, 0.5, 1];
    for (const score of [
      bayes,
      avg,
      (r: number[][], w: number[]) => maxAtKCi(r, 2, w).slice(0, 2),
    ]) {
      const base = score(R, weights),
        scaled = score(
          R,
          weights.map((x) => x * 1e155),
        );
      scaled.forEach((v, i) => expect(v / 1e155).toBeCloseTo(base[i]!, 12));
    }
    const base = maxAtKCi(R, 2, weights),
      shifted = maxAtKCi(
        R,
        2,
        weights.map((x) => x + 1e12),
      );
    expect(shifted[1]).toBeCloseTo(base[1], 12);
    expect(normalCredibleInterval(2, 0.1, 0.95, true, [0, 1])).toEqual([1, 1]);
    expect(normalCredibleInterval(-2, 0.1, 0.95, true, [0, 1])).toEqual([0, 0]);
  });
});
