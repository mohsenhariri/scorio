import { describe, expect, it } from "vitest";
import * as sinf from "../src/sinf/index.js";
import fixtures from "./fixtures/sinf.json";
const fx = fixtures;
function close(a: number, b: unknown, label: string): void {
  const expected = b === "inf" ? Infinity : (b as number);
  if (Number.isFinite(expected)) expect(a, label).toBeCloseTo(expected, 8);
  else expect(a, label).toBe(expected);
}
describe("fixed-look legacy API", () => {
  const L = fx.legacy;
  it("rankingConfidence (+ degenerate branches)", () => {
    const rc = sinf.rankingConfidence(
      L.mus[0]!,
      L.sigmas[0]!,
      L.mus[1]!,
      L.sigmas[1]!,
    );
    close(rc.rho, L.ranking_confidence.rho, "rc/rho");
    close(rc.z, L.ranking_confidence.z, "rc/z");

    const tie = sinf.rankingConfidence(0.8, 0, 0.8, 0);
    expect(tie).toEqual({ rho: 0.5, z: Infinity });
    const certain = sinf.rankingConfidence(0.8, 0, 0.7, 0);
    close(certain.rho, L.ranking_conf_certain.rho, "certain/rho");
    close(certain.z, L.ranking_conf_certain.z, "certain/z"); // Infinity
  });
  it("ciFromMuSigma (+ clip) / shouldStop", () => {
    const ci = sinf.ciFromMuSigma(0.7, 0.05, { confidence: 0.9 });
    close(ci.lo, L.ci_90.lo, "ci/lo");
    close(ci.hi, L.ci_90.hi, "ci/hi");
    const cic = sinf.ciFromMuSigma(0.97, 0.05, {
      confidence: 0.9,
      clip: [0, 1],
    });
    close(cic.lo, L.ci_90_clip.lo, "ciClip/lo");
    close(cic.hi, L.ci_90_clip.hi, "ciClip/hi");

    expect(
      sinf.shouldStop(0.01, { confidence: 0.95, maxHalfWidth: 0.02 }),
    ).toBe(L.should_stop_half);
    expect(
      sinf.shouldStop(0.02, { confidence: 0.95, maxHalfWidth: 0.02 }),
    ).toBe(L.should_stop_half_false);
    expect(sinf.shouldStop(0.02, { confidence: 0.95, maxCiWidth: 0.1 })).toBe(
      L.should_stop_ci,
    );
  });
  it("shouldStopTop1 / suggestNextAllocation (ci_overlap + zscore)", () => {
    const ci = sinf.shouldStopTop1(L.mus, L.sigmas, { method: "ci_overlap" });
    expect(ci.stop).toBe(L.top1_ci_overlap.stop);
    expect(ci.leader).toBe(L.top1_ci_overlap.leader);
    expect(ci.ambiguous).toEqual(L.top1_ci_overlap.ambiguous);
    const zs = sinf.shouldStopTop1(L.mus, L.sigmas, { method: "zscore" });
    expect(zs.stop).toBe(L.top1_zscore.stop);
    expect(zs.leader).toBe(L.top1_zscore.leader);
    expect(zs.ambiguous).toEqual(L.top1_zscore.ambiguous);

    const aci = sinf.suggestNextAllocation(L.mus, L.sigmas, {
      method: "ci_overlap",
    });
    expect([aci.leader, aci.competitor]).toEqual(L.alloc_ci_overlap);
    const azs = sinf.suggestNextAllocation(L.mus, L.sigmas, {
      method: "zscore",
    });
    expect([azs.leader, azs.competitor]).toEqual(L.alloc_zscore);
  });
});
