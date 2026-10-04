"""Regenerate deterministic Python reference values consumed by both ports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scorio import eval
from scorio.eval import tailpass_weights as weights
from scorio.eval.tailpass import TailPassDraws


def main():
    cases = []
    settings = [
        ([[0, 1, 1], [1, 1, 1]], 4, [0, 1], None, 1, 1, None),
        (
            [[0, 1, 1], [1, 0, 1]],
            3,
            [0, 1],
            [[1, 0], [1, 1]],
            0.3,
            [[2, 0.5], [0.5, 3]],
            [0, 0.28, 2 / 3, 1],
        ),
        (
            [[0, 1, 2, 1], [2, 2, 0, 1]],
            3,
            [0, 0.2, 1],
            [[2], [0]],
            0.7,
            [0.5, 1.2, 3],
            [0, 0.1, 0.2, 0.5, 1],
        ),
        ([[0, 1, 2], [1, 2, 2]], 2, [0, 0, 1], None, 1, 1, None),
        ([[0, 1, 2]], 3, [0.1, 0.2, 0.3], None, 1, 1, [0.1, 0.2, 0.3, 1]),
        ([[0, 1, 0]], 3, [0.5, 0.5], None, 1, 1, [0, 0.5, 1]),
        ([[0, 0, 0]], 2, [0], None, 1, 1, None),
        ([[0, 0, 0]], 2, [1], None, 1, 1, None),
    ]
    settings.append(
        (
            [[0, 1, 2]],
            3,
            [0.1, 0.2, 0.3],
            None,
            1,
            1,
            [float(np.nextafter(0.1, 1)), 0.3],
        )
    )
    for R, k, w, R0, eta, prior, thresholds in settings:
        profile = eval.tailpass(
            R, k, w, R0, eta=eta, prior=prior, thresholds=thresholds
        )
        options = dict(eta=eta, prior=prior, thresholds=profile.thresholds.tolist())
        linear_weights = weights.uniform_weights(len(profile.thresholds))
        item = dict(
            R=R,
            k=k,
            w=w,
            R0=R0,
            options=options,
            mean=profile.mean.tolist(),
            std=profile.std.tolist(),
            question_mean=profile.question_mean.tolist(),
            covariance=profile.covariance.tolist(),
            linear=profile.linear(linear_weights),
            discovery=profile.discovery(),
            stability=profile.stability(),
            moments=[
                dict(lam=lam, result=profile.moment(lam)) for lam in [0.5, 1, 2, 4]
            ],
            empirical=eval.tailpass_empirical(
                R, min(k, len(R[0])), w, thresholds=profile.thresholds
            ).tolist(),
        )
        levels = len(profile._scores)
        generator = np.random.default_rng(12)
        probs = generator.dirichlet(np.ones(levels), size=(4, len(R)))
        draws = TailPassDraws(profile, probs)
        item["draws"] = dict(
            probabilities=probs.tolist(),
            profile=draws.profile.tolist(),
            linear=draws.linear(linear_weights).tolist(),
            moment=draws.moment(2).tolist(),
            qrs=draws.qrs().tolist(),
            qrs_profile=draws.qrs(aggregation="profile").tolist(),
            rollout=draws.rollout(3).tolist(),
            harmonic=draws.harmonic().tolist(),
            shortfall=draws.shortfall(
                np.linspace(1, 0.5, len(profile.thresholds))
            ).tolist(),
            summary=[x.tolist() for x in draws.summary()],
            at_k=draws.at_k(k + 1).moment(2).tolist(),
        )
        cases.append(item)
    weight_cases = []
    for name, args in [
        ("uniform_weights", [7]),
        ("threshold_weights", [5, 3]),
        ("discovery_weights", [4]),
        ("stability_weights", [4]),
        ("moment_weights", [7, 0.01]),
        ("moment_weights", [7, 4]),
        ("beta_weights", [7, 0.8, 5]),
        ("beta_weights", [7, 0.1, 0.1]),
        ("maxent_weights", [7, 0.15]),
        ("maxent_weights", [7, 0.85]),
        ("maxent_weights", [7, 0.5]),
        ("payoff_weights", [[0, 0.1, 0.6, 1]]),
    ]:
        weight_cases.append(
            dict(name=name, args=args, expected=getattr(weights, name)(*args).tolist())
        )
    root = Path(__file__).resolve().parents[2]
    content = (
        json.dumps(dict(cases=cases, weights=weight_cases), indent=2, allow_nan=False)
        + "\n"
    )
    for relative in (
        "js/scorio/test/fixtures/tailpass.json",
        "julia/Scorio.jl/test/fixtures/tailpass.json",
    ):
        (root / relative).write_text(content)


if __name__ == "__main__":
    main()
