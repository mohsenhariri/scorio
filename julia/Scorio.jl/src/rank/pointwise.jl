"""Pointwise ranking methods."""

"""
    inverse_difficulty(
        R;
        method="competition",
        return_scores=false,
        clip_range=(0.01, 0.99),
    )

Rank models by question accuracy weighted by inverse empirical question
difficulty.

Each question weight is proportional to `1 / p_correct(question)`, after
clipping `p_correct` to `clip_range` and normalizing weights to sum to 1.

Let ``k_{lm} = \\sum_{n=1}^{N} R_{lmn}`` and
``\\hat p_{lm} = k_{lm}/N``. Define the global per-question solve rate
``\\bar p_m = \\frac{1}{L}\\sum_l \\hat p_{lm}`` and weights:

```math
w_m \\propto \\frac{1}{\\operatorname{clip}(\\bar p_m, a, b)},
\\qquad \\sum_{m=1}^{M} w_m = 1
```

The model score is:

```math
s_l^{\\mathrm{inv\\text{-}diff}} = \\sum_{m=1}^{M} w_m \\hat p_{lm}
```

# Reference
Inverse probability weighting:
https://en.wikipedia.org/wiki/Inverse_probability_weighting
"""
function inverse_difficulty(
    R;
    method="competition",
    return_scores=false,
    clip_range=(0.01, 0.99),
)
    Rv = validate_input(R)
    L, M, N = size(Rv)

    clip_len = try
        length(clip_range)
    catch
        error("clip_range must be a length-2 tuple (low, high).")
    end

    if clip_len != 2
        error("clip_range must be a length-2 tuple (low, high).")
    end

    low = Float64(clip_range[1])
    high = Float64(clip_range[2])
    if !isfinite(low) || !isfinite(high)
        error("clip_range values must be finite.")
    end
    if !(0.0 < low < high <= 1.0)
        error("clip_range must satisfy 0 < low < high <= 1.")
    end

    question_difficulty = zeros(Float64, M)
    for m in 1:M
        s = 0.0
        for l in 1:L, n in 1:N
            s += Rv[l, m, n]
        end
        question_difficulty[m] = s / (L * N)
    end
    question_difficulty = clamp.(question_difficulty, low, high)

    weights = 1.0 ./ question_difficulty
    total_weight = sum(weights)
    if !isfinite(total_weight) || total_weight <= 0.0
        error("inverse-difficulty weights are not finite; choose a stricter clip_range.")
    end
    weights ./= total_weight

    scores = zeros(Float64, L)
    for l in 1:L
        score = 0.0
        for m in 1:M
            acc = sum(@view Rv[l, m, :]) / N
            score += acc * weights[m]
        end
        scores[l] = score
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end
