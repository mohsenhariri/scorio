"""Eval-metric-based ranking methods."""

"""
    avg(R; method="competition", return_scores=false)

Rank models by per-model mean accuracy across all questions and trials.

For each model `l`, compute the scalar score:

```math
s_l^{\\mathrm{avg}} = \\frac{1}{MN}\\sum_{m=1}^{M}\\sum_{n=1}^{N} R_{lmn}
```

Higher scores are better; ranking is produced by `rank_scores`.

# Arguments
- `R`: binary response tensor `(L, M, N)` or matrix `(L, M)` promoted to `(L, M, 1)`.
- `method`: tie-handling rule for `rank_scores`.
- `return_scores`: if `true`, return `(ranking, scores)`.
"""
function avg(R; method="competition", return_scores=false)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        scores[model] = avg(@view Rv[model, :, :])
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

# Rational approximation of the inverse standard normal CDF.
function _norm_ppf(p::Float64)::Float64
    if p == 0.0
        return -Inf
    elseif p == 1.0
        return Inf
    end

    # Coefficients from Peter J. Acklam's inverse-normal approximation.
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    c1 = -0.00778489400243029
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    d1 = 0.00778469570904146
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow
        q = sqrt(-2.0 * log(p))
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return num / den
    elseif p > phigh
        q = sqrt(-2.0 * log(1.0 - p))
        num = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        den = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return -num / den
    else
        q = p - 0.5
        r = q * q
        num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return num / den
    end
end

"""
    bayes(
        R::AbstractArray{<:Integer, 3},
        w=nothing;
        R0=nothing,
        quantile=nothing,
        method="competition",
        return_scores=false,
    )

Rank models by Bayes@N scores computed independently per model.

If `quantile` is provided, models are ranked by `mu + z_q * sigma`; otherwise
by posterior mean `mu`.

# References
Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
*arXiv:2510.04265*. https://arxiv.org/abs/2510.04265

# Formula
For each model `l`, let `(mu_l, sigma_l) = Scorio.bayes(R_l, w, R0_l)`.

```math
s_l =
\\begin{cases}
\\mu_l, & \\text{if quantile is not set} \\\\
\\mu_l + \\Phi^{-1}(q)\\,\\sigma_l, & \\text{if quantile}=q \\in [0,1]
\\end{cases}
```

# Arguments
- `R`: integer tensor `(L, M, N)` with values in `{0, ..., C}`.
- `w`: class weights of length `C+1`. If not provided and R is binary (contains only 0 and 1),
  defaults to `[0.0, 1.0]`. For non-binary R, w is required.
- `R0`: optional shared prior `(M, D)` or model-specific prior `(L, M, D)`.
- `quantile`: optional value in `[0, 1]` for quantile-adjusted ranking.
- `method`, `return_scores`: ranking output controls.
"""
function bayes(
    R::AbstractArray{<:Integer, 3},
    w=nothing;
    R0=nothing,
    quantile=nothing,
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R; binary_only=false)
    L, M, _ = size(Rv)

    z = nothing
    if !isnothing(quantile)
        q = Float64(quantile)
        if !(0.0 <= q <= 1.0)
            error("quantile must be in [0, 1]; got $quantile")
        end
        z = _norm_ppf(q)
    end

    R0_shared = nothing
    R0_per_model = nothing

    if !isnothing(R0)
        R0_arr = Int.(Array(R0))

        if ndims(R0_arr) == 2
            if size(R0_arr, 1) != M
                error("Shared R0 must have shape (M=$M, D), got $(size(R0_arr))")
            end
            R0_shared = R0_arr
        elseif ndims(R0_arr) == 3
            if size(R0_arr, 1) != L || size(R0_arr, 2) != M
                error(
                    "Model-specific R0 must have shape (L=$L, M=$M, D), got $(size(R0_arr))",
                )
            end
            R0_per_model = R0_arr
        else
            error(
                "R0 must be shape (M, D) or (L, M, D); got ndim=$(ndims(R0_arr)) with shape $(size(R0_arr))",
            )
        end
    end

    scores = zeros(Float64, L)
    for model in 1:L
        model_R0 = isnothing(R0_shared) ? nothing : R0_shared
        if !isnothing(R0_per_model)
            model_R0 = @view R0_per_model[model, :, :]
        end

        mu, sigma = bayes(@view(Rv[model, :, :]), w, model_R0)
        scores[model] = isnothing(z) ? mu : (mu + z * sigma)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    pass_at_k(R::AbstractArray{<:Integer, 3}, k; method="competition", return_scores=false)

Rank models by per-model Pass@k scores.

For each model `l`, define per-question success counts
`nu_{lm} = \\sum_{n=1}^{N} R_{lmn}`. Then:

```math
s_l^{\\mathrm{Pass@}k}
= \\frac{1}{M}\\sum_{m=1}^{M}
\\left(1 - \\frac{\\binom{N-\\nu_{lm}}{k}}{\\binom{N}{k}}\\right)
```

# References
Chen, M., Tworek, J., Jun, H., et al. (2021).
Evaluating Large Language Models Trained on Code.
*arXiv:2107.03374*. https://arxiv.org/abs/2107.03374
"""
function pass_at_k(
    R::AbstractArray{<:Integer, 3},
    k;
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        scores[model] = pass_at_k(@view(Rv[model, :, :]), k)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    pass_hat_k(R::AbstractArray{<:Integer, 3}, k; method="competition", return_scores=false)

Rank models by per-model Pass-hat@k (G-Pass@k) scores.

With `nu_{lm} = \\sum_{n=1}^{N} R_{lmn}`:

```math
s_l^{\\widehat{\\mathrm{Pass@}k}}
= \\frac{1}{M}\\sum_{m=1}^{M}
\\frac{\\binom{\\nu_{lm}}{k}}{\\binom{N}{k}}
```

# References
Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
*arXiv:2406.12045*. https://arxiv.org/abs/2406.12045
"""
function pass_hat_k(
    R::AbstractArray{<:Integer, 3},
    k;
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        scores[model] = pass_hat_k(@view(Rv[model, :, :]), k)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    g_pass_at_k_tau(
        R::AbstractArray{<:Integer, 3},
        k,
        tau;
        method="competition",
        return_scores=false,
)

Rank models by generalized G-Pass@k_τ per model.

Let `X_{lm} ~ Hypergeometric(N, nu_{lm}, k)` where
`nu_{lm} = \\sum_{n=1}^{N} R_{lmn}`. The score is:

```math
s_l^{\\mathrm{G\\text{-}Pass@}k_{\\tau}}
= \\frac{1}{M}\\sum_{m=1}^{M}
\\Pr\\!\\left(X_{lm}\\ge \\lceil \\tau k \\rceil\\right)
```

```math
\\Pr(X_{lm}\\ge \\lceil \\tau k \\rceil)
= \\sum_{j=\\lceil \\tau k \\rceil}^{k}
\\frac{\\binom{\\nu_{lm}}{j}\\binom{N-\\nu_{lm}}{k-j}}{\\binom{N}{k}}
```

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv:2412.13147*. https://arxiv.org/abs/2412.13147
"""
function g_pass_at_k_tau(
    R::AbstractArray{<:Integer, 3},
    k,
    tau;
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        scores[model] = g_pass_at_k_tau(@view(Rv[model, :, :]), k, tau)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end

"""
    mg_pass_at_k(R::AbstractArray{<:Integer, 3}, k; method="competition", return_scores=false)

Rank models by per-model mG-Pass@k scores.

With `X_{lm} ~ Hypergeometric(N, nu_{lm}, k)` and `m_0 = \\lceil k/2 \\rceil`:

```math
s_l^{\\mathrm{mG\\text{-}Pass@}k}
= \\frac{1}{M}\\sum_{m=1}^{M}
\\frac{2}{k}\\,\\mathbb{E}\\!\\left[(X_{lm}-m_0)_+\\right]
```

Equivalent discrete form:

```math
\\frac{2}{k}\\sum_{i=m_0+1}^{k}\\Pr(X_{lm}\\ge i)
```

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv:2412.13147*. https://arxiv.org/abs/2412.13147
"""
function mg_pass_at_k(
    R::AbstractArray{<:Integer, 3},
    k;
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)
    L = size(Rv, 1)

    scores = zeros(Float64, L)
    for model in 1:L
        scores[model] = mg_pass_at_k(@view(Rv[model, :, :]), k)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end
