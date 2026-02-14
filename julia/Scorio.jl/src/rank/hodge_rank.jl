"""HodgeRank method scaffold."""

using LinearAlgebra

function _pairwise_flow_binary(
    wins::AbstractMatrix{<:Real},
    ties::AbstractMatrix{<:Real},
)::Matrix{Float64}
    total = wins .+ transpose(wins) .+ ties
    Y = zeros(Float64, size(wins, 1), size(wins, 2))
    mask = total .> 0.0
    Y[mask] .= (transpose(wins)[mask] .- wins[mask]) ./ total[mask]
    for i in 1:size(Y, 1)
        Y[i, i] = 0.0
    end
    return Y
end

function _pairwise_flow_log_odds(
    wins::AbstractMatrix{<:Real},
    ties::AbstractMatrix{<:Real};
    epsilon=0.5,
)::Matrix{Float64}
    epsilon_f = Float64(epsilon)
    if !isfinite(epsilon_f) || epsilon_f <= 0.0
        error("epsilon must be > 0 for log-odds smoothing")
    end

    total = wins .+ transpose(wins) .+ ties
    Y = zeros(Float64, size(wins, 1), size(wins, 2))
    mask = total .> 0.0

    numerator = (total .- wins .+ epsilon_f)[mask]
    denom = (total .- transpose(wins) .+ epsilon_f)[mask]
    Y[mask] .= log.(numerator ./ denom)
    for i in 1:size(Y, 1)
        Y[i, i] = 0.0
    end
    return Y
end

function _weights_from_counts(
    wins::AbstractMatrix{<:Real},
    ties::AbstractMatrix{<:Real};
    weight_method="total",
)::Matrix{Float64}
    total = wins .+ transpose(wins) .+ ties
    method_s = string(weight_method)

    w = if method_s == "total"
        Float64.(total)
    elseif method_s == "decisive"
        Float64.(wins .+ transpose(wins))
    elseif method_s == "uniform"
        Float64.(total .> 0.0)
    else
        error("weight_method must be one of: \"total\", \"decisive\", \"uniform\"")
    end

    for i in 1:size(w, 1)
        w[i, i] = 0.0
    end
    return w
end

function _laplacian_from_weights(w::AbstractMatrix{<:Real})::Matrix{Float64}
    L = -Float64.(w)
    for i in 1:size(L, 1)
        L[i, i] = 0.0
    end
    for i in 1:size(L, 1)
        L[i, i] = -sum(@view L[i, :])
    end
    return L
end

function _divergence(w::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real})::Vector{Float64}
    return vec(sum(Float64.(w) .* Float64.(Y); dims=2))
end

function _grad(scores::AbstractVector{<:Real})::Matrix{Float64}
    s = Float64.(scores)
    return reshape(s, 1, :) .- reshape(s, :, 1)
end

"""
    hodge_rank(
        R;
        pairwise_stat="binary",
        weight_method="total",
        epsilon=0.5,
        method="competition",
        return_scores=false,
        return_diagnostics=false,
    )

Rank models with l2 HodgeRank on a weighted pairwise-comparison graph.

Let `Y_{ij}` be a skew-symmetric observed pairwise flow and `w_{ij}\\ge 0`
edge weights. HodgeRank solves:

```math
s^\\star \\in \\arg\\min_s
\\sum_{i<j} w_{ij}\\left((s_j-s_i)-Y_{ij}\\right)^2
```

Equivalent normal equations:

```math
\\Delta_0 s^\\star = -\\operatorname{div}(Y),
\\qquad
s^\\star = -\\Delta_0^\\dagger \\operatorname{div}(Y)
```

where `\\Delta_0^\\dagger` is the Laplacian pseudoinverse.

# Reference
Jiang, X., Lim, L.-H., Yao, Y., & Ye, Y. (2009).
Statistical Ranking and Combinatorial Hodge Theory.
https://arxiv.org/abs/0811.1067
"""
function hodge_rank(
    R;
    pairwise_stat="binary",
    weight_method="total",
    epsilon=0.5,
    method="competition",
    return_scores=false,
    return_diagnostics=false,
)
    Rv = validate_input(R)
    L = size(Rv, 1)
    wins, ties = build_pairwise_counts(Rv)

    pairwise_stat_s = string(pairwise_stat)
    Y = if pairwise_stat_s == "binary"
        _pairwise_flow_binary(wins, ties)
    elseif pairwise_stat_s == "log_odds"
        _pairwise_flow_log_odds(wins, ties; epsilon=epsilon)
    else
        error("pairwise_stat must be one of: \"binary\", \"log_odds\"")
    end

    w = _weights_from_counts(wins, ties; weight_method=weight_method)
    if !any(w .> 0.0)
        scores = fill(1.0 / L, L)
        ranking = rank_scores(scores)[string(method)]
        if !return_scores
            return ranking
        end
        if !return_diagnostics
            return ranking, scores
        end
        return ranking, scores, Dict("residual_l2" => 0.0, "relative_residual_l2" => 0.0)
    end

    Lap = _laplacian_from_weights(w)
    div = _divergence(w, Y)
    scores = -pinv(Lap) * div

    ranking = rank_scores(scores)[string(method)]
    if !return_diagnostics && !return_scores
        return ranking
    end
    if !return_diagnostics && return_scores
        return ranking, scores
    end

    grad_s = _grad(scores)
    resid = Y .- grad_s

    w_half = Float64[]
    r_half = Float64[]
    y_half = Float64[]
    for i in 1:L
        for j in (i + 1):L
            if w[i, j] > 0.0
                push!(w_half, w[i, j])
                push!(r_half, resid[i, j])
                push!(y_half, Y[i, j])
            end
        end
    end

    residual_l2 = sqrt(sum(w_half .* (r_half .^ 2)))
    denom = sqrt(sum(w_half .* (y_half .^ 2)))
    rel = denom > 0.0 ? (residual_l2 / denom) : 0.0

    return ranking, scores, Dict("residual_l2" => residual_l2, "relative_residual_l2" => rel)
end
