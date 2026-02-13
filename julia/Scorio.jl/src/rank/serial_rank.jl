"""SerialRank method scaffold."""

using LinearAlgebra

function _comparison_matrix_from_counts(
    wins::AbstractMatrix{<:Real},
    ties::AbstractMatrix{<:Real};
    comparison="prob_diff",
)::Matrix{Float64}
    comparison_s = string(comparison)

    if comparison_s ∈ ("prob_diff", "fractional")
        total = wins .+ transpose(wins) .+ ties
        C = zeros(Float64, size(wins, 1), size(wins, 2))
        mask = total .> 0.0
        C[mask] .= (wins[mask] .- transpose(wins)[mask]) ./ total[mask]
        for i in 1:size(C, 1)
            C[i, i] = 0.0
        end
        return C
    end

    if comparison_s ∈ ("sign", "majority")
        diff = wins .- transpose(wins)
        C = Float64.(sign.(diff))
        for i in 1:size(C, 1)
            C[i, i] = 0.0
        end
        return C
    end

    error("comparison must be \"prob_diff\" or \"sign\"")
end

function _serialrank_similarity(C::AbstractMatrix{<:Real})::Matrix{Float64}
    n = size(C, 1)
    return 0.5 .* (n .* ones(Float64, n, n) .+ (Float64.(C) * transpose(Float64.(C))))
end

function _laplacian(S::AbstractMatrix{<:Real})::Matrix{Float64}
    Sf = Float64.(S)
    d = vec(sum(Sf; dims=2))
    return Diagonal(d) .- Sf
end

function _fiedler_vector(L::AbstractMatrix{<:Real})
    ef = eigen(Symmetric(Float64.(L)))
    w = ef.values
    V = ef.vectors

    if size(V, 2) < 2
        return ones(Float64, size(L, 1)), false
    end

    v = V[:, 2]
    if size(V, 2) == 2
        return v, true
    end

    scale = max(1.0, maximum(abs.(w)))
    eigengap = w[3] - w[2]
    unique = isfinite(eigengap) && eigengap > 1e-10 * scale
    return v, unique
end

function _orientation_key(scores::AbstractVector{<:Real}, C::AbstractMatrix{<:Real})
    s = Float64.(scores)
    Cf = Float64.(C)
    n = length(s)

    upsets = 0
    weighted_upsets = 0.0
    corr = 0.0
    nonzero_seen = false

    for i in 1:n
        for j in (i + 1):n
            cij = Cf[i, j]
            if cij == 0.0
                continue
            end
            nonzero_seen = true

            pred = sign(s[i] - s[j])
            disagree = (pred == 0.0) || ((pred * cij) < 0.0)
            if disagree
                upsets += 1
                weighted_upsets += abs(cij)
            end
            corr += pred * cij
        end
    end

    if !nonzero_seen
        return (0, 0.0, 0.0)
    end
    return (upsets, weighted_upsets, -corr)
end

function _mean_accuracy_scores(Rv::AbstractArray{<:Integer,3})::Vector{Float64}
    L, M, N = size(Rv)
    denom = Float64(M * N)
    return vec(sum(Rv; dims=(2, 3))) ./ denom
end

"""
    serial_rank(R; comparison="prob_diff", method="competition", return_scores=false)

Rank models with SerialRank spectral seriation using a Fiedler-vector ordering
from comparison-induced similarity.

With pairwise comparison matrix `C` (skew-symmetric), SerialRank builds:

```math
S = \\frac{1}{2}\\left(L\\mathbf{1}\\mathbf{1}^{\\top} + C C^{\\top}\\right),
\\qquad
L_S = \\operatorname{diag}(S\\mathbf{1}) - S
```

Scores are the oriented Fiedler vector of `L_S` (eigenvector of the
second-smallest eigenvalue), with sign chosen to best match observed
pairwise directions.

# Reference
Fogel, F., d'Aspremont, A., & Vojnovic, M. (2016).
Spectral Ranking Using Seriation. *JMLR*.
"""
function serial_rank(
    R;
    comparison="prob_diff",
    method="competition",
    return_scores=false,
)
    Rv = validate_input(R)

    wins, ties = build_pairwise_counts(Rv)
    C = _comparison_matrix_from_counts(wins, ties; comparison=comparison)
    S = _serialrank_similarity(C)
    Ls = _laplacian(S)

    v, is_unique = _fiedler_vector(Ls)
    if !is_unique || any(x -> !isfinite(x), v) || all(abs.(v .- v[1]) .<= 1e-12)
        scores = _mean_accuracy_scores(Rv)
        ranking = rank_scores(scores)[string(method)]
        return return_scores ? (ranking, scores) : ranking
    end

    key_pos = _orientation_key(v, C)
    key_neg = _orientation_key(-v, C)
    scores = key_pos <= key_neg ? Float64.(v) : Float64.(-v)

    mean_s = sum(scores) / length(scores)
    variance = sum((scores .- mean_s) .^ 2) / length(scores)
    if sqrt(max(variance, 0.0)) < 1e-12
        scores = _mean_accuracy_scores(Rv)
    end

    ranking = rank_scores(scores)[string(method)]
    return return_scores ? (ranking, scores) : ranking
end
