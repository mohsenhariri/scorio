"""
    competition_ranks_from_scores(scores_in_id_order::AbstractVector{<:Real}; tol::Real=1e-12) -> Vector{Int}

Compute competition ranks from scores.

Given L models with ids 1..L and their scores, returns competition ranks (1,2,3,3,5,...).
Models with tied scores (within tolerance) receive the same rank.

# Arguments
- `scores_in_id_order::AbstractVector{<:Real}`: list/array of scores aligned to ids 1..L
- `tol::Real=1e-12`: tolerance for considering scores as tied

# Returns
- `Vector{Int}`: competition ranks for each model

# Examples
```julia
scores = [0.95, 0.87, 0.87, 0.72, 0.65]
ranks = competition_ranks_from_scores(scores)
# Returns: [1, 2, 2, 4, 5]
```
"""
function competition_ranks_from_scores(
    scores_in_id_order::AbstractVector{<:Real};
    tol::Real=1e-12
)::Vector{Int}
    
    scores = Float64.(scores_in_id_order)
    n = length(scores)
    
    # Get indices that would sort scores in descending order
    order = sortperm(scores, rev=true)
    
    ranks = zeros(Int, n)
    rank = 1
    i = 1
    
    while i <= n
        # Find tie block
        j = i
        while j + 1 <= n && abs(scores[order[j + 1]] - scores[order[i]]) <= tol
            j += 1
        end
        
        # Assign same rank to the whole block
        for t in i:j
            ranks[order[t]] = rank
        end
        
        # Next rank skips by block size
        rank += j - i + 1
        i = j + 1
    end
    
    return ranks
end

"""
    rank_scores(scores_in_id_order::AbstractVector{<:Real}; tol::Real=1e-12) -> Dict{String, Vector{Float64}}

Convert scores to ranks using multiple tie-handling methods.

Scores are sorted in descending order, near-equal values (within `tol`) are grouped,
and ranks are returned in original id order for:

- `"competition"`: min-rank competition (1,2,2,4,...)
- `"competition_max"`: max-rank competition (1,3,3,4,...)
- `"dense"`: dense ranking (1,2,2,3,...)
- `"avg"`: average/fractional ranking (1,2.5,2.5,4,...)
"""
function rank_scores(
    scores_in_id_order::AbstractVector{<:Real};
    tol::Real=1e-12
)::Dict{String, Vector{Float64}}

    scores = Float64.(scores_in_id_order)
    n = length(scores)

    order = sortperm(scores, rev=true)
    sorted_scores = scores[order]

    # Group near-equal scores
    grouped_scores = copy(sorted_scores)
    for i in 2:n
        if abs(grouped_scores[i] - grouped_scores[i - 1]) <= tol
            grouped_scores[i] = grouped_scores[i - 1]
        end
    end

    competition_sorted = zeros(Float64, n)
    competition_max_sorted = zeros(Float64, n)
    dense_sorted = zeros(Float64, n)
    avg_sorted = zeros(Float64, n)

    i = 1
    dense_rank = 1
    while i <= n
        j = i
        while j + 1 <= n && grouped_scores[j + 1] == grouped_scores[i]
            j += 1
        end

        min_rank = Float64(i)
        max_rank = Float64(j)
        avg_rank = (min_rank + max_rank) / 2.0
        dense_rank_float = Float64(dense_rank)

        for t in i:j
            competition_sorted[t] = min_rank
            competition_max_sorted[t] = max_rank
            dense_sorted[t] = dense_rank_float
            avg_sorted[t] = avg_rank
        end

        dense_rank += 1
        i = j + 1
    end

    competition = similar(competition_sorted)
    competition_max = similar(competition_max_sorted)
    dense = similar(dense_sorted)
    avg = similar(avg_sorted)

    for k in eachindex(order)
        competition[order[k]] = competition_sorted[k]
        competition_max[order[k]] = competition_max_sorted[k]
        dense[order[k]] = dense_sorted[k]
        avg[order[k]] = avg_sorted[k]
    end

    return Dict(
        "competition" => competition,
        "competition_max" => competition_max,
        "dense" => dense,
        "avg" => avg,
    )
end

export competition_ranks_from_scores, rank_scores
