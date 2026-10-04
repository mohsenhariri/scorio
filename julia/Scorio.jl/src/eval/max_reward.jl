"""Max-reward evaluation metrics for categorical outcomes."""

using SpecialFunctions: loggamma

function _prepare_categorical_input(R, w=nothing, R0=nothing)
    Rm = _as_2d_int_matrix(R)
    if isnothing(w)
        unique_vals = unique(Rm)
        is_binary = length(unique_vals) <= 2 && all(v -> v == 0 || v == 1, unique_vals)
        if !is_binary
            unique_str = join(sort(unique_vals), ", ")
            error(
                "R contains more than 2 unique values ($unique_str), so weight vector 'w' must be provided. " *
                "Please specify a weight vector of length $(length(unique_vals)) to map each category to a score.",
            )
        end
        wv = [0.0, 1.0]
    else
        wv = _tp_vector(w, "w")
        isempty(wv) && error("w must be nonempty")
    end

    M, _ = size(Rm)
    C = length(wv) - 1
    _validate_matrix_range(Rm, 0, C, "R")

    if isnothing(R0)
        R0m = zeros(Int, M, 0)
    else
        R0m = _as_eval_int_array(R0, "R0")
        if ndims(R0m) == 1
            try
                # Match NumPy's row-major `reshape(M, -1)` for flat priors.
                R0m = permutedims(reshape(R0m, :, M))
            catch
                error("R0 must have the same number of rows (M) as R.")
            end
        elseif ndims(R0m) != 2
            error("R0 must be a 1D or 2D array.")
        end
        if size(R0m, 1) != M
            error("R0 must have the same number of rows (M) as R.")
        end
        _validate_matrix_range(R0m, 0, C, "R0")
    end
    return Rm, wv, R0m
end

function _row_bincount_eval(A::AbstractMatrix{<:Integer}, width::Integer)::Matrix{Int}
    out = zeros(Int, size(A, 1), width)
    @inbounds for row in axes(A, 1)
        for col in axes(A, 2)
            out[row, A[row, col] + 1] += 1
        end
    end
    return out
end

function _grouped_posterior_params(R, w=nothing, R0=nothing)
    Rm, wv, R0m = _prepare_categorical_input(R, w, R0)
    C = length(wv) - 1
    levels = sort(unique(wv))
    n_counts = _row_bincount_eval(Rm, C + 1)
    n0_counts = _row_bincount_eval(R0m, C + 1) .+ 1
    alpha_cat = n_counts .+ n0_counts
    gamma = zeros(Float64, size(Rm, 1), length(levels))

    @inbounds for cat in 1:(C + 1)
        level_idx = findfirst(isequal(wv[cat]), levels)
        gamma[:, level_idx] .+= alpha_cat[:, cat]
    end
    return gamma, levels
end

function _eval_logsumexp(values::AbstractVector{<:Real})::Float64
    max_value = maximum(values)
    if max_value == -Inf
        return -Inf
    end
    return Float64(max_value + log(sum(exp(Float64(v) - max_value) for v in values)))
end


"""
    max_at_k(R, k, w=nothing) -> Float64

Expected best reward among `k` samples drawn without replacement from each
question's observed response bank.
"""
function max_at_k(R, k::Integer, w)::Float64
    Rm, wv, _ = _prepare_categorical_input(R, w); M, N = size(Rm)
    _tp_integer(k); k <= N || error("k must not exceed N")
    levels = sort(unique(wv)); offset, scale, normalized = _tp_scaled_rewards(levels)
    scale == 0 && return offset
    gaps = diff(normalized)
    means = [normalized[1]+sum(gaps[j]*_pass_probability(N, count(c -> wv[c+1] > levels[j], row), k) for j in eachindex(gaps)) for row in eachrow(Rm)]
    return offset+scale*(sum(means)/M)
end

function _max_at_k_bayes(R, k::Integer, w=nothing, R0=nothing)
    _tp_integer(k); gamma, levels = _grouped_posterior_params(R, w, R0)
    offset, scale, normalized = _tp_scaled_rewards(levels)
    scale == 0 && return offset, 0.0, levels
    gaps = diff(normalized)
    moments = [begin
        total = sum(row); cum = cumsum(row)[1:end-1]
        logs = [_tp_log_beta_power(a, total-a, k) for a in cum]
        mean = normalized[1]+dot(gaps, -expm1.(logs))
        var = 0.0
        for i in eachindex(gaps), j in eachindex(gaps)
            lower, upper = minmax(i, j)
            cross = _tp_log_beta_power(cum[upper], total-cum[upper], 2k) + (i == j ? 0.0 : _tp_log_beta_power(cum[lower], cum[upper]-cum[lower], k))
            var += gaps[i]*gaps[j]*_tp_log_difference(cross, logs[i]+logs[j])
        end
        (mean, sqrt(_tp_variance(var)))
    end for row in eachrow(gamma)]
    M = size(gamma, 1)
    return offset+scale*(sum(first, moments)/M), scale*foldl(hypot, last.(moments); init=0.0)/M, levels
end

"""
    max_at_k_ci(R, k, w=nothing, R0=nothing, confidence=0.95, bounds=nothing)

Bayesian posterior `(mu, sigma, lo, hi)` for latent Max@k.
"""
function max_at_k_ci(
    R,
    k::Integer,
    w,
    R0,
    confidence::Real,
    bounds,
)::Tuple{Float64, Float64, Float64, Float64}
    _tp_integer(k)
    if k == 1
        return bayes_ci(R, w, R0, confidence, bounds)
    end

    mu, sigma, levels = _max_at_k_bayes(R, k, w, R0)
    interval_bounds = isnothing(bounds) ? (Float64(minimum(levels)), Float64(maximum(levels))) : bounds
    lo, hi = normal_credible_interval(
        mu,
        sigma;
        credibility=confidence,
        two_sided=true,
        bounds=interval_bounds,
    )
    return Float64(mu), Float64(sigma), Float64(lo), Float64(hi)
end
