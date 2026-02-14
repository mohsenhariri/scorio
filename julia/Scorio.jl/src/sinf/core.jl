"""Sequential inference helpers."""

# NOTE: Python currently calls ranking_confidence() in this module,
# but that helper is not defined there. We use the intended normal
# separation criterion: P(mu_a > mu_b) with independent Gaussian uncertainty.
function _ranking_confidence(
    mu_a::Real,
    sigma_a::Real,
    mu_b::Real,
    sigma_b::Real,
)::Tuple{Float64, Float64}
    dmu = Float64(mu_a) - Float64(mu_b)
    s_a = Float64(sigma_a)
    s_b = Float64(sigma_b)
    denom = sqrt(s_a * s_a + s_b * s_b)

    if denom == 0.0
        if dmu > 0.0
            return 1.0, Inf
        elseif dmu < 0.0
            return 0.0, -Inf
        end
        return 0.5, 0.0
    end

    z = dmu / denom
    rho = _normal_cdf(z)
    return Float64(rho), Float64(z)
end

function ci_from_mu_sigma(
    mu::Real,
    sigma::Real,
    confidence::Real=0.95,
    clip::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
)::Tuple{Float64, Float64}
    sigma_f = Float64(sigma)
    if sigma_f < 0.0
        error("sigma must be >= 0.")
    end

    z = _z_value(confidence; two_sided=true)
    lo = Float64(mu) - z * sigma_f
    hi = Float64(mu) + z * sigma_f

    if !isnothing(clip)
        lo = max(Float64(clip[1]), lo)
        hi = min(Float64(clip[2]), hi)
    end

    return Float64(lo), Float64(hi)
end

function should_stop(
    sigma::Real;
    confidence::Real=0.95,
    max_ci_width::Union{Nothing, Real}=nothing,
    max_half_width::Union{Nothing, Real}=nothing,
)::Bool
    if (isnothing(max_ci_width) && isnothing(max_half_width)) ||
       (!isnothing(max_ci_width) && !isnothing(max_half_width))
        error("Provide exactly one of max_ci_width or max_half_width.")
    end

    sigma_f = Float64(sigma)
    if sigma_f < 0.0
        error("sigma must be >= 0.")
    end

    z = _z_value(confidence; two_sided=true)
    half = z * sigma_f

    if !isnothing(max_half_width)
        return half <= Float64(max_half_width)
    end

    return 2.0 * half <= Float64(max_ci_width)
end

function should_stop_top1(
    mus_in_id_order,
    sigmas_in_id_order;
    confidence::Real=0.95,
    method::AbstractString="ci_overlap",
)::Dict{String, Any}
    mus = Float64.(collect(mus_in_id_order))
    sigmas = Float64.(collect(sigmas_in_id_order))

    if ndims(mus) != 1 || ndims(sigmas) != 1 || length(mus) != length(sigmas)
        error("mus_in_id_order and sigmas_in_id_order must be 1D and same shape.")
    end
    if isempty(mus)
        error("Empty inputs.")
    end

    leader = argmax(mus)
    ambiguous = Int[]

    if method == "ci_overlap"
        z = _z_value(confidence; two_sided=true)
        lo_leader = mus[leader] - z * sigmas[leader]

        for j in eachindex(mus)
            if j == leader
                continue
            end
            hi_j = mus[j] + z * sigmas[j]
            if lo_leader <= hi_j
                push!(ambiguous, j)
            end
        end

        return Dict(
            "stop" => isempty(ambiguous),
            "leader" => leader,
            "ambiguous" => ambiguous,
        )
    end

    if method == "zscore"
        for j in eachindex(mus)
            if j == leader
                continue
            end
            rho, _ = _ranking_confidence(mus[leader], sigmas[leader], mus[j], sigmas[j])
            if rho < Float64(confidence)
                push!(ambiguous, j)
            end
        end

        return Dict(
            "stop" => isempty(ambiguous),
            "leader" => leader,
            "ambiguous" => ambiguous,
        )
    end

    error("method must be 'ci_overlap' or 'zscore'.")
end

function suggest_next_allocation(
    mus_in_id_order,
    sigmas_in_id_order;
    confidence::Real=0.95,
    method::AbstractString="ci_overlap",
)::Tuple{Int, Int}
    mus = Float64.(collect(mus_in_id_order))
    sigmas = Float64.(collect(sigmas_in_id_order))

    if ndims(mus) != 1 || ndims(sigmas) != 1 || length(mus) != length(sigmas)
        error("mus_in_id_order and sigmas_in_id_order must be 1D and same shape.")
    end
    if length(mus) < 2
        error("Need at least two methods to allocate.")
    end

    leader = argmax(mus)
    candidates = [j for j in eachindex(mus) if j != leader]

    if method == "ci_overlap"
        z = _z_value(confidence; two_sided=true)
        lo_leader = mus[leader] - z * sigmas[leader]

        function margin(j::Int)::Float64
            return lo_leader - (mus[j] + z * sigmas[j])
        end

        competitor = candidates[argmin([margin(j) for j in candidates])]
        return leader, competitor
    end

    if method == "zscore"
        function zsep(j::Int)::Float64
            _, z = _ranking_confidence(mus[leader], sigmas[leader], mus[j], sigmas[j])
            return z
        end

        competitor = candidates[argmin([zsep(j) for j in candidates])]
        return leader, competitor
    end

    error("method must be 'ci_overlap' or 'zscore'.")
end
