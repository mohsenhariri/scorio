"""Shared helpers for eval metrics and credible intervals."""

function _as_2d_int_matrix(R)::Matrix{Int}
    if !(R isa AbstractArray)
        error("R must be a 1D or 2D array.")
    end

    Rm = Int.(Array(R))
    if ndims(Rm) == 1
        return reshape(Rm, 1, :)
    end
    if ndims(Rm) != 2
        error("R must be a 1D or 2D array.")
    end
    return Rm
end

function _validate_matrix_range(
    R::AbstractMatrix{<:Integer},
    low::Integer,
    high::Integer,
    name::AbstractString,
)::Nothing
    if isempty(R)
        return nothing
    end
    if minimum(R) < low || maximum(R) > high
        error("Entries of $name must be integers in [$low, $high].")
    end
    return nothing
end

function _validate_binary(R::AbstractMatrix{<:Integer}, name::AbstractString="R")::Nothing
    _validate_matrix_range(R, 0, 1, name)
    return nothing
end

# Rational approximation of the inverse standard normal CDF.
function _normal_ppf(p::Float64)::Float64
    if p == 0.0
        return -Inf
    elseif p == 1.0
        return Inf
    end

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
    end

    q = p - 0.5
    r = q * q
    num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    return num / den
end

# Abramowitz-Stegun normal CDF approximation.
function _normal_cdf(x::Float64)::Float64
    z = abs(x)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.319381530 +
        t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + 1.330274429 * t)))
    )
    pdf = 0.3989422804014327 * exp(-0.5 * z * z)
    cdf = 1.0 - pdf * poly
    return x >= 0.0 ? cdf : 1.0 - cdf
end

function _z_value(confidence::Real; two_sided::Bool=true)::Float64
    conf = Float64(confidence)
    if !(0.0 < conf < 1.0)
        error("confidence must be in (0,1); got $confidence")
    end
    if two_sided
        return _normal_ppf(0.5 + 0.5 * conf)
    end
    return _normal_ppf(conf)
end

function normal_credible_interval(
    mu::Real,
    sigma::Real;
    credibility::Real=0.95,
    two_sided::Bool=true,
    bounds::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
)::Tuple{Float64, Float64}
    mu_f = Float64(mu)
    sigma_f = Float64(sigma)

    if sigma_f < 0.0
        error("sigma must be >= 0; got $sigma")
    end

    z = _z_value(credibility; two_sided=two_sided)
    if two_sided
        lo = mu_f - z * sigma_f
        hi = mu_f + z * sigma_f
    else
        lo = -Inf
        hi = mu_f + z * sigma_f
    end

    if !isnothing(bounds)
        b_lo = Float64(bounds[1])
        b_hi = Float64(bounds[2])
        if b_lo > b_hi
            error("bounds must satisfy bounds[1] <= bounds[2]")
        end
        lo = max(lo, b_lo)
        hi = min(hi, b_hi)
    end

    return lo, hi
end
