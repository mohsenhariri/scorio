"""Prior penalties used by MAP ranking estimators."""

"""
    Prior

Abstract supertype for prior penalty specifications used in MAP rankers.
"""
abstract type Prior end

"""
    EmpiricalPrior(R0; var=1.0, eps=1e-6)

Empirical Gaussian-style prior whose mean is inferred from baseline outcomes
`R0` (via clipped logit accuracy), centered for identifiability.
"""
struct EmpiricalPrior <: Prior
    R0::Array
    var::Float64
    eps::Float64
    prior_mean::Vector{Float64}
end

function EmpiricalPrior(R0; var::Real=1.0, eps::Real=1e-6)
    if var <= 0
        error("Variance must be positive")
    end

    R0_arr = Array(R0)

    if ndims(R0_arr) == 2
        R0_arr = reshape(R0_arr, size(R0_arr, 1), size(R0_arr, 2), 1)
    elseif ndims(R0_arr) != 3
        error("R0 must be 2D (L, M) or 3D (L, M, D), got ndim=$(ndims(R0_arr))")
    end

    L = size(R0_arr, 1)
    acc = [sum(@view R0_arr[l, :, :]) / length(@view R0_arr[l, :, :]) for l in 1:L]

    acc_clipped = clamp.(Float64.(acc), Float64(eps), 1.0 - Float64(eps))
    prior_mean = log.(acc_clipped ./ (1.0 .- acc_clipped))
    prior_mean .-= (sum(prior_mean) / length(prior_mean))

    return EmpiricalPrior(R0_arr, Float64(var), Float64(eps), prior_mean)
end

"""
    GaussianPrior(mean=0.0, var=1.0)

Gaussian prior on latent parameters with quadratic penalty.
"""
struct GaussianPrior <: Prior
    mean::Float64
    var::Float64
    function GaussianPrior(mean::Real=0.0, var::Real=1.0)
        if var <= 0
            error("Variance must be positive")
        end
        return new(Float64(mean), Float64(var))
    end
end

"""
    LaplacePrior(loc=0.0, scale=1.0)

Laplace prior on latent parameters with L1 penalty.
"""
struct LaplacePrior <: Prior
    loc::Float64
    scale::Float64
    function LaplacePrior(loc::Real=0.0, scale::Real=1.0)
        if scale <= 0
            error("Scale must be positive")
        end
        return new(Float64(loc), Float64(scale))
    end
end

"""
    CauchyPrior(loc=0.0, scale=1.0)

Cauchy prior on latent parameters with log-quadratic penalty.
"""
struct CauchyPrior <: Prior
    loc::Float64
    scale::Float64
    function CauchyPrior(loc::Real=0.0, scale::Real=1.0)
        if scale <= 0
            error("Scale must be positive")
        end
        return new(Float64(loc), Float64(scale))
    end
end

"""
    UniformPrior()

Improper flat prior with zero penalty.
"""
struct UniformPrior <: Prior end

"""
    CustomPrior(penalty_fn)

User-defined prior from a callable penalty function `penalty_fn(theta)`.
"""
struct CustomPrior{F} <: Prior
    penalty_fn::F
    function CustomPrior(penalty_fn)
        if isempty(methods(penalty_fn))
            error("penalty_fn must be callable")
        end
        return new{typeof(penalty_fn)}(penalty_fn)
    end
end

function penalty(prior::EmpiricalPrior, theta)
    if length(theta) != length(prior.prior_mean)
        error(
            "theta length ($(length(theta))) must match number of models ($(length(prior.prior_mean)))",
        )
    end

    θ = Float64.(theta)
    return sum((θ .- prior.prior_mean) .^ 2) / (2.0 * prior.var)
end

function penalty(prior::GaussianPrior, theta)
    θ = Float64.(theta)
    return sum((θ .- prior.mean) .^ 2) / (2.0 * prior.var)
end

function penalty(prior::LaplacePrior, theta)
    θ = Float64.(theta)
    return sum(abs.(θ .- prior.loc)) / prior.scale
end

function penalty(prior::CauchyPrior, theta)
    θ = Float64.(theta)
    z = (θ .- prior.loc) ./ prior.scale
    return sum(log1p.(z .^ 2))
end

function penalty(::UniformPrior, theta)
    return 0.0
end

function penalty(prior::CustomPrior, theta)
    return prior.penalty_fn(theta)
end

function penalty(prior::Prior, theta)
    error("No penalty method implemented for prior type $(typeof(prior))")
end
