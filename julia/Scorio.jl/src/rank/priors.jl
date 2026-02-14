"""Prior penalties used by MAP ranking estimators."""

"""
    Prior

Abstract supertype for prior penalty specifications used by MAP rankers.

Concrete subtypes define hyperparameters for an internal penalty term over the
latent score vector `theta`.
"""
abstract type Prior end

"""
    EmpiricalPrior(R0; var=1.0, eps=1e-6)

Empirical Gaussian-style prior centered at logits inferred from baseline
outcomes.

`R0` is accepted as shape `(L, M)` or `(L, M, D)`. A 2D input is promoted to
`(L, M, 1)`.

# Arguments
- `R0`: baseline outcomes per model. Typically binary outcomes in `{0,1}`.
- `var::Real=1.0`: variance used in the quadratic penalty; must be positive.
- `eps::Real=1e-6`: clipping level used before logit transform.
  No explicit range check is applied; choose `0 < eps < 0.5` in practice.

# Returns
- `EmpiricalPrior`: stores `R0`, `var`, `eps`, and centered `prior_mean`.

# Formula
For model ``l``:

```math
a_l = \\frac{1}{M D}\\sum_{m=1}^{M}\\sum_{d=1}^{D} R^0_{lmd}
```

```math
\\tilde a_l = \\operatorname{clip}(a_l, \\varepsilon, 1-\\varepsilon), \\qquad
\\mu_l = \\log\\!\\left(\\frac{\\tilde a_l}{1-\\tilde a_l}\\right)
```

Then mean-center ``\\mu`` for identifiability and use:

```math
\\operatorname{penalty}(\\theta)
= \\frac{1}{2\\,\\mathrm{var}}\\sum_{l=1}^{L}(\\theta_l-\\mu_l)^2
```

# Examples
```julia
R0 = Int[
    1 1 1 0 1
    0 1 0 0 1
]
prior = EmpiricalPrior(R0; var=2.0, eps=1e-6)
```
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

# Arguments
- `mean::Real=0.0`: prior mean.
- `var::Real=1.0`: prior variance; must be positive.

# Returns
- `GaussianPrior`

# Formula

```math
\\operatorname{penalty}(\\theta)
= \\frac{1}{2\\,\\mathrm{var}}\\sum_i (\\theta_i-\\mathrm{mean})^2
```
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

# Arguments
- `loc::Real=0.0`: location parameter.
- `scale::Real=1.0`: scale parameter; must be positive.

# Returns
- `LaplacePrior`

# Formula

```math
\\operatorname{penalty}(\\theta)
= \\frac{1}{\\mathrm{scale}}\\sum_i \\left|\\theta_i-\\mathrm{loc}\\right|
```
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

# Arguments
- `loc::Real=0.0`: location parameter.
- `scale::Real=1.0`: scale parameter; must be positive.

# Returns
- `CauchyPrior`

# Formula
Let ``z_i = (\\theta_i-\\mathrm{loc})/\\mathrm{scale}``.

```math
\\operatorname{penalty}(\\theta) = \\sum_i \\log(1 + z_i^2)
```
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

# Returns
- `UniformPrior`

# Formula

```math
\\operatorname{penalty}(\\theta) = 0
```
"""
struct UniformPrior <: Prior end

"""
    CustomPrior(penalty_fn)

User-defined prior from a callable penalty function.

# Arguments
- `penalty_fn`: callable with signature `penalty_fn(theta)` returning a scalar
  penalty value.

# Returns
- `CustomPrior`

# Notes
`penalty_fn` is used directly with no transformation of `theta`.
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
