# TailPass posterior profiles and utilities (Hariri et al., 2026, Sections 2–3 and Appendix C).
import Base: summary
using Random: AbstractRNG, MersenneTwister, rand, randn, default_rng

function _tp_thresholds(k, thresholds)
    _tp_integer(k)
    if isnothing(thresholds)
        k <= _TP_MAX_STATES ||
            error("Default threshold grid is too large; supply explicit thresholds")
        return collect(1:k) ./ k
    end
    grid = _tp_vector(thresholds, "thresholds")
    !isempty(grid) && all(0 .<= grid .<= 1) && all(diff(grid) .> 0) ||
        error("thresholds must be nonempty and strictly increasing in [0, 1]")
    return grid
end

"""A reusable posterior profile; `mean`, `question_mean`, `covariance`, and `std` are computed lazily."""
mutable struct TailPassProfile
    k::Int
    thresholds::Vector{Float64}
    _alpha::Matrix{Float64}
    _scores::Vector{Float64}
    _grid::Any
    _question_mean::Union{Nothing,Matrix{Float64}}
    _covariance::Union{Nothing,Matrix{Float64}}
end
TailPassProfile(k, thresholds, alpha, scores) = TailPassProfile(
    k,
    copy(thresholds),
    copy(alpha),
    copy(scores),
    nothing,
    nothing,
    nothing,
)

"""
    tailpass(R, k; w=nothing, R0=nothing, eta=1.0, prior=1.0, thresholds=nothing)

Fit `prior + counts(R) + eta*counts(R0)`. Rubric scores are in [0, 1]; repeated
scores merge by summing concentrations. Binary prior order is failure, success.
The reporting budget k may exceed the observed trial count. Threshold zero is
always met; discovery means strictly positive credit.
"""
function tailpass(R, k, w, R0; eta = 1.0, prior = 1.0, thresholds = nothing)
    k = _tp_integer(k)
    grid = _tp_thresholds(k, thresholds)
    Rm, scores, aux = _prepare_categorical_input(R, w, R0)
    all(0 .<= scores .<= 1) || error("TailPass rubric scores must lie in [0, 1]")
    eta = _tp_unit(eta, "eta")
    counts = _row_bincount_eval(Rm, length(scores))
    prior_counts = _row_bincount_eval(aux, length(scores))
    M, L = size(counts)
    base = if prior isa Real
        fill(_tp_positive(prior, "prior"), M, L)
    else
        raw = if prior isa AbstractMatrix
            Float64.(prior)
        elseif !isempty(prior) && first(prior) isa Union{AbstractVector,Tuple}
            permutedims(hcat(collect.(prior)...))
        else
            permutedims(_tp_vector(prior, "prior"))
        end
        size(raw, 1) in (1, M) && size(raw, 2) in (1, L) || error(
            "prior must be a scalar, category vector, or question-by-category matrix",
        )
        all(x -> x isa Real && isfinite(x) && x > 0, raw) ||
            error("prior must contain positive finite concentrations")
        repeat(raw, M ÷ size(raw, 1), L ÷ size(raw, 2))
    end
    alpha = base .+ counts .+ eta .* prior_counts
    all(isfinite, sum(alpha; dims = 2)) ||
        error("Posterior concentration totals must be finite")
    levels = sort(unique(scores))
    grouped = hcat([vec(sum(alpha[:, scores .== s]; dims = 2)) for s in levels]...)
    return TailPassProfile(k, grid, grouped, levels)
end
tailpass(R, k; w = nothing, R0 = nothing, kwargs...) = tailpass(R, k, w, R0; kwargs...)
tailpass(R, k, w; R0 = nothing, kwargs...) = tailpass(R, k, w, R0; kwargs...)

function _tp_grid(p::TailPassProfile)
    if isnothing(getfield(p, :_grid))
        counts = _tp_counts(p.k, length(p._scores))
        # BigFloat is used only for score-boundary rounding, never for probabilities.
        scores = [
            Float64(sum(BigFloat.(row) .* BigFloat.(p._scores))/p.k) for
            row in eachrow(counts)
        ]
        p._grid = (
            counts = counts,
            scores = scores,
            coeff = _tp_logcoeff(counts),
            order = sortperm(scores),
        )
    end
    return getfield(p, :_grid)
end
function _tp_tails(p::TailPassProfile, pmf)
    g = _tp_grid(p)
    tails = [reverse(cumsum(reverse(pmf[g.order]))); 0.0]
    return clamp.(tails[searchsortedfirst.(Ref(g.scores[g.order]), p.thresholds)], 0, 1)
end
function _tp_question_mean(p::TailPassProfile)
    if isnothing(p._question_mean)
        g = _tp_grid(p)
        p._question_mean = permutedims(
            hcat(
                [
                    _tp_tails(p, _tp_predictive(g.counts, a, g.coeff)) for
                    a in eachrow(p._alpha)
                ]...,
            ),
        )
    end
    return copy(p._question_mean)
end
function _tp_profile_covariance(p::TailPassProfile)
    if isnothing(p._covariance)
        g = _tp_grid(p)
        max(size(g.counts, 1), length(p.thresholds))^2 <= _TP_MAX_PAIRS ||
            error("Exact covariance is too large; use shared posterior draws")
        values = Float64.(g.scores .>= p.thresholds')
        covariance = zeros(length(p.thresholds), length(p.thresholds))
        for a in eachrow(p._alpha)
            covariance .+=
                _tp_covariance(g.counts, a, values, g.coeff) ./ p.question_count^2
        end
        for j in axes(covariance, 1)
            covariance[j, j] = _tp_variance(covariance[j, j])
        end
        p._covariance = covariance
    end
    return copy(p._covariance)
end
function Base.getproperty(p::TailPassProfile, name::Symbol)
    name === :question_count && return size(getfield(p, :_alpha), 1)
    name === :question_mean && return _tp_question_mean(p)
    name === :mean && return vec(sum(_tp_question_mean(p); dims = 1)) ./ p.question_count
    name === :covariance && return _tp_profile_covariance(p)
    name === :std && return sqrt.(diag(_tp_profile_covariance(p)))
    return getfield(p, name)
end
Base.propertynames(::TailPassProfile, private::Bool = false) =
    private ? fieldnames(TailPassProfile) :
    (:k, :thresholds, :question_count, :question_mean, :mean, :covariance, :std)
function _tp_aggregate(p, moments)
    return sum(first, moments)/p.question_count,
    sqrt(_tp_variance(sum(last, moments)/p.question_count^2))
end
function _tp_payoff(p, values)
    g = _tp_grid(p)
    return _tp_aggregate(
        p,
        [
            (
                dot(_tp_predictive(g.counts, a, g.coeff), values),
                _tp_covariance(g.counts, a, reshape(values, :, 1), g.coeff)[1, 1],
            ) for a in eachrow(p._alpha)
        ],
    )
end
"""Exact mean/std of a convex combination, including cross-threshold covariance."""
function linear(p::TailPassProfile, weights)
    w = TailPassWeights._validate_weights(weights, length(p.thresholds))
    return _tp_payoff(p, [sum(w[p.thresholds .<= s]) for s in _tp_grid(p).scores])
end
"""Exact mean/std of the full-score moment, independent of the reporting grid."""
function moment(p::TailPassProfile, lam)
    lam = _tp_positive(lam, "lam")
    if lam == 1
        return _tp_aggregate(
            p,
            [
                begin
                    total = sum(a)
                    probabilities = a ./ total
                    mean = dot(probabilities, p._scores)
                    (mean, dot(probabilities, (p._scores .- mean) .^ 2)/(total+1))
                end for a in eachrow(p._alpha)
            ],
        )
    elseif lam in (2, 4) && (length(p._scores) != 2 || p.k >= _TP_MAX_STATES)
        return _tp_aggregate(
            p,
            [
                _tp_polynomial_moments(p._scores, p.k, Int(lam), a) for
                a in eachrow(p._alpha)
            ],
        )
    end
    return _tp_payoff(p, _tp_grid(p).scores .^ lam)
end
function _tp_endpoint(p::TailPassProfile, full::Bool)
    mask = full ? p._scores .== 1 : p._scores .> 0
    (all(mask) || !any(mask)) && return Float64(all(mask)), 0.0
    return _tp_aggregate(
        p,
        [_tp_endpoint(p.k, sum(a[mask]), sum(a[.!mask]), !full) for a in eachrow(p._alpha)],
    )
end
"""Exact mean/std of positive bank credit."""
discovery(p::TailPassProfile) = _tp_endpoint(p, false)
"""Exact mean/std of every attempt receiving full credit."""
stability(p::TailPassProfile) = _tp_endpoint(p, true)
"""Reuse the fitted posterior at another positive reporting budget."""
at_k(p::TailPassProfile, k; thresholds = nothing) =
    TailPassProfile(_tp_integer(k), _tp_thresholds(k, thresholds), p._alpha, p._scores)

"""Shared latent probability draws. Utilities return one scalar per draw."""
struct TailPassDraws
    _source::TailPassProfile
    _probabilities::Array{Float64,3}
end
function _tp_loggamma_draw(rng, shape)
    shape < 1 &&
        return _tp_loggamma_draw(rng, shape+1) + log(max(nextfloat(0.0), rand(rng)))/shape
    d = shape - 1/3
    c = 1/sqrt(9d)
    while true
        x = randn(rng)
        base = 1+c*x
        base <= 0 && continue
        v = base^3
        u = rand(rng)
        if u < 1-0.0331*x^4 || log(u) < x*x/2 + d*(1-v+log(v))
            return log(d)+3log(base)
        end
    end
end
"""Draw independent question probabilities; an integer seed is reproducible within Julia."""
function sample(p::TailPassProfile, n_draws = 4000; rng = nothing)
    n = _tp_integer(n_draws, "n_draws", 2)
    generator =
        isnothing(rng) ? default_rng() :
        rng isa Integer ? MersenneTwister(_tp_integer(rng, "rng", 0)) : rng
    generator isa AbstractRNG || error("rng must be an integer seed or AbstractRNG")
    values = Array{Float64}(undef, n, size(p._alpha)...)
    for h = 1:n, question = 1:p.question_count
        a = p._alpha[question, :]
        values[h, question, :] =
            length(a) == 1 ? [1.0] :
            _tp_normalize([_tp_loggamma_draw(generator, v) for v in a])
    end
    return TailPassDraws(p, values)
end
function _tp_draw_profile(d::TailPassDraws)
    p = d._source
    g = _tp_grid(p)
    result = zeros(size(d._probabilities, 1), length(p.thresholds))
    for h in axes(result, 1), question = 1:p.question_count
        result[h, :] .+=
            _tp_tails(
                p,
                _tp_conditional(g.counts, d._probabilities[h, question, :], g.coeff),
            ) ./ p.question_count
    end
    return result
end
Base.getproperty(d::TailPassDraws, name::Symbol) =
    name === :profile ? _tp_draw_profile(d) : getfield(d, name)
Base.propertynames(::TailPassDraws, private::Bool = false) =
    private ? fieldnames(TailPassDraws) : (:profile,)
function _tp_apply(transform, d::TailPassDraws, aggregation = "question")
    aggregation == "profile" && return [transform(row) for row in eachrow(d.profile)]
    aggregation == "question" || error("aggregation must be 'question' or 'profile'")
    p = d._source
    g = _tp_grid(p)
    return [
        sum(
            transform(
                _tp_tails(
                    p,
                    _tp_conditional(g.counts, d._probabilities[h, question, :], g.coeff),
                ),
            ) for question = 1:p.question_count
        )/p.question_count for h in axes(d._probabilities, 1)
    ]
end
linear(d::TailPassDraws, weights) =
    d.profile * TailPassWeights._validate_weights(weights, length(d._source.thresholds))
function moment(d::TailPassDraws, lam)
    lam = _tp_positive(lam, "lam")
    p = d._source
    if lam in (1, 2, 4)
        return [
            sum(
                _tp_conditional_moment(d._probabilities[h, q, :], p._scores, p.k, Int(lam))
                for q = 1:p.question_count
            )/p.question_count for h in axes(d._probabilities, 1)
        ]
    end
    g = _tp_grid(p)
    payoff = g.scores .^ lam
    return [
        sum(
            dot(_tp_conditional(g.counts, d._probabilities[h, q, :], g.coeff), payoff) for
            q = 1:p.question_count
        )/p.question_count for h in axes(d._probabilities, 1)
    ]
end
"""Weighted spectrum power means, computed before averaging questions by default."""
function power_mean(d::TailPassDraws, q, weights = nothing; aggregation = "question")
    q = _tp_positive(q, "q")
    size = length(d._source.thresholds)
    w = TailPassWeights._validate_weights(
        isnothing(weights) ? TailPassWeights.uniform_weights(size) : weights,
        size,
    )
    return _tp_apply(d, aggregation) do values
        q == 1 && return dot(values, w)
        selected = values[w .> 0]
        active = w[w .> 0]
        scale = maximum(selected)
        scale == 0 && return 0.0
        logs = q .* log.(selected ./ scale)
        delta = dot(expm1.(logs), active)
        log_mean = abs(delta) < 0.25 ? log1p(delta) : log(dot(exp.(logs), active))
        return scale * exp(log_mean/q)
    end
end
qrs(d::TailPassDraws, weights = nothing; aggregation = "question") =
    power_mean(d, 2.0, weights; aggregation = aggregation)
"""Common-threshold m-rollout utility on its probability scale, m >= 2."""
function rollout(d::TailPassDraws, m, weights = nothing)
    m = _tp_integer(m, "m", 2)
    size = length(d._source.thresholds)
    w = TailPassWeights._validate_weights(
        isnothing(weights) ? TailPassWeights.uniform_weights(size) : weights,
        size,
    )
    return _tp_apply(v -> dot(v .^ m, w), d)
end
"""Question-first harmonic balance; alpha weights discovery, 1-alpha weights stability."""
function harmonic(d::TailPassDraws, alpha = 0.5)
    alpha = _tp_unit(alpha, "alpha")
    p = d._source
    result = zeros(size(d._probabilities, 1))
    for h in eachindex(result), q = 1:p.question_count
        probabilities = d._probabilities[h, q, :]
        zero = clamp(sum(probabilities[p._scores .== 0]), 0, 1)
        one = clamp(sum(probabilities[p._scores .== 1]), 0, 1)
        discovery = -expm1(p.k * log(zero))
        stability = one^p.k
        value =
            alpha == 1 ? discovery :
            alpha == 0 ? stability :
            discovery > 0 && stability > 0 ? 1/(alpha/discovery + (1-alpha)/stability) : 0.0
        result[h] += value/p.question_count
    end
    return result
end
discovery(d::TailPassDraws) = harmonic(d, 1.0)
stability(d::TailPassDraws) = harmonic(d, 0.0)
"""Question-first reference-profile shortfall; lower is better. Weights need not sum to one."""
function shortfall(d::TailPassDraws, target, weights = nothing; epsilon = 0.01)
    t = _tp_vector(target, "target")
    size = length(d._source.thresholds)
    length(t) == size && all(0 .<= t .<= 1) && all(diff(t) .<= 0) ||
        error("target must be nonincreasing in [0, 1] and match thresholds")
    w = isnothing(weights) ? ones(size) : _tp_vector(weights, "weights")
    length(w) == size && all(w .> 0) ||
        error("shortfall weights must be positive and match thresholds")
    epsilon = _tp_positive(epsilon, "epsilon")
    return _tp_apply(d) do values
        gaps = max.(t .- values, 0) .* w
        maximum(gaps) + epsilon*sum(gaps)
    end
end
at_k(d::TailPassDraws, k; thresholds = nothing) =
    TailPassDraws(at_k(d._source, k; thresholds = thresholds), d._probabilities)
"""Monte Carlo (mean, sample std, lower, upper), with equal-tailed pointwise intervals."""
function summary(d::TailPassDraws, values = nothing; confidence = 0.95)
    _z_value(confidence)
    samples = isnothing(values) ? d.profile : values
    samples isa AbstractArray &&
    ndims(samples) in (1, 2) &&
    size(samples, 1) == size(d._probabilities, 1) &&
    all(x -> x isa Real && isfinite(x), samples) ||
        error("values must be finite and match this draw set's sample count")
    matrix = ndims(samples) == 2
    rows = matrix ? samples : reshape(samples, :, 1)
    results = zeros(4, size(rows, 2))
    for j in axes(rows, 2)
        column = sort(Float64.(rows[:, j]))
        n = length(column)
        mean = sum(column)/n
        std = sqrt(sum((column .- mean) .^ 2)/(n-1))
        function quantile(p)
            at = p*(n-1)+1
            i = floor(Int, at)
            return column[i] + (at-i)*(column[min(i+1, n)]-column[i])
        end
        results[:, j] = [mean, std, quantile((1-confidence)/2), quantile((1+confidence)/2)]
    end
    return matrix ? Tuple(copy(row) for row in eachrow(results)) : Tuple(results[:, 1])
end
"""Profile summary; method='mc' uses equal-tailed draws, 'normal' uses exact moments."""
function ci(
    p::TailPassProfile,
    confidence = 0.95;
    method = "mc",
    n_draws = 4000,
    rng = nothing,
)
    z = _z_value(confidence)
    method == "mc" && return summary(sample(p, n_draws; rng = rng); confidence = confidence)
    method == "normal" || error("method must be 'mc' or 'normal'")
    mu, std = p.mean, p.std
    return mu, std, clamp.(mu .- z .* std, 0, 1), clamp.(mu .+ z .* std, 0, 1)
end
function linear_ci(
    p::TailPassProfile,
    weights,
    confidence = 0.95;
    method = "mc",
    n_draws = 4000,
    rng = nothing,
)
    _z_value(confidence)
    TailPassWeights._validate_weights(weights, length(p.thresholds))
    if method == "normal"
        mu, std = linear(p, weights)
        return (
            mu,
            std,
            normal_credible_interval(mu, std; credibility = confidence, bounds = (0, 1))...,
        )
    end
    method == "mc" || error("method must be 'mc' or 'normal'")
    d = sample(p, n_draws; rng = rng)
    return summary(d, linear(d, weights); confidence = confidence)
end
function moment_ci(
    p::TailPassProfile,
    lam,
    confidence = 0.95;
    method = "mc",
    n_draws = 4000,
    rng = nothing,
)
    _z_value(confidence)
    _tp_positive(lam, "lam")
    if method == "normal"
        mu, std = moment(p, lam)
        return (
            mu,
            std,
            normal_credible_interval(mu, std; credibility = confidence, bounds = (0, 1))...,
        )
    end
    method == "mc" || error("method must be 'mc' or 'normal'")
    d = sample(p, n_draws; rng = rng)
    return summary(d, moment(d, lam); confidence = confidence)
end
"""Finite-bank profile from sampling k observed trials without replacement, k <= N."""
function tailpass_empirical(R, k, w; thresholds = nothing)
    k = _tp_integer(k)
    grid = _tp_thresholds(k, thresholds)
    Rm, scores, _ = _prepare_categorical_input(R, w)
    k <= size(Rm, 2) || error("Empirical k must not exceed observed trials")
    all(0 .<= scores .<= 1) || error("TailPass rubric scores must lie in [0, 1]")
    levels = sort(unique(scores))
    observed = _row_bincount_eval(Rm, length(scores))
    grouped = hcat([vec(sum(observed[:, scores .== s]; dims = 2)) for s in levels]...)
    p = TailPassProfile(k, grid, Float64.(grouped), levels)
    g = _tp_grid(p)
    result = zeros(length(grid))
    for counts in eachrow(grouped)
        logs = [
            sum(_log_comb(counts[j], n) for (j, n) in enumerate(row)) for
            row in eachrow(g.counts)
        ]
        result .+= _tp_tails(p, _tp_normalize(logs)) ./ size(Rm, 1)
    end
    return result
end
tailpass_empirical(R, k; w = nothing, thresholds = nothing) =
    tailpass_empirical(R, k, w; thresholds = thresholds)

export tailpass,
    tailpass_empirical,
    TailPassProfile,
    TailPassDraws,
    TailPassWeights,
    tailpass_weights,
    linear,
    moment,
    discovery,
    stability,
    at_k,
    sample,
    summary,
    ci,
    linear_ci,
    moment_ci,
    power_mean,
    qrs,
    rollout,
    harmonic,
    shortfall
