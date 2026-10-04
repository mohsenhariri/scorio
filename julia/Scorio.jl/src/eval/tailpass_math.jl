# Numerical kernels for TailPass count distributions and posterior moments.
using LinearAlgebra: SymTridiagonal, eigen, dot, diag

const _TP_MAX_STATES = 20_000
const _TP_MAX_PAIRS = 4_000_000

function _tp_integer(x, name = "k", minimum = 1)
    x isa Integer && !(x isa Bool) && x >= minimum ||
        error("$name must be an integer >= $minimum")
    return Int(x)
end
function _tp_positive(x, name)
    x isa Real && !(x isa Bool) && isfinite(x) && x > 0 ||
        error("$name must be finite and positive")
    return Float64(x)
end
function _tp_unit(x, name)
    x isa Real && isfinite(x) && 0 <= x <= 1 || error("$name must be in [0, 1]")
    return Float64(x)
end
function _tp_vector(x, name)
    (x isa AbstractVector || x isa Tuple) && all(v -> v isa Real && isfinite(v), x) ||
        error("$name must be a finite numeric vector")
    return Float64.(collect(x))
end
function _tp_variance(x)
    isfinite(x) && x >= -1e-12 || error("Invalid posterior variance: $x")
    return max(0.0, x)
end
function _tp_normalize(logs)
    weights = exp.(logs .- maximum(logs))
    total = sum(weights)
    isfinite(total) && total > 0 || error("Invalid probability normalization")
    return weights ./ total
end
function _tp_counts(k, categories)
    size = binomial(big(k) + categories - 1, categories - 1)
    size <= _TP_MAX_STATES || error(
        "Exact profile exceeds 20000 count states; use a smaller k/rubric or moment(1, 2, 4)",
    )
    rows = Vector{Int}[]
    function visit(prefix, remaining, slots)
        if slots == 1
            push!(rows, [prefix; remaining])
        else
            for n = remaining:-1:0
                visit([prefix; n], remaining - n, slots - 1)
            end
        end
    end
    visit(Int[], k, categories)
    return permutedims(hcat(rows...))
end
_tp_logcoeff(counts) =
    [loggamma(sum(row) + 1) - sum(loggamma.(row .+ 1)) for row in eachrow(counts)]
function _tp_beta_binomial(k, a, b)
    logs = zeros(k + 1)
    for j = 0:(k-1)
        logs[j+2] = logs[j+1] + log(k-j) - log(j+1) + log(a+j) - log(b+k-j-1)
    end
    return _tp_normalize(logs)
end
function _tp_predictive(counts, alpha, coeff)
    k = sum(counts[1, :])
    length(alpha) == 2 && return _tp_beta_binomial(k, alpha[2], alpha[1])
    logs = copy(coeff)
    for (j, a) in enumerate(alpha)
        rising = [0.0; cumsum(log.(a .+ (0:(k-1))))]
        logs .+= rising[counts[:, j] .+ 1]
    end
    return _tp_normalize(logs)
end
function _tp_binomial(k, p)
    if p <= 0 || p >= 1
        result = zeros(k+1)
        result[p <= 0 ? 1 : end] = 1
        return result
    end
    logs = zeros(k+1)
    odds = log(p) - log1p(-p)
    for j = 0:(k-1)
        logs[j+2] = logs[j+1] + log(k-j) - log(j+1) + odds
    end
    return _tp_normalize(logs)
end
function _tp_conditional(counts, p, coeff)
    length(p) == 2 && return _tp_binomial(sum(counts[1, :]), p[2])
    return _tp_normalize([
        coeff[i] + sum(n == 0 ? 0.0 : n * log(p[j]) for (j, n) in enumerate(row)) for
        (i, row) in enumerate(eachrow(counts))
    ])
end
function _tp_beta_quadrature(k, a, b)
    total = a + b
    diagonal = zeros(k + 1)
    diagonal[1] = a / total
    for n = 1:k
        diagonal[n+1] =
            ((n+a)/(2n+total)) * ((n+total-1)/(2n+total-1)) +
            (n/(2n+total-1)) * ((n+b-1)/(2n+total-2))
    end
    off = zeros(k)
    off[1] = sqrt((a/total) * (b/total) / (total+1))
    for n = 2:k
        off[n] = sqrt(
            (n/(2n+total-2)) *
            ((n+a-1)/(2n+total-2)) *
            ((n+b-1)/(2n+total-1)) *
            ((n+total-2)/(2n+total-3)),
        )
    end
    solution = eigen(SymTridiagonal(diagonal, off))
    weights = solution.vectors[1, :] .^ 2
    return clamp.(solution.values, 0, 1), weights ./ sum(weights)
end
function _tp_covariance(counts, alpha, values, coeff)
    size = Base.size(counts, 1)
    (length(alpha) == 2 || max(size, Base.size(values, 2))^2 <= _TP_MAX_PAIRS) ||
        error("Exact covariance exceeds 4000000 count pairs; use shared posterior draws")
    if length(alpha) == 2
        reflect = alpha[2] > alpha[1]
        nodes, weights = _tp_beta_quadrature(
            sum(counts[1, :]),
            alpha[reflect ? 1 : 2],
            alpha[reflect ? 2 : 1],
        )
        v = reflect ? reverse(values; dims = 1) : values
        v = v .- v[1:1, :]
        evaluated = zeros(length(nodes), Base.size(values, 2))
        for (i, p) in enumerate(nodes)
            evaluated[i, :] = vec(_tp_binomial(sum(counts[1, :]), p)' * v)
        end
        centered = evaluated .- (weights' * evaluated)
        return centered' * (centered .* weights)
    end
    pmf = _tp_predictive(counts, alpha, coeff)
    centered = values .- (pmf' * values)
    cov = zeros(Base.size(values, 2), Base.size(values, 2))
    for i = 1:size
        conditional = _tp_predictive(counts, alpha .+ counts[i, :], coeff)
        cov .+= pmf[i] .* (centered[i, :] * (conditional' * centered))
    end
    return (cov + cov') / 2
end
function _tp_endpoint(k, a, b, discovery = false)
    discovery && ((a, b) = (b, a))
    log_mean, log_second, log_ratio = 0.0, 0.0, 0.0
    for j = 0:(2k-1)
        complement = b/(a+b+j)
        term = complement <= 0.5 ? log1p(-complement) : log(a+j)-log(a+b+j)
        isfinite(term) || (term = log(a+j) - log(a+b+j))
        log_second += term
        if j < k
            log_mean += term
            log_ratio += log1p((k/(a+j)) * (b/(a+b+k+j)))
        end
    end
    return discovery ? -expm1(log_mean) : exp(log_mean),
    exp(log_second) * -expm1(-log_ratio)
end
function _tp_integer_terms(k, power)
    power == 1 && return [(1.0, [1])]
    power == 2 && return [(1.0/k, [2]), ((k-1)/k, [1, 1])]
    terms = [(1.0/Float64(k)^3, [4])]
    k >= 2 && append!(terms, [(4(k-1)/Float64(k)^3, [3, 1]), (3(k-1)/Float64(k)^3, [2, 2])])
    k >= 3 && push!(terms, (6*((k-1)/k)*((k-2)/k)/k, [2, 1, 1]))
    k >= 4 && push!(terms, (((k-1)/k)*((k-2)/k)*((k-3)/k), [1, 1, 1, 1]))
    return terms
end
_tp_conditional_moment(p, scores, k, power) = sum(
    c * prod(dot(p, scores .^ f) for f in factors) for
    (c, factors) in _tp_integer_terms(k, power)
)
function _tp_polynomial_moments(scores, k, power, alpha)
    binomial(big(length(scores))+power, power) <= _TP_MAX_STATES ||
        error("Too many rubric categories for exact moment expansion")
    terms = Dict{Tuple,Float64}()
    for (coefficient, factors) in _tp_integer_terms(k, power)
        current = Dict{Tuple,Float64}(Tuple(zeros(Int, length(scores))) => coefficient)
        for factor in factors
            next = Dict{Tuple,Float64}()
            for (powers, value) in current, j in eachindex(scores)
                scores[j] == 0 && continue
                updated = collect(powers)
                updated[j] += 1
                key = Tuple(updated)
                next[key] = get(next, key, 0.0) + value * scores[j]^factor
            end
            current = next
        end
        for (powers, value) in current
            terms[powers] = get(terms, powers, 0.0) + value
        end
    end
    length(terms)^2 <= _TP_MAX_PAIRS ||
        error("Too many moment terms for exact variance; use posterior draws")
    isempty(terms) && return (0.0, 0.0)
    powers, coeff = collect(keys(terms)), collect(values(terms))
    total = sum(alpha)
    function monomial(v)
        result, used = 1.0, 0
        for (j, power) in enumerate(v), n = 0:(power-1)
            result *= (alpha[j]+n)/(total+used)
            used += 1
        end
        return result
    end
    first = monomial.(powers)
    mean = dot(coeff, first)
    var = sum(
        coeff[i]*coeff[j]*(monomial(powers[i] .+ powers[j])-first[i]*first[j]) for
        i in eachindex(powers), j in eachindex(powers)
    )
    return mean, _tp_variance(var)
end

function _tp_binary_moments(k, a, b, values)
    counts = hcat(k .- (0:k), collect(0:k))
    pmf = _tp_beta_binomial(k, a, b)
    return vec(pmf' * values), _tp_covariance(counts, [b, a], values, Float64[])
end

function _tp_scaled_rewards(values)
    offset = values[1]
    centered = values .- offset
    if !all(isfinite, centered)
        offset = 0.0
        centered = copy(values)
    end
    scale = maximum(abs, centered)
    return offset, scale, scale == 0 ? zeros(length(values)) : centered ./ scale
end
function _tp_log_beta_power(a, b, k)
    return sum(
        begin
            denominator = a+b+j
            complement = b/denominator
            complement <= 0.5 ? log1p(-complement) : log(a+j)-log(denominator)
        end for j = 0:(k-1);
        init = 0.0,
    )
end
function _tp_log_difference(log_large, log_small)
    difference = log_small-log_large
    difference <= 64eps(Float64)*max(1, abs(log_large), abs(log_small)) ||
        error("Posterior covariance is materially negative")
    return exp(log_large)*(-expm1(min(difference, 0)))
end
