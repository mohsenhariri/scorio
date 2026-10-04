"""Convex threshold weights for TailPass. Use `moment(profile, lam)` for categorical moments."""
module TailPassWeights
using SpecialFunctions: beta_inc
using ..Scorio: _tp_integer, _tp_positive, _tp_vector

function _validate_weights(weights, size)
    v = _tp_vector(weights, "weights")
    length(v) == size && all(v .>= 0) && abs(sum(v)-1) <= 1e-12 ||
        error("weights must match thresholds, be nonnegative, and sum to one")
    return v
end
"""Equal mass on each of the k thresholds."""
uniform_weights(k) = fill(1.0/_tp_integer(k), k)
"""All value on reaching `threshold` successes, from 1 to k."""
function threshold_weights(k, threshold)
    _tp_integer(k)
    _tp_integer(threshold, "threshold")
    threshold <= k || error("threshold must be <= k")
    w = zeros(k)
    w[threshold] = 1
    return w
end
discovery_weights(k) = threshold_weights(k, 1)
stability_weights(k) = threshold_weights(k, k)
"""Marginal payoff increments `(t/k)^lam - ((t-1)/k)^lam`."""
function moment_weights(k, lam)
    _tp_integer(k)
    lam = _tp_positive(lam, "lam")
    w = [exp(lam*log(t/k)) * -expm1(lam*log1p(-1.0/t)) for t = 1:k]
    return w ./ sum(w)
end
"""Bin Beta threshold mass by mean theta and concentration kappa."""
function beta_weights(k, theta, kappa)
    _tp_integer(k)
    theta = _tp_positive(theta, "theta")
    kappa = _tp_positive(kappa, "kappa")
    theta < 1 || error("theta must be in (0, 1)")
    a, b = theta*kappa, (1-theta)*kappa
    _tp_positive(a, "Beta alpha")
    _tp_positive(b, "Beta beta")
    pairs = [beta_inc(a, b, t/k) for t = 0:k]
    w = [
        pairs[t+1][1] <= 0.5 ? pairs[t+1][1]-pairs[t][1] : pairs[t][2]-pairs[t+1][2] for
        t = 1:k
    ]
    all(isfinite, w) && all(w .>= 0) || error("Could not compute finite Beta weights")
    return w ./ sum(w)
end
"""Integrate the continuous maximum-entropy threshold density on [0, 1]."""
function maxent_weights(k, mean_threshold)
    _tp_integer(k)
    target = _tp_positive(mean_threshold, "mean_threshold")
    target < 1 || error("mean_threshold must be in (0, 1)")
    target == 0.5 && return uniform_weights(k)
    distance = min(target, 1-target)
    lo, hi = 0.0, 1.0/distance
    isfinite(hi) || return target < 0.5 ? discovery_weights(k) : stability_weights(k)
    for _ = 1:160
        tau = lo + (hi-lo)/2
        mean = tau < 1e-3 ? 0.5-tau/12+tau^3/720 : 1/tau-exp(-tau)/(-expm1(-tau))
        mean > distance ? (lo = tau) : (hi = tau)
        hi-lo <= eps(Float64)*max(1, hi) && break
    end
    tau = (lo+hi)/2
    tau == 0 && return uniform_weights(k)
    w = [exp(-tau*(t/k)) * (-expm1(-tau/k))/(-expm1(-tau)) for t = 0:(k-1)]
    w ./= sum(w)
    return target < 0.5 ? w : reverse(w)
end
"""Convert monotone u(0),...,u(k), with endpoints zero and one, to weights."""
function payoff_weights(payoff)
    v = _tp_vector(payoff, "payoff")
    length(v) >= 2 && v[1] == 0 && v[end] == 1 && all(diff(v) .>= 0) ||
        error("payoff must be nondecreasing, start at zero, and end at one")
    return diff(v)
end
export uniform_weights,
    threshold_weights,
    discovery_weights,
    stability_weights,
    moment_weights,
    beta_weights,
    maxent_weights,
    payoff_weights
end
const tailpass_weights = TailPassWeights
