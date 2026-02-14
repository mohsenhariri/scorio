"""
    bayes(R::AbstractMatrix{<:Integer}, w::Union{AbstractVector{<:Real}, Nothing}=nothing, R0::Union{AbstractMatrix{<:Integer}, Nothing}=nothing) -> Tuple{Float64, Float64}

Performance evaluation using the Bayes@N framework.

# References
Hariri, M., Samandar, A., Hinczewski, M., & Chaudhary, V. (2026).
Don't Pass@k: A Bayesian Framework for Large Language Model Evaluation.
*arXiv preprint arXiv:2510.04265*.
https://arxiv.org/abs/2510.04265

# Arguments
- `R::AbstractMatrix{<:Integer}`: ``M \\times N`` int matrix with entries in ``\\{0,\\ldots,C\\}``.
  Row ``\\alpha`` are the N outcomes for question ``\\alpha``.
- `w::Union{AbstractVector{<:Real}, Nothing}`: length ``(C+1)`` weight vector ``(w_0,\\ldots,w_C)``
  that maps category k to score ``w_k``. If not provided and R is binary (contains only 0 and 1),
  defaults to `[0.0, 1.0]`. For non-binary R, w is required.
- `R0::Union{AbstractMatrix{<:Integer}, Nothing}`: optional ``M \\times D`` int matrix supplying
  D prior outcomes per row. If omitted, ``D=0``.

# Returns
- `Tuple{Float64, Float64}`: ``(\\mu, \\sigma)`` performance metric estimate and its uncertainty.

# Notation
``\\delta_{a,b}`` is the Kronecker delta. For each row ``\\alpha`` and class ``k \\in \\{0,\\ldots,C\\}``:

```math
n_{\\alpha k} = \\sum_{i=1}^N \\delta_{k, R_{\\alpha i}} \\quad \\text{(counts in R)}
```

```math
n^0_{\\alpha k} = 1 + \\sum_{i=1}^D \\delta_{k, R^0_{\\alpha i}} \\quad \\text{(Dirichlet(+1) prior)}
```

```math
\\nu_{\\alpha k} = n_{\\alpha k} + n^0_{\\alpha k}
```

Effective sample size: ``T = 1 + C + D + N`` (scalar)

# Formula

```math
\\mu = w_0 + \\frac{1}{M \\cdot T} \\sum_{\\alpha=1}^M \\sum_{j=0}^C \\nu_{\\alpha j} (w_j - w_0)
```

```math
\\sigma = \\sqrt{ \\frac{1}{M^2(T+1)} \\sum_{\\alpha=1}^M \\left[
    \\sum_j \\frac{\\nu_{\\alpha j}}{T} (w_j - w_0)^2
    - \\left( \\sum_j \\frac{\\nu_{\\alpha j}}{T} (w_j - w_0) \\right)^2 \\right] }
```

# Examples
```julia
R = [0 1 2 2 1;
     1 1 0 2 2]
w = [0.0, 0.5, 1.0]
R0 = [0 2;
      1 2]

# With prior (D=2 → T=10)
mu, sigma = bayes(R, w, R0)
# Expected: mu ≈ 0.575, sigma ≈ 0.084275

# Without prior (D=0 → T=8)
mu2, sigma2 = bayes(R, w)
# Expected: mu2 ≈ 0.5625, sigma2 ≈ 0.091998
```
"""
function bayes(
    R::AbstractMatrix{<:Integer},
    w::Union{AbstractVector{<:Real}, Nothing}=nothing,
    R0::Union{AbstractMatrix{<:Integer}, Nothing}=nothing
)::Tuple{Float64, Float64}

    M, N = size(R)

    # Auto-detect binary matrix and set default w if not provided
    if isnothing(w)
        unique_vals = unique(R)
        is_binary = length(unique_vals) <= 2 && all(v -> v in [0, 1], unique_vals)

        if is_binary
            w = [0.0, 1.0]
        else
            unique_str = join(sort(unique_vals), ", ")
            error("R contains more than 2 unique values ($unique_str), so weight vector 'w' must be provided. " *
                  "Please specify a weight vector of length $(length(unique_vals)) to map each category to a score.")
        end
    end

    C = length(w) - 1
    
    # Handle R0 (prior outcomes)
    if isnothing(R0)
        D = 0
        R0m = zeros(Int, M, 0)
    else
        R0m = R0
        if size(R0m, 1) != M
            error("R0 must have the same number of rows (M) as R.")
        end
        D = size(R0m, 2)
    end
    
    # Validate value ranges
    if !isempty(R) && (minimum(R) < 0 || maximum(R) > C)
        error("Entries of R must be integers in [0, C].")
    end
    if !isempty(R0m) && (minimum(R0m) < 0 || maximum(R0m) > C)
        error("Entries of R0 must be integers in [0, C].")
    end
    
    T = 1 + C + D + N
    
    # Helper function to count occurrences of 0..C in each row
    function row_bincount(A::AbstractMatrix{<:Integer}, num_classes::Int)
        M_local, N_local = size(A)
        if N_local == 0
            return zeros(Int, M_local, num_classes)
        end
        out = zeros(Int, M_local, num_classes)
        for i in 1:M_local
            for j in 1:N_local
                out[i, A[i,j]+1] += 1  # Julia is 1-indexed
            end
        end
        return out
    end
    
    # n_{αk} and n^0_{αk}
    n_counts = row_bincount(R, C + 1)
    n0_counts = row_bincount(R0m, C + 1) .+ 1  # add 1 to every class (Dirichlet prior)
    
    # ν_{αk} = n_{αk} + n^0_{αk}
    nu = n_counts .+ n0_counts  # shape: (M, C+1)
    
    # μ = w0 + (1/(M T)) * Σ_α Σ_j ν_{αj} (w_j - w0)
    delta_w = w .- w[1]
    mu = w[1] + sum(nu * delta_w) / (M * T)
    
    # σ = [ (1/(M^2 (T+1))) * Σ_α { Σ_j (ν_{αj}/T)(w_j-w0)^2
    #       - ( Σ_j (ν_{αj}/T)(w_j-w0) )^2 } ]^{1/2}
    nu_over_T = nu ./ T
    termA = sum(nu_over_T .* (delta_w' .^ 2), dims=2)
    termB = (nu_over_T * delta_w) .^ 2
    sigma = sqrt(sum(termA .- termB) / (M^2 * (T + 1)))
    
    return Float64(mu), Float64(sigma)
end


"""
    avg(R::AbstractArray{<:Real}) -> Float64

Simple average of all entries in R.

Computes the arithmetic mean of all entries in the result matrix.

# Arguments
- `R::AbstractArray{<:Real}`: ``M \\times N`` result matrix with entries in ``\\{0, 1\\}``.
  Row ``\\alpha`` are the N outcomes for question ``\\alpha``.

# Returns
- `Float64`: The arithmetic mean of all entries in R.

# Notation
``R_{\\alpha i}`` is the outcome for question ``\\alpha`` on trial ``i``.

# Formula

```math
\\text{avg} = \\frac{1}{M \\cdot N} \\sum_{\\alpha=1}^{M} \\sum_{i=1}^{N} R_{\\alpha i}
```

# Examples
```julia
R = [0 1 1 0 1;
     1 1 0 1 1]
avg(R)  # Returns 0.7
```
"""
function avg(R::AbstractArray{<:Real})::Float64
    return Float64(sum(R) / length(R))
end


"""
    pass_at_k(R::AbstractMatrix{<:Integer}, k::Integer) -> Float64

Unbiased Pass@k estimator.

Computes the probability that at least one of k randomly selected samples
is correct, averaged over all M questions.

# References
Chen, M., Tworek, J., Jun, H., et al. (2021).
Evaluating Large Language Models Trained on Code.
*arXiv preprint arXiv:2107.03374*.
https://arxiv.org/abs/2107.03374

# Arguments
- `R::AbstractMatrix{<:Integer}`: ``M \\times N`` binary matrix with entries in ``\\{0, 1\\}``.
  ``R_{\\alpha i} = 1`` if trial ``i`` for question ``\\alpha`` passed, 0 otherwise.
- `k::Integer`: Number of samples to select (``1 \\le k \\le N``).

# Returns
- `Float64`: The average Pass@k score across all M questions.

# Notation
For each row ``\\alpha``:

```math
\\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}
```

``C(a, b)`` denotes the binomial coefficient ``\\binom{a}{b}``.

# Formula

```math
\\text{Pass@k}_\\alpha = 1 - \\frac{C(N - \\nu_\\alpha, k)}{C(N, k)}
```

```math
\\text{Pass@k} = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\text{Pass@k}_\\alpha
```

# Examples
```julia
R = [0 1 1 0 1;
     1 1 0 1 1]
pass_at_k(R, 1)  # Returns 0.7
pass_at_k(R, 2)  # Returns 0.95
```
"""
function pass_at_k(R::AbstractMatrix{<:Integer}, k::Integer)::Float64
    M, N = size(R)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end
    
    nu = vec(sum(R, dims=2))
    denom = binomial(BigInt(N), k)
    
    # vals = 1 - comb(N - nu, k) / denom
    # We need to handle element-wise operations. 
    # Note: binomial(n, k) returns 0 if k > n, which handles the case where N - nu < k correctly.
    # Use BigInt to avoid overflow for large N
    vals = [1.0 - float(binomial(BigInt(N - n), k)) / denom for n in nu]
    
    return sum(vals) / M
end


"""
    pass_hat_k(R::AbstractMatrix{<:Integer}, k::Integer) -> Float64

Pass^k (Pass-hat@k): probability that all k selected trials are correct.

Computes the probability that k randomly selected samples are ALL correct,
averaged over all M questions. Also known as G-Pass@k.

# References
Yao, S., Shinn, N., Razavi, P., & Narasimhan, K. (2024).
τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
*arXiv preprint arXiv:2406.12045*.
https://arxiv.org/abs/2406.12045

# Arguments
- `R::AbstractMatrix{<:Integer}`: ``M \\times N`` binary matrix with entries in ``\\{0, 1\\}``.
  ``R_{\\alpha i} = 1`` if trial ``i`` for question ``\\alpha`` passed, 0 otherwise.
- `k::Integer`: Number of samples to select (``1 \\le k \\le N``).

# Returns
- `Float64`: The average Pass^k score across all M questions.

# Notation
For each row ``\\alpha``:

```math
\\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}
```

``C(a, b)`` denotes the binomial coefficient ``\\binom{a}{b}``.

# Formula

```math
\\text{Pass}\\hat{\\text{@}}\\text{k}_\\alpha = \\frac{C(\\nu_\\alpha, k)}{C(N, k)}
```

```math
\\text{Pass}\\hat{\\text{@}}\\text{k} = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\text{Pass}\\hat{\\text{@}}\\text{k}_\\alpha
```

# Examples
```julia
R = [0 1 1 0 1;
     1 1 0 1 1]
pass_hat_k(R, 1)  # Returns 0.7
pass_hat_k(R, 2)  # Returns 0.45
```
"""
function pass_hat_k(R::AbstractMatrix{<:Integer}, k::Integer)::Float64
    M, N = size(R)
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end
    
    nu = vec(sum(R, dims=2))
    denom = binomial(BigInt(N), k)
    
    # vals = comb(nu, k) / denom
    # Use BigInt to avoid overflow for large N
    vals = [float(binomial(BigInt(n), k)) / denom for n in nu]
    
    return sum(vals) / M
end


"""
    g_pass_at_k(R::AbstractMatrix{<:Integer}, k::Integer) -> Float64

Alias for `pass_hat_k`. See [`pass_hat_k`](@ref) for documentation.

This function is provided for compatibility with literature that uses
the G-Pass@k naming convention.
"""
function g_pass_at_k(R::AbstractMatrix{<:Integer}, k::Integer)::Float64
    return pass_hat_k(R, k)
end


"""
    g_pass_at_k_tau(R::AbstractMatrix{<:Integer}, k::Integer, tau::Real) -> Float64

G-Pass@k_τ: Generalized Pass@k with threshold τ.

Computes the probability that at least ``\\lceil \\tau \\cdot k \\rceil`` of k randomly selected
samples are correct, averaged over all M questions.

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv preprint arXiv:2412.13147*.
https://arxiv.org/abs/2412.13147

# Arguments
- `R::AbstractMatrix{<:Integer}`: ``M \\times N`` binary matrix with entries in ``\\{0, 1\\}``.
  ``R_{\\alpha i} = 1`` if trial ``i`` for question ``\\alpha`` passed, 0 otherwise.
- `k::Integer`: Number of samples to select (``1 \\le k \\le N``).
- `tau::Real`: Threshold parameter ``\\tau \\in [0, 1]``. Requires at least
  ``\\lceil \\tau \\cdot k \\rceil`` successes.
  When ``\\tau = 0``, equivalent to Pass@k.
  When ``\\tau = 1``, equivalent to Pass^k.

# Returns
- `Float64`: The average G-Pass@k_τ score across all M questions.

# Notation
For each row ``\\alpha``:

```math
\\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}
```

``C(a, b)`` denotes the binomial coefficient ``\\binom{a}{b}``.

``j_0 = \\lceil \\tau \\cdot k \\rceil`` is the minimum number of successes required.

# Formula

```math
\\text{G-Pass@k}_{\\tau, \\alpha} = \\sum_{j=j_0}^{k} \\frac{C(\\nu_\\alpha, j) \\cdot C(N - \\nu_\\alpha, k - j)}{C(N, k)}
```

```math
\\text{G-Pass@k}_\\tau = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\text{G-Pass@k}_{\\tau, \\alpha}
```

# Examples
```julia
R = [0 1 1 0 1;
     1 1 0 1 1]
g_pass_at_k_tau(R, 2, 0.5)  # Returns ≈ 0.95
g_pass_at_k_tau(R, 2, 1.0)  # Returns ≈ 0.45
```
"""
function g_pass_at_k_tau(R::AbstractMatrix{<:Integer}, k::Integer, tau::Real)::Float64
    M, N = size(R)
    
    if !(0.0 <= tau <= 1.0)
        error("tau must be in [0, 1]; got $tau")
    end
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end
    
    # Edge case: if tau -> 0, return pass_at_k(R, k)
    if tau <= 0.0
        return pass_at_k(R, k)
    end
    
    nu = vec(sum(R, dims=2))
    denom = binomial(BigInt(N), k)
    
    j0 = Int(ceil(tau * k))
    if j0 > k
        return 0.0
    end
    
    vals = zeros(Float64, M)
    for j in j0:k
        for (idx, n) in enumerate(nu)
            vals[idx] += float(binomial(BigInt(n), j) * binomial(BigInt(N - n), k - j)) / denom
        end
    end
    
    return sum(vals) / M
end


"""
    mg_pass_at_k(R::AbstractMatrix{<:Integer}, k::Integer) -> Float64

mG-Pass@k: mean Generalized Pass@k.

Computes the mean of G-Pass@k_τ over the range τ ∈ [0.5, 1.0], inspired by the
mean Average Precision (mAP) metric. This provides a comprehensive metric that
integrates performance potential and stability across multiple thresholds.

# References
Liu, J., Liu, H., Xiao, L., et al. (2024).
Are Your LLMs Capable of Stable Reasoning?
*arXiv preprint arXiv:2412.13147*.
https://arxiv.org/abs/2412.13147

# Arguments
- `R::AbstractMatrix{<:Integer}`: ``M \\times N`` binary matrix with entries in ``\\{0, 1\\}``.
  ``R_{\\alpha i} = 1`` if trial ``i`` for question ``\\alpha`` passed, 0 otherwise.
- `k::Integer`: Number of samples to select (``1 \\le k \\le N``).

# Returns
- `Float64`: The average mG-Pass@k score across all M questions.

# Notation
For each row ``\\alpha``:

```math
\\nu_\\alpha = \\sum_{i=1}^{N} R_{\\alpha i} \\quad \\text{(number of correct samples)}
```

``m = \\lceil k/2 \\rceil`` is the majority threshold (the integration starts at ``\\tau = 0.5``).

The metric is defined as the integral of G-Pass@k_τ over τ ∈ [0.5, 1.0]:

```math
\\text{mG-Pass@k} = 2 \\int_{0.5}^{1.0} \\text{G-Pass@k}_\\tau \\, d\\tau
```

# Formula
The discrete approximation used in computation:

```math
\\text{mG-Pass@k}_\\alpha = \\frac{2}{k} \\sum_{j=m+1}^{k} (j - m) \\cdot P(X = j)
```

where ``X \\sim \\text{Hypergeometric}(N, \\nu_\\alpha, k)`` and the probability mass function is:

```math
P(X = j) = \\frac{C(\\nu_\\alpha, j) \\cdot C(N - \\nu_\\alpha, k - j)}{C(N, k)}
```

The final metric is averaged over all questions:

```math
\\text{mG-Pass@k} = \\frac{1}{M} \\sum_{\\alpha=1}^{M} \\text{mG-Pass@k}_\\alpha
```

# Examples
```julia
R = [0 1 1 0 1;
     1 1 0 1 1]
mg_pass_at_k(R, 2)  # Returns ≈ 0.45
mg_pass_at_k(R, 3)  # Returns ≈ 0.166667
```
"""
function mg_pass_at_k(R::AbstractMatrix{<:Integer}, k::Integer)::Float64
    M, N = size(R)
    
    if !(1 <= k <= N)
        error("k must satisfy 1 <= k <= N (N=$N); got k=$k")
    end
    
    nu = vec(sum(R, dims=2))
    denom = binomial(BigInt(N), k)
    
    majority = Int(ceil(0.5 * k))
    if majority >= k
        return 0.0
    end
    
    vals = zeros(Float64, M)
    # mG per-question = (2/k) * E[(X - majority)_+], X ~ Hypergeom(N, nu, k)
    for j in (majority + 1):k
        for (idx, n) in enumerate(nu)
            pmf = float(binomial(BigInt(n), j) * binomial(BigInt(N - n), k - j)) / denom
            vals[idx] += (j - majority) * pmf
        end
    end
    
    vals .*= 2.0 / k
    return sum(vals) / M
end


export bayes, avg, pass_at_k, pass_hat_k, g_pass_at_k, g_pass_at_k_tau, mg_pass_at_k
