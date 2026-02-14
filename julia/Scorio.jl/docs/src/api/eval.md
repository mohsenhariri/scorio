# Evaluation API

Evaluation methods operate on outcome matrices `R` with shape `(M, N)` (or vectors
coerced to `1 x N`).

## Bayes Family

```@docs
bayes
bayes_ci
```

## Avg Family

```@docs
avg
avg_ci
```

## Pass Family (Point Metrics)

```@docs
pass_at_k(::Union{AbstractVector, AbstractMatrix}, ::Integer)
pass_hat_k(::Union{AbstractVector, AbstractMatrix}, ::Integer)
g_pass_at_k(::Union{AbstractVector, AbstractMatrix}, ::Integer)
g_pass_at_k_tau(::Union{AbstractVector, AbstractMatrix}, ::Integer, ::Real)
mg_pass_at_k(::Union{AbstractVector, AbstractMatrix}, ::Integer)
```

## Pass Family (Posterior + CI)

```@docs
pass_at_k_ci
pass_hat_k_ci
g_pass_at_k_ci
g_pass_at_k_tau_ci
mg_pass_at_k_ci
```
