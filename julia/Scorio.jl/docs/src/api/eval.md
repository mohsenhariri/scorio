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
unanimous_at_k
```

## Pass Family (Posterior + CI)

```@docs
pass_at_k_ci
pass_hat_k_ci
g_pass_at_k_ci
g_pass_at_k_tau_ci
mg_pass_at_k_ci
unanimous_at_k_ci
```

## AUC and Majority Families

```@docs
auc_at_k
auc_at_k_ci
maj_at_k
maj_at_k_ci
```

## Max-Reward and Threshold Spectrum

```@docs
max_at_k
max_at_k_ci
threshold_spectrum_at_k
threshold_spectrum_at_k_ci
```

## Geometric Families

```@docs
geom_at_k
geom_at_k_ci
geom_ds_at_k
geom_ds_at_k_ci
geo_spectrum_at_k
geo_spectrum_at_k_ci
geo_spectrum_star_at_k
geo_spectrum_star_at_k_ci
```

## TailPass profiles and utilities

```julia
using Scorio
const E = Scorio.Eval

profile = E.tailpass([0 1 1; 1 1 1], 8)
mu, sigma = E.linear(profile, E.TailPassWeights.moment_weights(8, 2))
draws = E.sample(profile, 4000; rng=42)
qrs_mean, qrs_std, lo, hi = E.summary(draws, E.qrs(draws))
```

`tailpass(R, k; w, R0, eta=1, prior=1, thresholds)` returns a reusable posterior
profile with `mean`, `question_mean`, `covariance`, and `std` properties.
The posterior is `prior + counts(R) + eta * counts(R0)`. Rubric scores lie in
[0, 1]; binary concentrations follow failure, success order. Equal scores merge
by summing concentrations. Reporting budgets may exceed the observed trials.

Julia uses functions on profiles and draws: `linear`, `moment`, `discovery`,
`stability`, and `at_k`. `moment(profile, lam)` integrates the full attainable
score distribution for categorical rubrics. `tailpass_empirical` provides the
finite-bank counterpart, requiring `k <= N`.

`sample(profile, n_draws; rng)` produces shared latent probability draws.
`power_mean`, `qrs`, `rollout`, `harmonic`, and `shortfall` transform each question
before averaging. `power_mean` and `qrs` also accept `aggregation="profile"`.
`summary(draws, values; confidence=0.95)` returns mean, sample standard deviation,
and pointwise equal-tailed intervals. `ci`, `linear_ci`, and `moment_ci` accept
`method="mc"` (default) or `method="normal"`; normal intervals use exact moments.
These describe latent expected performance, not future sampled banks.
Integer seeds are reproducible within Julia; random streams differ across languages.

`TailPassWeights` (also `tailpass_weights`) supplies `uniform_weights`,
`threshold_weights`, `discovery_weights`, `stability_weights`, `moment_weights`,
`beta_weights`, `maxent_weights`, and `payoff_weights`.

Exact count enumeration is limited to 20,000 states; exact profile covariance
is limited to 4,000,000 count/threshold pairs. First, second, and fourth moments
and discovery/stability avoid count enumeration. Budgets above 20,000 require
explicit thresholds. Shared draws can estimate covariance when exact covariance
is too large but the profile is feasible.

```@docs
tailpass
tailpass_empirical
TailPassProfile
TailPassDraws
```
