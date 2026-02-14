# API Reference

## Evaluation

```@docs
bayes
avg(::AbstractArray{<:Real})
pass_at_k
pass_hat_k
g_pass_at_k
g_pass_at_k_tau(::AbstractMatrix{<:Integer}, ::Integer, ::Real)
mg_pass_at_k
```

## Priors

```@docs
Prior
GaussianPrior
LaplacePrior
CauchyPrior
UniformPrior
CustomPrior
EmpiricalPrior
```

## Ranking

```@docs
competition_ranks_from_scores
rank_scores
avg(::Any; method, return_scores)
g_pass_at_k_tau(::AbstractArray{<:Integer, 3}, ::Any, ::Any; method, return_scores)
inverse_difficulty
elo
trueskill
glicko
thompson
bayesian_mcmc
bradley_terry
bradley_terry_map
bradley_terry_davidson
bradley_terry_davidson_map
rao_kupper
rao_kupper_map
rasch
rasch_map
rasch_2pl
rasch_2pl_map
rasch_3pl
rasch_3pl_map
dynamic_irt
rasch_mml
rasch_mml_credible
borda
copeland
win_rate
minimax
schulze
ranked_pairs
kemeny_young
nanson
baldwin
majority_judgment
pagerank
spectral
alpharank
nash
rank_centrality
serial_rank
hodge_rank
plackett_luce
plackett_luce_map
davidson_luce
davidson_luce_map
bradley_terry_luce
bradley_terry_luce_map
```
