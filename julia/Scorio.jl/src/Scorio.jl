"""
    Scorio

A Julia package for Bayesian evaluation and ranking of LLMs.

Scorio provides:
- Evaluation metrics on `(M, N)` outcome matrices (for example `bayes`, `pass_at_k`)
- Ranking methods on `(L, M, N)` response tensors (for example paired-comparison, IRT, graph, and voting families)
- Utility helpers for converting scores to ranks (`competition_ranks_from_scores`, `rank_scores`)

For full API coverage, see the generated docs pages:
- `API Reference`
- `Examples`

# Quickstart

```julia
using Scorio

# Evaluation example (M x N)
R_eval = [0 1 2 2 1;
          1 1 0 2 2]
w = [0.0, 0.5, 1.0]
mu, sigma = bayes(R_eval, w)

# Ranking example (L x M x N)
R_rank = reshape([
    1 1 0 1 0;
    1 0 0 1 0;
    0 1 0 1 1;
    0 0 0 1 0
], 4, 5, 1)

# Ranking functions are available under the Scorio module namespace.
ranks, scores = Scorio.bradley_terry(R_rank; return_scores=true)
```

# Installation

```julia
using Pkg
Pkg.add("Scorio")
```
"""
module Scorio

# Version
const VERSION = v"0.2.0"

# Include submodules
include("eval.jl")
include("rank.jl")
include("utils.jl")

# Re-export main APIs
export bayes, avg, pass_at_k, pass_hat_k, g_pass_at_k, g_pass_at_k_tau, mg_pass_at_k  # from eval.jl
export elo  # from rank.jl
export competition_ranks_from_scores, rank_scores  # from utils.jl

end # module Scorio
