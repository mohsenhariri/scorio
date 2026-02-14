"""Sequential inference public API."""

include("sinf/core.jl")

export ci_from_mu_sigma,
    should_stop,
    should_stop_top1,
    suggest_next_allocation
