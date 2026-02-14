using Test
using Scorio

const RUN_SLOW = lowercase(get(ENV, "SCORIO_JL_RUN_SLOW", "0")) in ("1", "true", "yes")

include("helpers.jl")
include("test_eval_ranking.jl")
include("test_rank_public_api.jl")
include("test_priors.jl")
include("test_pointwise_pairwise.jl")
include("test_bradley_terry.jl")
include("test_bayesian_methods.jl")
include("test_voting_methods.jl")
include("test_graph_seriation_hodge.jl")
include("test_listwise.jl")
include("test_irt_fast.jl")

if RUN_SLOW
    include("test_rank_slow.jl")
end
