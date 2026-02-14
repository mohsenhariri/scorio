using Test
using Scorio

@testset "Scorio.jl" begin
    @test Scorio.VERSION == v"0.2.0"

    @test isdefined(Scorio, :Eval)
    @test isdefined(Scorio, :Rank)
    @test isdefined(Scorio, :SInf)
    @test isdefined(Scorio, :Utils)

    @test isdefined(Scorio.Eval, :bayes)
    @test isdefined(Scorio.Eval, :pass_at_k)
    @test isdefined(Scorio.Rank, :avg)
    @test isdefined(Scorio.Rank, :bradley_terry)
    @test isdefined(Scorio.SInf, :should_stop)
    @test isdefined(Scorio.Utils, :rank_scores)
end

include("eval/test_eval_apis.jl")
include("rank/test_eval_ranking.jl")
