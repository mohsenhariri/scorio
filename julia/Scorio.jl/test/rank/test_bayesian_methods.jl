using Test
using Scorio

@testset "rank/test_bayesian_methods.jl" begin
    R_small = ordered_binary_small_R()
    R_equal = equal_information_R()

    @testset "thompson seed determinism" begin
        out1 = Scorio.Rank.thompson(R_small; n_samples=1500, seed=11, return_scores=true)
        out2 = Scorio.Rank.thompson(R_small; n_samples=1500, seed=11, return_scores=true)

        ranking1, scores1 = assert_ranking_and_scores(out1)
        ranking2, scores2 = assert_ranking_and_scores(out2)

        @test scores1 ≈ scores2
        @test ranking1 ≈ ranking2
    end

    @testset "bayesian_mcmc seed determinism" begin
        out1 = Scorio.Rank.bayesian_mcmc(
            R_small;
            n_samples=800,
            burnin=200,
            seed=13,
            return_scores=true,
        )
        out2 = Scorio.Rank.bayesian_mcmc(
            R_small;
            n_samples=800,
            burnin=200,
            seed=13,
            return_scores=true,
        )

        ranking1, scores1 = assert_ranking_and_scores(out1)
        ranking2, scores2 = assert_ranking_and_scores(out2)

        @test scores1 ≈ scores2
        @test ranking1 ≈ ranking2
    end

    @testset "equal-information behavior" begin
        ranking_ts, scores_ts = Scorio.Rank.thompson(
            R_equal;
            n_samples=3000,
            seed=19,
            return_scores=true,
        )
        @test all(isapprox.(scores_ts, fill(first(scores_ts), length(scores_ts))))
        @test all(isapprox.(ranking_ts, fill(first(ranking_ts), length(ranking_ts))))

        ranking_mcmc, scores_mcmc = Scorio.Rank.bayesian_mcmc(
            R_equal;
            n_samples=700,
            burnin=100,
            seed=19,
            return_scores=true,
        )
        @test all(isapprox.(scores_mcmc, fill(first(scores_mcmc), length(scores_mcmc))))
        @test all(isapprox.(ranking_mcmc, fill(first(ranking_mcmc), length(ranking_mcmc))))
    end

    @testset "validation errors" begin
        @test_throws ErrorException Scorio.Rank.thompson(R_small; n_samples=0)
        @test_throws ErrorException Scorio.Rank.thompson(R_small; prior_alpha=0.0)
        @test_throws ErrorException Scorio.Rank.thompson(R_small; prior_beta=0.0)

        @test_throws ErrorException Scorio.Rank.bayesian_mcmc(R_small; n_samples=0)
        @test_throws ErrorException Scorio.Rank.bayesian_mcmc(R_small; burnin=-1)
        @test_throws ErrorException Scorio.Rank.bayesian_mcmc(R_small; prior_var=0.0)
    end
end
