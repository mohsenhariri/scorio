using Test
using Scorio

@testset "rank/test_listwise.jl" begin
    R = ordered_binary_R()
    R_small = ordered_binary_small_R()

    @testset "listwise smoke and ordering" begin
        calls = [
            () -> Scorio.Rank.plackett_luce(R; max_iter=100, return_scores=true),
            () -> Scorio.Rank.plackett_luce_map(R; prior=1.0, max_iter=100, return_scores=true),
            () -> Scorio.Rank.davidson_luce(R; max_iter=100, return_scores=true),
            () -> Scorio.Rank.davidson_luce_map(R; prior=1.0, max_iter=100, return_scores=true),
            () -> Scorio.Rank.bradley_terry_luce(R; max_iter=100, return_scores=true),
            () -> Scorio.Rank.bradley_terry_luce_map(
                R;
                prior=1.0,
                max_iter=100,
                return_scores=true,
            ),
        ]

        for run in calls
            ranking, _ = assert_ranking_and_scores(run())
            assert_ordering_sanity(ranking; best_idx=1, worst_idx=4)
        end
    end

    @testset "plackett_luce_map prior coercion" begin
        _, scores_float = Scorio.Rank.plackett_luce_map(
            R_small;
            prior=1.0,
            max_iter=80,
            return_scores=true,
        )
        _, scores_object = Scorio.Rank.plackett_luce_map(
            R_small;
            prior=Scorio.Rank.GaussianPrior(0.0, 1.0),
            max_iter=80,
            return_scores=true,
        )

        @test length(scores_float) == length(scores_object)
        @test all(isfinite, scores_float)
        @test all(isfinite, scores_object)
    end

    @testset "validation errors" begin
        L = size(R_small, 1)

        @test_throws ErrorException Scorio.Rank.plackett_luce(R_small; max_iter=0)
        @test_throws ErrorException Scorio.Rank.plackett_luce_map(R_small; prior=0.0)
        @test_throws ErrorException Scorio.Rank.davidson_luce(R_small; max_tie_order=L + 1)
        @test_throws ErrorException Scorio.Rank.bradley_terry_luce_map(R_small; prior="bad")
    end
end
