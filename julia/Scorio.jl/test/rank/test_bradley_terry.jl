using Test
using Scorio

@testset "rank/test_bradley_terry.jl" begin
    R = ordered_binary_R()
    R_tie = tie_heavy_R()

    @testset "BT family smoke and ordering" begin
        calls = [
            () -> Scorio.Rank.bradley_terry(R; max_iter=100, return_scores=true),
            () -> Scorio.Rank.bradley_terry_map(R; prior=1.0, max_iter=100, return_scores=true),
            () -> Scorio.Rank.bradley_terry_davidson(R; max_iter=100, return_scores=true),
            () -> Scorio.Rank.bradley_terry_davidson_map(
                R;
                prior=1.0,
                max_iter=100,
                return_scores=true,
            ),
            () -> Scorio.Rank.rao_kupper(
                R;
                tie_strength=1.1,
                max_iter=100,
                return_scores=true,
            ),
            () -> Scorio.Rank.rao_kupper_map(
                R;
                tie_strength=1.1,
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

    @testset "MAP prior coercion" begin
        _, scores_float = Scorio.Rank.bradley_terry_map(
            R;
            prior=1.0,
            max_iter=80,
            return_scores=true,
        )
        _, scores_object = Scorio.Rank.bradley_terry_map(
            R;
            prior=Scorio.Rank.GaussianPrior(0.0, 1.0),
            max_iter=80,
            return_scores=true,
        )

        @test length(scores_float) == length(scores_object)
        @test all(isfinite, scores_float)
        @test all(isfinite, scores_object)
    end

    @testset "Validation errors" begin
        @test_throws ErrorException Scorio.Rank.bradley_terry(R; max_iter=0)
        @test_throws ErrorException Scorio.Rank.bradley_terry_map(R; prior=-1.0)
        @test_throws ErrorException Scorio.Rank.rao_kupper(R; tie_strength=0.9)
        @test_throws ErrorException Scorio.Rank.rao_kupper(R_tie; tie_strength=1.0)
        @test_throws ErrorException Scorio.Rank.rao_kupper_map(R; prior="bad")
    end
end
