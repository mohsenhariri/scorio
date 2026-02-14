using Test
using Scorio

@testset "rank/test_voting_methods.jl" begin
    R = ordered_binary_R()
    R_small = ordered_binary_small_R()

    @testset "voting methods smoke and ordering" begin
        calls = [
            () -> Scorio.Rank.borda(R; return_scores=true),
            () -> Scorio.Rank.copeland(R; return_scores=true),
            () -> Scorio.Rank.win_rate(R; return_scores=true),
            () -> Scorio.Rank.minimax(R; variant="margin", tie_policy="half", return_scores=true),
            () -> Scorio.Rank.schulze(R; tie_policy="half", return_scores=true),
            () -> Scorio.Rank.ranked_pairs(
                R;
                strength="margin",
                tie_policy="half",
                return_scores=true,
            ),
            () -> Scorio.Rank.kemeny_young(
                R;
                tie_policy="half",
                time_limit=1.0,
                return_scores=true,
            ),
            () -> Scorio.Rank.nanson(R; rank_ties="average", return_scores=true),
            () -> Scorio.Rank.baldwin(R; rank_ties="average", return_scores=true),
            () -> Scorio.Rank.majority_judgment(R; return_scores=true),
        ]

        for run in calls
            ranking, _ = assert_ranking_and_scores(run())
            assert_ordering_sanity(ranking; best_idx=1, worst_idx=4)
        end
    end

    @testset "option branches" begin
        out_minimax = Scorio.Rank.minimax(
            R_small;
            variant="winning_votes",
            tie_policy="ignore",
            return_scores=true,
        )
        out_ranked_pairs = Scorio.Rank.ranked_pairs(
            R_small;
            strength="winning_votes",
            tie_policy="ignore",
            return_scores=true,
        )
        out_kemeny = Scorio.Rank.kemeny_young(
            R_small;
            tie_policy="ignore",
            tie_aware=false,
            time_limit=1.0,
            return_scores=true,
        )

        assert_ranking_and_scores(out_minimax)
        assert_ranking_and_scores(out_ranked_pairs)
        assert_ranking_and_scores(out_kemeny)
    end

    @testset "validation errors" begin
        @test_throws ErrorException Scorio.Rank.minimax(R_small; variant="bad")
        @test_throws ErrorException Scorio.Rank.ranked_pairs(R_small; strength="bad")
        @test_throws ErrorException Scorio.Rank.schulze(R_small; tie_policy="bad")
        @test_throws ErrorException Scorio.Rank.kemeny_young(R_small; time_limit=0.0)
        @test_throws ErrorException Scorio.Rank.nanson(R_small; rank_ties="bad")
    end
end
