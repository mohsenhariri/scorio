using Test
using Scorio

@testset "rank/test_pointwise_pairwise.jl" begin
    R = ordered_binary_R()
    R_small = ordered_binary_small_R()
    R_tie = tie_heavy_R()

    @testset "inverse_difficulty" begin
        ranking, scores = assert_ranking_and_scores(
            Scorio.Rank.inverse_difficulty(R; return_scores=true),
        )
        assert_ordering_sanity(ranking; best_idx=1, worst_idx=4)
        @test all(0.0 .<= scores .<= 1.0)

        @test_throws ErrorException Scorio.Rank.inverse_difficulty(
            R;
            clip_range=(0.9, 0.5),
        )
    end

    @testset "elo" begin
        ranking, _ = assert_ranking_and_scores(Scorio.Rank.elo(R; return_scores=true))
        assert_ordering_sanity(ranking; best_idx=1, worst_idx=4)

        _, scores_skip = Scorio.Rank.elo(R_tie; tie_handling="skip", return_scores=true)
        _, scores_draw = Scorio.Rank.elo(R_tie; tie_handling="draw", return_scores=true)
        @test !isapprox(scores_skip, scores_draw)

        @test_throws ErrorException Scorio.Rank.elo(R; K=0.0)
        @test_throws ErrorException Scorio.Rank.elo(R; tie_handling="invalid")
    end

    @testset "glicko" begin
        ranking, ratings, rd = Scorio.Rank.glicko(
            R;
            return_scores=true,
            return_deviation=true,
        )

        assert_ranking(ranking; expected_len=size(R, 1))
        assert_scores(ratings; expected_len=size(R, 1))
        @test length(rd) == size(R, 1)
        @test all(isfinite, rd)
        @test all(rd .> 0.0)
        assert_ordering_sanity(ranking; best_idx=1, worst_idx=4)

        @test_throws ErrorException Scorio.Rank.glicko(R; initial_rd=0.0)
        @test_throws ErrorException Scorio.Rank.glicko(R; tie_handling="invalid")
    end

    @testset "trueskill" begin
        ranking, _ = assert_ranking_and_scores(Scorio.Rank.trueskill(R; return_scores=true))
        assert_ordering_sanity(ranking; best_idx=1, worst_idx=4)

        _, scores_skip = Scorio.Rank.trueskill(
            R_tie;
            tie_handling="skip",
            draw_margin=0.0,
            return_scores=true,
        )
        _, scores_draw = Scorio.Rank.trueskill(
            R_tie;
            tie_handling="draw",
            draw_margin=0.1,
            return_scores=true,
        )
        @test !isapprox(scores_skip, scores_draw)

        @test_throws ErrorException Scorio.Rank.trueskill(R_small; draw_margin=-0.1)
        @test_throws ErrorException Scorio.Rank.trueskill(R_small; tie_handling="invalid")
    end
end
