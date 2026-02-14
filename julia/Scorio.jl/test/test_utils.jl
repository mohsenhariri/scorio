using Test
using Scorio

@testset "test_utils.jl" begin

    @testset "RankScores" begin
        @testset "basic competition ranking" begin
            scores = [95.0, 87.5, 87.5, 80.0, 75.0]
            ranks = Scorio.Utils.rank_scores(scores)
            @test ranks["competition"] == [1.0, 2.0, 2.0, 4.0, 5.0]
            @test ranks["competition_max"] == [1.0, 3.0, 3.0, 4.0, 5.0]
            @test ranks["dense"] == [1.0, 2.0, 2.0, 3.0, 4.0]
            @test ranks["avg"] ≈ [1.0, 2.5, 2.5, 4.0, 5.0]
        end

        @testset "no ties" begin
            scores = [10.0, 8.0, 6.0, 4.0, 2.0]
            ranks = Scorio.Utils.rank_scores(scores)
            @test ranks["competition"] == [1.0, 2.0, 3.0, 4.0, 5.0]
            @test ranks["dense"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        end

        @testset "all tied" begin
            scores = [5.0, 5.0, 5.0]
            ranks = Scorio.Utils.rank_scores(scores)
            @test ranks["competition"] == [1.0, 1.0, 1.0]
            @test ranks["competition_max"] == [3.0, 3.0, 3.0]
            @test ranks["dense"] == [1.0, 1.0, 1.0]
            @test ranks["avg"] ≈ [2.0, 2.0, 2.0]
        end

        @testset "higher scores get lower ranks" begin
            scores = [1.0, 3.0, 2.0]
            ranks = Scorio.Utils.rank_scores(scores)
            @test ranks["competition"][2] < ranks["competition"][3] < ranks["competition"][1]
        end

        @testset "tolerance groups near equal scores" begin
            scores = [10.0, 10.0 + 1e-14, 5.0]
            ranks = Scorio.Utils.rank_scores(scores; tol=1e-12)
            @test ranks["competition"][1] == ranks["competition"][2]
        end

        @testset "1d validation" begin
            @test_throws ErrorException Scorio.Utils.rank_scores([1.0 2.0; 3.0 4.0])
        end

        @testset "uncertainty aware ties zscore" begin
            scores = [10.0, 9.5, 5.0]
            sigmas = [1.0, 1.0, 0.1]
            ranks = Scorio.Utils.rank_scores(scores; sigmas_in_id_order=sigmas, confidence=0.95)

            @test haskey(ranks, "competition_ci")
            @test haskey(ranks, "dense_ci")
            @test ranks["dense_ci"][1] == ranks["dense_ci"][2]
            @test ranks["dense_ci"][3] > ranks["dense_ci"][1]
        end

        @testset "uncertainty aware ties ci_overlap" begin
            scores = [10.0, 9.5, 5.0]
            sigmas = [1.0, 1.0, 0.1]
            ranks = Scorio.Utils.rank_scores(scores; sigmas_in_id_order=sigmas, confidence=0.95, ci_tie_method="ci_overlap_adjacent")
            @test haskey(ranks, "competition_ci")
            @test ranks["dense_ci"][1] == ranks["dense_ci"][2]
        end

        @testset "no ci keys without sigmas" begin
            scores = [10.0, 5.0, 1.0]
            ranks = Scorio.Utils.rank_scores(scores)
            @test !haskey(ranks, "competition_ci")
        end

        @testset "sigmas shape mismatch raises" begin
            @test_throws ErrorException Scorio.Utils.rank_scores([1.0, 2.0, 3.0]; sigmas_in_id_order=[0.1, 0.2])
        end

        @testset "unknown ci tie method raises" begin
            @test_throws ErrorException Scorio.Utils.rank_scores([1.0, 2.0]; sigmas_in_id_order=[0.1, 0.1], ci_tie_method="bad")
        end

        @testset "with real Bayes scores" begin
            R = top_p_data()["aime25"]
            L = size(R, 1)
            scores_arr = zeros(Float64, L)
            sigmas_arr = zeros(Float64, L)
            for i in 1:L
                scores_arr[i], sigmas_arr[i] = Scorio.Eval.bayes(R[i, :, :])
            end

            ranks = Scorio.Utils.rank_scores(scores_arr; sigmas_in_id_order=sigmas_arr)

            for key in ["competition", "competition_max", "dense", "avg"]
                @test length(ranks[key]) == L
                @test minimum(ranks[key]) ≈ 1.0
                @test all(ranks[key] .>= 1)
                @test all(ranks[key] .<= L)
            end

            for key in ["competition_ci", "competition_max_ci", "dense_ci", "avg_ci"]
                @test haskey(ranks, key)
                @test length(ranks[key]) == L
            end

            best = argmax(scores_arr)
            @test ranks["competition"][best] == 1.0
        end
    end

    @testset "CompareRankings" begin
        @testset "identical rankings" begin
            a = [1, 2, 3, 4, 5]
            b = [1, 2, 3, 4, 5]
            result = Scorio.Utils.compare_rankings(a, b; method="all")
            tau, _ = result["kendalltau"]
            rho, _ = result["spearmanr"]
            @test tau ≈ 1.0 atol = 1e-10
            @test rho ≈ 1.0 atol = 1e-10
            @test result["fraction_mismatched"] ≈ 0.0 atol = 1e-12
            @test result["max_disp"] ≈ 0.0 atol = 1e-12
        end

        @testset "reversed rankings" begin
            a = [1, 2, 3, 4, 5]
            b = [5, 4, 3, 2, 1]
            result = Scorio.Utils.compare_rankings(a, b; method="all")
            tau, _ = result["kendalltau"]
            rho, _ = result["spearmanr"]
            @test tau ≈ -1.0 atol = 1e-10
            @test rho ≈ -1.0 atol = 1e-10
            @test result["fraction_mismatched"] ≈ 0.8 atol = 1e-12
        end

        @testset "single method kendall" begin
            a = [1, 2, 3, 4, 5]
            b = [1, 3, 2, 4, 5]
            tau, pval = Scorio.Utils.compare_rankings(a, b; method="kendall")
            @test tau ≈ 0.8 atol = 1e-10
            @test 0.0 <= pval <= 1.0
        end

        @testset "single method spearman" begin
            a = [1, 2, 3, 4, 5]
            b = [1, 3, 2, 4, 5]
            rho, pval = Scorio.Utils.compare_rankings(a, b; method="spearman")
            @test isfinite(rho)
            @test 0.0 <= pval <= 1.0
        end

        @testset "single method weighted kendall" begin
            a = [1, 2, 3, 4, 5]
            b = [1, 3, 2, 4, 5]
            wtau, _ = Scorio.Utils.compare_rankings(a, b; method="weighted_kendall")
            @test isfinite(wtau)
        end

        @testset "fraction mismatched manual" begin
            a = [1, 2, 3, 4, 5]
            b = [1, 2, 4, 3, 5]
            result = Scorio.Utils.compare_rankings(a, b; method="all")
            @test result["fraction_mismatched"] ≈ 2.0 / 5.0 atol = 1e-12
        end

        @testset "max displacement" begin
            a = [1, 2, 3, 4, 5]
            b = [5, 2, 3, 4, 1]
            result = Scorio.Utils.compare_rankings(a, b; method="all")
            @test result["max_disp"] ≈ 4.0 / 4.0 atol = 1e-12
        end

        @testset "different length raises" begin
            @test_throws ErrorException Scorio.Utils.compare_rankings([1, 2], [1, 2, 3])
        end

        @testset "empty raises" begin
            @test_throws ErrorException Scorio.Utils.compare_rankings(Float64[], Float64[])
        end

        @testset "nan raises" begin
            @test_throws ErrorException Scorio.Utils.compare_rankings([1.0, NaN], [1.0, 2.0])
        end

        @testset "inf raises" begin
            @test_throws ErrorException Scorio.Utils.compare_rankings([1.0, Inf], [1.0, 2.0])
        end

        @testset "invalid method raises" begin
            @test_throws ErrorException Scorio.Utils.compare_rankings([1, 2], [1, 2]; method="bad")
        end

        @testset "with real rank comparison" begin
            R = top_p_data()["aime25"][1:6, 1:10, 1:12]
            rank_avg = Scorio.Rank.avg(R)
            rank_bayes = Scorio.Rank.bayes(R)

            result = Scorio.Utils.compare_rankings(rank_avg, rank_bayes; method="all")
            tau, _ = result["kendalltau"]
            rho, _ = result["spearmanr"]
            @test tau > 0.5
            @test rho > 0.5
        end
    end

    @testset "LehmerHash" begin
        @testset "identity permutation" begin
            @test Scorio.Utils.lehmer_hash([0, 1, 2]) == 0
        end

        @testset "reverse permutation" begin
            @test Scorio.Utils.lehmer_hash([2, 1, 0]) == 5
        end

        @testset "known values" begin
            @test Scorio.Utils.lehmer_hash([0, 2, 1]) == 1
            @test Scorio.Utils.lehmer_hash([1, 0, 2]) == 2
            @test Scorio.Utils.lehmer_hash([1, 2, 0]) == 3
            @test Scorio.Utils.lehmer_hash([2, 0, 1]) == 4
        end

        @testset "all permutations of 3 are unique" begin
            hashes = Set{BigInt}()
            for p in [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
                h = Scorio.Utils.lehmer_hash(p)
                push!(hashes, h)
            end
            @test hashes == Set(big.(0:5))
        end

        @testset "not permutation raises" begin
            @test_throws ErrorException Scorio.Utils.lehmer_hash([0, 0, 1])
        end
    end

    @testset "LehmerUnhash" begin
        @testset "identity" begin
            @test Scorio.Utils.lehmer_unhash(0, 3) == [0, 1, 2]
        end

        @testset "reverse" begin
            @test Scorio.Utils.lehmer_unhash(5, 3) == [2, 1, 0]
        end

        @testset "known values" begin
            @test Scorio.Utils.lehmer_unhash(1, 3) == [0, 2, 1]
            @test Scorio.Utils.lehmer_unhash(2, 3) == [1, 0, 2]
            @test Scorio.Utils.lehmer_unhash(3, 3) == [1, 2, 0]
            @test Scorio.Utils.lehmer_unhash(4, 3) == [2, 0, 1]
        end

        @testset "roundtrip n3" begin
            for h in 0:5
                perm = Scorio.Utils.lehmer_unhash(h, 3)
                @test Scorio.Utils.lehmer_hash(perm) == h
            end
        end

        @testset "roundtrip n4" begin
            for h in 0:23
                perm = Scorio.Utils.lehmer_unhash(h, 4)
                @test Scorio.Utils.lehmer_hash(perm) == h
            end
        end

        @testset "out of range raises" begin
            @test_throws ErrorException Scorio.Utils.lehmer_unhash(6, 3)
            @test_throws ErrorException Scorio.Utils.lehmer_unhash(-1, 3)
        end
    end

    @testset "OrderedBell" begin
        @testset "known Fubini numbers" begin
            F = Scorio.ordered_bell(6)
            @test F == big.([1, 1, 3, 13, 75, 541, 4683])
        end

        @testset "F0 is 1" begin
            F = Scorio.ordered_bell(0)
            @test F == big.([1])
        end
    end

    @testset "RankingHash" begin
        @testset "no ties" begin
            @test Scorio.ranking_hash([0, 1, 2]) == 0
        end

        @testset "known values n3" begin
            @test Scorio.ranking_hash([0, 1, 1]) == 2
            @test Scorio.ranking_hash([0, 0, 1]) == 9
            @test Scorio.ranking_hash([0, 0, 0]) == 12
            @test Scorio.ranking_hash([1, 0, 1]) == 5
        end

        @testset "all rankings of 3 are unique" begin
            F = Scorio.ordered_bell(3)
            @test F[4] == 13
            hashes = Set{BigInt}()
            for h in 0:(F[4] - 1)
                ranking = Scorio.unhash_ranking(h, 3)
                r0 = [x - 1 for x in ranking]
                computed_h = Scorio.ranking_hash(r0)
                @test computed_h == h
                push!(hashes, computed_h)
            end
            @test length(hashes) == 13
        end

        @testset "roundtrip n3" begin
            F = Scorio.ordered_bell(3)
            for h in 0:(F[4] - 1)
                ranking = Scorio.unhash_ranking(h, 3)
                @test Scorio.ranking_hash([r - 1 for r in ranking]) == h
            end
        end

        @testset "roundtrip n4" begin
            F = Scorio.ordered_bell(4)
            for h in 0:(F[5] - 1)
                ranking = Scorio.unhash_ranking(h, 4)
                @test Scorio.ranking_hash([r - 1 for r in ranking]) == h
            end
        end
    end

    @testset "UnhashRanking" begin
        @testset "first ranking" begin
            @test Scorio.unhash_ranking(0, 3) == [1, 2, 3]
        end

        @testset "known values" begin
            @test Scorio.unhash_ranking(2, 3) == [1, 2, 2]
            @test Scorio.unhash_ranking(9, 3) == [1, 1, 3]
            @test Scorio.unhash_ranking(12, 3) == [1, 1, 1]
            @test Scorio.unhash_ranking(5, 3) == [2, 1, 2]
        end

        @testset "out of range raises" begin
            F = Scorio.ordered_bell(3)
            @test_throws ErrorException Scorio.unhash_ranking(F[4], 3)
            @test_throws ErrorException Scorio.unhash_ranking(-1, 3)
        end
    end

    @testset "CombRankUnrank" begin
        @testset "first combination" begin
            @test Scorio.comb_rank_lex([0, 1], 4, 2) == 0
        end

        @testset "last combination" begin
            @test Scorio.comb_rank_lex([2, 3], 4, 2) == 5
        end

        @testset "roundtrip" begin
            n, k = 5, 3
            total = binomial(n, k)
            for r in 0:(total - 1)
                combo = Scorio.comb_unrank_lex(r, n, k)
                @test Scorio.comb_rank_lex(combo, n, k) == r
            end
        end

        @testset "empty combination" begin
            @test Scorio.comb_unrank_lex(0, 5, 0) == Int[]
        end

        @testset "out of range raises" begin
            @test_throws ErrorException Scorio.comb_unrank_lex(10, 4, 2)
        end
    end

    @testset "BlocksFromRankList" begin
        @testset "no ties" begin
            blocks = Scorio.blocks_from_rank_list([1, 2, 3])
            @test blocks == [[0], [1], [2]]
        end

        @testset "all tied" begin
            blocks = Scorio.blocks_from_rank_list([1, 1, 1])
            @test blocks == [[0, 1, 2]]
        end

        @testset "some ties" begin
            blocks = Scorio.blocks_from_rank_list([1, 2, 2, 4])
            @test blocks == [[0], [1, 2], [3]]
        end

        @testset "empty" begin
            @test Scorio.blocks_from_rank_list(Float64[]) == Vector{Vector{Int}}()
        end

        @testset "reversed order" begin
            blocks = Scorio.blocks_from_rank_list([3, 2, 1])
            @test blocks == [[2], [1], [0]]
        end

        @testset "float tolerance" begin
            blocks = Scorio.blocks_from_rank_list([1.0, 1.0 + 1e-14, 2.0]; tol=1e-12)
            @test length(blocks) == 2
            @test blocks[1] == [0, 1]
        end

        @testset "with real rank output" begin
            R = top_p_data()["aime25"][1:6, 1:10, 1:12]
            ranking = Scorio.Rank.avg(R)
            blocks = Scorio.blocks_from_rank_list(ranking)
            all_items = sort(vcat(blocks...))
            @test all_items == collect(0:5)
        end
    end

    @testset "Utils public API exports" begin
        expected = Set([
            :competition_ranks_from_scores,
            :rank_scores,
            :compare_rankings,
            :lehmer_hash,
            :lehmer_unhash,
            :ranking_hash,
            :unhash_ranking,
        ])
        actual = Set(filter(name -> name != :Utils, names(Scorio.Utils; all=false, imported=true)))
        @test actual == expected
    end
end
