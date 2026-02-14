using Test
using Scorio

@testset "sinf/test_sinf.jl" begin

    @testset "RankingConfidence" begin
        @testset "identical scores zero sigma returns tie" begin
            rho, z = Scorio._ranking_confidence(0.5, 0.0, 0.5, 0.0)
            @test rho == 0.5
            @test z == Inf
        end

        @testset "different scores zero sigma returns certain" begin
            rho, z = Scorio._ranking_confidence(0.7, 0.0, 0.3, 0.0)
            @test rho == 1.0
            @test z == Inf
        end

        @testset "well separated scores high confidence" begin
            rho, z = Scorio._ranking_confidence(0.9, 0.01, 0.1, 0.01)
            @test rho > 0.999
            @test z > 3.0
        end

        @testset "overlapping scores low confidence" begin
            rho, z = Scorio._ranking_confidence(0.51, 0.1, 0.49, 0.1)
            @test rho < 0.6
            @test z < 0.5
        end

        @testset "symmetry" begin
            rho_ab, z_ab = Scorio._ranking_confidence(0.6, 0.05, 0.4, 0.03)
            rho_ba, z_ba = Scorio._ranking_confidence(0.4, 0.03, 0.6, 0.05)
            @test rho_ab ≈ rho_ba atol = 1e-12
            @test z_ab ≈ z_ba atol = 1e-12
        end

        @testset "manual z computation" begin
            mu_a, sigma_a = 0.7, 0.05
            mu_b, sigma_b = 0.4, 0.03
            expected_z = abs(mu_a - mu_b) / sqrt(sigma_a^2 + sigma_b^2)
            rho, z = Scorio._ranking_confidence(mu_a, sigma_a, mu_b, sigma_b)
            @test z ≈ expected_z atol = 1e-12
            expected_rho = Scorio._normal_cdf(expected_z)
            @test rho ≈ expected_rho atol = 1e-12
        end

        @testset "with real Bayes posteriors" begin
            data = top_p_task_aime25()
            mu_0, sigma_0 = Scorio.Eval.bayes(data[1, :, :])
            mu_1, sigma_1 = Scorio.Eval.bayes(data[2, :, :])
            rho, z = Scorio._ranking_confidence(mu_0, sigma_0, mu_1, sigma_1)
            @test 0.5 <= rho <= 1.0
            @test z >= 0.0
            @test isfinite(rho)
            @test isfinite(z)
        end
    end

    @testset "CiFromMuSigma" begin
        @testset "basic interval" begin
            z95 = Scorio._z_value(0.95; two_sided=true)
            lo, hi = Scorio.SInf.ci_from_mu_sigma(0.5, 0.1, 0.95)
            @test lo ≈ 0.5 - z95 * 0.1 atol = 1e-12
            @test hi ≈ 0.5 + z95 * 0.1 atol = 1e-12
        end

        @testset "clipping" begin
            lo, hi = Scorio.SInf.ci_from_mu_sigma(0.05, 0.1, 0.95, (0.0, 1.0))
            @test lo >= 0.0
            @test hi <= 1.0
        end

        @testset "zero sigma gives point interval" begin
            lo, hi = Scorio.SInf.ci_from_mu_sigma(0.5, 0.0, 0.99)
            @test lo ≈ 0.5 atol = 1e-12
            @test hi ≈ 0.5 atol = 1e-12
        end

        @testset "higher confidence wider interval" begin
            lo_90, hi_90 = Scorio.SInf.ci_from_mu_sigma(0.5, 0.1, 0.90)
            lo_99, hi_99 = Scorio.SInf.ci_from_mu_sigma(0.5, 0.1, 0.99)
            @test (hi_99 - lo_99) > (hi_90 - lo_90)
        end

        @testset "invalid confidence raises" begin
            @test_throws ErrorException Scorio.SInf.ci_from_mu_sigma(0.5, 0.1, 0.0)
            @test_throws ErrorException Scorio.SInf.ci_from_mu_sigma(0.5, 0.1, 1.0)
        end

        @testset "negative sigma raises" begin
            @test_throws ErrorException Scorio.SInf.ci_from_mu_sigma(0.5, -0.1)
        end

        @testset "interval contains mu" begin
            for conf in [0.5, 0.8, 0.9, 0.95, 0.99]
                lo, hi = Scorio.SInf.ci_from_mu_sigma(0.42, 0.07, conf)
                @test lo <= 0.42 <= hi
            end
        end

        @testset "with real Bayes posterior" begin
            data = top_p_task_aime25()
            mu, sigma = Scorio.Eval.bayes(data[1, :, :])
            lo, hi = Scorio.SInf.ci_from_mu_sigma(mu, sigma, 0.95, (0.0, 1.0))
            @test 0.0 <= lo <= mu <= hi <= 1.0
        end
    end

    @testset "ShouldStop" begin
        @testset "half width criterion" begin
            @test Scorio.SInf.should_stop(0.005; confidence=0.95, max_half_width=0.02) == true
            @test Scorio.SInf.should_stop(0.05; confidence=0.95, max_half_width=0.02) == false
        end

        @testset "ci width criterion" begin
            z95 = Scorio._z_value(0.95; two_sided=true)
            sigma = 0.005
            full_width = 2 * z95 * sigma
            @test Scorio.SInf.should_stop(sigma; confidence=0.95, max_ci_width=full_width + 0.001) == true
            @test Scorio.SInf.should_stop(sigma; confidence=0.95, max_ci_width=full_width - 0.001) == false
        end

        @testset "half width and ci width consistency" begin
            sigma = 0.03
            z95 = Scorio._z_value(0.95; two_sided=true)
            hw = z95 * sigma
            result_hw = Scorio.SInf.should_stop(sigma; confidence=0.95, max_half_width=hw + 1e-10)
            result_cw = Scorio.SInf.should_stop(sigma; confidence=0.95, max_ci_width=2 * hw + 1e-10)
            @test result_hw == result_cw
        end

        @testset "must provide exactly one criterion" begin
            @test_throws ErrorException Scorio.SInf.should_stop(0.01)
            @test_throws ErrorException Scorio.SInf.should_stop(0.01; max_ci_width=0.1, max_half_width=0.05)
        end

        @testset "zero sigma always stops" begin
            @test Scorio.SInf.should_stop(0.0; confidence=0.95, max_half_width=0.001) == true
            @test Scorio.SInf.should_stop(0.0; confidence=0.95, max_ci_width=0.001) == true
        end

        @testset "with real data progressive precision" begin
            data = top_p_task_aime25()
            model_data = data[1, :, :]
            sigmas = Float64[]
            for n_trials in [1, 5, 10, 20, 40, 80]
                R_sub = model_data[:, 1:n_trials]
                _, sigma = Scorio.Eval.bayes(R_sub)
                push!(sigmas, sigma)
            end

            for i in 2:length(sigmas)
                @test sigmas[i] <= sigmas[i - 1] + 1e-10
            end

            @test Scorio.SInf.should_stop(sigmas[end]; max_half_width=0.1) == true
        end
    end

    @testset "ShouldStopTop1" begin
        @testset "clear leader stops ci_overlap" begin
            mus = [0.9, 0.3, 0.2]
            sigmas = [0.01, 0.01, 0.01]
            result = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95, method="ci_overlap")
            @test result["stop"] == true
            @test result["leader"] == 1  # Julia 1-based
            @test result["ambiguous"] == []
        end

        @testset "clear leader stops zscore" begin
            mus = [0.9, 0.3, 0.2]
            sigmas = [0.01, 0.01, 0.01]
            result = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95, method="zscore")
            @test result["stop"] == true
            @test result["leader"] == 1
            @test result["ambiguous"] == []
        end

        @testset "ambiguous leader does not stop" begin
            mus = [0.51, 0.49, 0.2]
            sigmas = [0.1, 0.1, 0.01]
            result = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95, method="ci_overlap")
            @test result["stop"] == false
            @test result["leader"] == 1
            @test 2 in result["ambiguous"]  # Julia 1-based
        end

        @testset "zscore ambiguous" begin
            mus = [0.51, 0.49, 0.2]
            sigmas = [0.1, 0.1, 0.01]
            result = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95, method="zscore")
            @test result["stop"] == false
            @test 2 in result["ambiguous"]
        end

        @testset "all equal means no stop" begin
            mus = [0.5, 0.5, 0.5]
            sigmas = [0.1, 0.1, 0.1]
            result = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95)
            @test result["stop"] == false
        end

        @testset "leader is argmax" begin
            mus = [0.3, 0.7, 0.5, 0.1]
            sigmas = [0.01, 0.01, 0.01, 0.01]
            result = Scorio.SInf.should_stop_top1(mus, sigmas)
            @test result["leader"] == 2  # Julia 1-based argmax
        end

        @testset "validation shape mismatch" begin
            @test_throws ErrorException Scorio.SInf.should_stop_top1([0.5, 0.3], [0.1])
        end

        @testset "validation empty input" begin
            @test_throws ErrorException Scorio.SInf.should_stop_top1(Float64[], Float64[])
        end

        @testset "invalid method" begin
            @test_throws ErrorException Scorio.SInf.should_stop_top1([0.5, 0.3], [0.1, 0.1]; method="bad")
        end

        @testset "with real Bayes posteriors" begin
            R = top_p_data()["aime25"]
            L = size(R, 1)
            mus = zeros(Float64, L)
            sigmas = zeros(Float64, L)
            for i in 1:L
                mus[i], sigmas[i] = Scorio.Eval.bayes(R[i, :, :])
            end

            result_ci = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95, method="ci_overlap")
            result_zs = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95, method="zscore")

            @test isa(result_ci["stop"], Bool)
            @test isa(result_zs["stop"], Bool)
            @test 1 <= result_ci["leader"] <= L
            @test 1 <= result_zs["leader"] <= L
            @test all(1 <= j <= L for j in result_ci["ambiguous"])
            @test !(result_ci["leader"] in result_ci["ambiguous"])
        end

        @testset "reducing uncertainty leads to stop" begin
            mus = [0.7, 0.5, 0.3]
            for sigma in [1.0, 0.1, 0.01, 0.001]
                sigmas = [sigma, sigma, sigma]
                result = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95, method="ci_overlap")
                if sigma <= 0.01
                    @test result["stop"] == true
                end
            end
        end
    end

    @testset "SuggestNextAllocation" begin
        @testset "returns leader and competitor" begin
            mus = [0.7, 0.5, 0.3]
            sigmas = [0.05, 0.05, 0.05]
            leader, competitor = Scorio.SInf.suggest_next_allocation(mus, sigmas)
            @test leader == 1
            @test competitor != leader
            @test 1 <= competitor <= 3
        end

        @testset "most ambiguous competitor selected ci_overlap" begin
            mus = [0.7, 0.65, 0.2]
            sigmas = [0.05, 0.05, 0.05]
            leader, competitor = Scorio.SInf.suggest_next_allocation(mus, sigmas; method="ci_overlap")
            @test leader == 1
            @test competitor == 2
        end

        @testset "most ambiguous competitor selected zscore" begin
            mus = [0.7, 0.65, 0.2]
            sigmas = [0.05, 0.05, 0.05]
            leader, competitor = Scorio.SInf.suggest_next_allocation(mus, sigmas; method="zscore")
            @test leader == 1
            @test competitor == 2
        end

        @testset "high sigma competitor is more ambiguous" begin
            mus = [0.7, 0.4, 0.4]
            sigmas = [0.01, 0.01, 0.15]
            leader, competitor = Scorio.SInf.suggest_next_allocation(mus, sigmas; method="ci_overlap")
            @test leader == 1
            @test competitor == 3
        end

        @testset "validation need at least two" begin
            @test_throws ErrorException Scorio.SInf.suggest_next_allocation([0.5], [0.1])
        end

        @testset "validation shape mismatch" begin
            @test_throws ErrorException Scorio.SInf.suggest_next_allocation([0.5, 0.3], [0.1])
        end

        @testset "invalid method" begin
            @test_throws ErrorException Scorio.SInf.suggest_next_allocation([0.5, 0.3], [0.1, 0.1]; method="bad")
        end

        @testset "with real Bayes posteriors" begin
            R = top_p_data()["aime25"]
            L = size(R, 1)
            mus = zeros(Float64, L)
            sigmas = zeros(Float64, L)
            for i in 1:L
                mus[i], sigmas[i] = Scorio.Eval.bayes(R[i, :, :])
            end

            leader, competitor = Scorio.SInf.suggest_next_allocation(mus, sigmas; confidence=0.95, method="ci_overlap")
            @test 1 <= leader <= L
            @test 1 <= competitor <= L
            @test leader != competitor
            @test leader == argmax(mus)
        end

        @testset "consistency with should_stop_top1" begin
            mus = [0.6, 0.55, 0.3, 0.1]
            sigmas = [0.1, 0.1, 0.05, 0.05]

            stop_result = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95, method="ci_overlap")
            if !stop_result["stop"]
                leader, competitor = Scorio.SInf.suggest_next_allocation(mus, sigmas; confidence=0.95, method="ci_overlap")
                @test leader == stop_result["leader"]
                @test competitor in stop_result["ambiguous"]
            end
        end
    end

    @testset "AdaptiveWorkflowIntegration" begin
        @testset "sequential evaluation workflow" begin
            R_full = top_p_data()["aime25"]
            L, M, N = size(R_full)

            for n_trials in 5:5:N
                R_sub = R_full[:, :, 1:n_trials]
                mus = zeros(Float64, L)
                sigmas = zeros(Float64, L)
                for i in 1:L
                    mus[i], sigmas[i] = Scorio.Eval.bayes(R_sub[i, :, :])
                end

                stop_result = Scorio.SInf.should_stop_top1(mus, sigmas; confidence=0.95, method="ci_overlap")

                if stop_result["stop"]
                    leader = stop_result["leader"]
                    @test mus[leader] == maximum(mus)
                    break
                end

                leader, competitor = Scorio.SInf.suggest_next_allocation(mus, sigmas; confidence=0.95, method="ci_overlap")
                @test leader == argmax(mus)
                @test competitor != leader
            end
        end

        @testset "CI width convergence" begin
            data = top_p_task_aime25()
            model_data = data[1, :, :]
            prev_width = Inf

            for n_trials in [2, 5, 10, 20, 40, 80]
                R_sub = model_data[:, 1:n_trials]
                mu, sigma = Scorio.Eval.bayes(R_sub)
                lo, hi = Scorio.SInf.ci_from_mu_sigma(mu, sigma, 0.95, (0.0, 1.0))
                width = hi - lo
                @test width <= prev_width + 1e-10
                prev_width = width
            end
        end

        @testset "pairwise confidence with more trials" begin
            data = top_p_task_aime25()
            model_a_data = data[1, :, :]
            model_b_data = data[6, :, :]

            z_values = Float64[]
            for n_trials in [5, 10, 20, 40, 80]
                mu_a, sigma_a = Scorio.Eval.bayes(model_a_data[:, 1:n_trials])
                mu_b, sigma_b = Scorio.Eval.bayes(model_b_data[:, 1:n_trials])
                rho, z = Scorio._ranking_confidence(mu_a, sigma_a, mu_b, sigma_b)
                push!(z_values, z)
                @test isfinite(z)
            end

            @test abs(z_values[end]) > 0.0
        end
    end

    @testset "SInf public API exports" begin
        expected = Set([:ci_from_mu_sigma, :should_stop, :should_stop_top1, :suggest_next_allocation])
        actual = Set(filter(name -> name != :SInf, names(Scorio.SInf; all=false, imported=true)))
        @test actual == expected
    end
end
