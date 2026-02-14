using Test
using Scorio

@testset "eval/apis.jl" begin
    R = Int[
        0 1 1 0 1
        1 1 0 1 1
    ]
    R_multi = Int[
        0 1 2 2 1
        1 1 0 2 2
    ]
    w = [0.0, 0.5, 1.0]
    R0 = Int[
        0 2
        1 2
    ]

    @testset "Bayes family" begin
        mu_prior, sigma_prior = Scorio.Eval.bayes(R_multi, w, R0)
        @test mu_prior ≈ 0.575 atol = 1e-12
        @test sigma_prior ≈ 0.08427498280790524 atol = 1e-12

        mu_noprior, sigma_noprior = Scorio.Eval.bayes(R_multi, w)
        @test mu_noprior ≈ 0.5625 atol = 1e-12
        @test sigma_noprior ≈ 0.0919975090242484 atol = 1e-12

        mu_auto, sigma_auto = Scorio.Eval.bayes(R)
        mu_explicit, sigma_explicit = Scorio.Eval.bayes(R, [0.0, 1.0])
        @test mu_auto ≈ mu_explicit atol = 1e-12
        @test sigma_auto ≈ sigma_explicit atol = 1e-12

        @test_throws ErrorException Scorio.Eval.bayes(R_multi)
        @test_throws ErrorException Scorio.Eval.bayes(R_multi, w, zeros(Int, 3, 2))

        mu, sigma, lo, hi = Scorio.Eval.bayes_ci(R, nothing, nothing, 0.9, (0.0, 1.0))
        z = Scorio._z_value(0.9; two_sided=true)
        expected_lo = max(mu - z * sigma, 0.0)
        expected_hi = min(mu + z * sigma, 1.0)
        @test lo ≈ expected_lo atol = 1e-12
        @test hi ≈ expected_hi atol = 1e-12
        @test lo <= mu <= hi
    end

    @testset "Avg family" begin
        a, sigma_a = Scorio.Eval.avg(R)
        @test a ≈ 0.7 atol = 1e-12
        @test sigma_a ≈ 0.16583123951776998 atol = 1e-12

        a_w, sigma_w = Scorio.Eval.avg(R_multi, w)
        @test a_w ≈ 0.6 atol = 1e-12
        @test sigma_w ≈ 0.14719601443879746 atol = 1e-12

        @test_throws ErrorException Scorio.Eval.avg(R_multi)

        a_ci, sigma_ci, lo, hi = Scorio.Eval.avg_ci(R, nothing, 0.8, (0.0, 1.0))
        z = Scorio._z_value(0.8; two_sided=true)
        expected_lo = max(a_ci - z * sigma_ci, 0.0)
        expected_hi = min(a_ci + z * sigma_ci, 1.0)
        @test lo ≈ expected_lo atol = 1e-12
        @test hi ≈ expected_hi atol = 1e-12
        @test lo <= a_ci <= hi
    end

    @testset "Pass family point metrics" begin
        @test Scorio.Eval.pass_at_k(R, 1) ≈ 0.7 atol = 1e-12
        @test Scorio.Eval.pass_at_k(R, 2) ≈ 0.95 atol = 1e-12
        @test Scorio.Eval.pass_hat_k(R, 1) ≈ 0.7 atol = 1e-12
        @test Scorio.Eval.pass_hat_k(R, 2) ≈ 0.45 atol = 1e-12
        @test Scorio.Eval.g_pass_at_k(R, 2) ≈ 0.45 atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(R, 2, 0.5) ≈ 0.95 atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(R, 2, 1.0) ≈ 0.45 atol = 1e-12
        @test Scorio.Eval.mg_pass_at_k(R, 2) ≈ 0.45 atol = 1e-12
        @test Scorio.Eval.mg_pass_at_k(R, 3) ≈ (1 / 6) atol = 1e-12
        @test Scorio.Eval.mg_pass_at_k(R, 1) ≈ 0.0 atol = 1e-12
    end

    @testset "Pass family CI and aliases" begin
        mu1, sigma1, lo1, hi1 = Scorio.Eval.pass_at_k_ci(R, 1)
        @test mu1 ≈ 0.6428571428571428 atol = 1e-12
        @test sigma1 ≈ 0.11845088536983578 atol = 1e-12
        @test lo1 ≈ 0.4106976734083569 atol = 1e-12
        @test hi1 ≈ 0.8750166123059286 atol = 1e-12

        mu2, sigma2, lo2, hi2 = Scorio.Eval.pass_hat_k_ci(R, 2)
        @test mu2 ≈ 0.4464285714285715 atol = 1e-12
        @test sigma2 ≈ 0.14616701378343605 atol = 1e-12
        @test lo2 ≈ 0.15994648845448006 atol = 1e-12
        @test hi2 ≈ 0.7329106544026629 atol = 1e-12

        @test Scorio.Eval.g_pass_at_k(R, 3) ≈ Scorio.Eval.pass_hat_k(R, 3) atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(R, 3, 0.0) ≈ Scorio.Eval.pass_at_k(R, 3) atol = 1e-12
        @test Scorio.Eval.g_pass_at_k_tau(R, 3, 1.0) ≈ Scorio.Eval.pass_hat_k(R, 3) atol = 1e-12

        c1 = Scorio.Eval.g_pass_at_k_ci(R, 3)
        c2 = Scorio.Eval.pass_hat_k_ci(R, 3)
        @test all(isapprox.(collect(c1), collect(c2); atol=1e-12))

        c3 = Scorio.Eval.g_pass_at_k_tau_ci(R, 3, 0.0)
        c4 = Scorio.Eval.pass_at_k_ci(R, 3)
        @test all(isapprox.(collect(c3), collect(c4); atol=1e-12))

        c5 = Scorio.Eval.g_pass_at_k_tau_ci(R, 3, 1.0)
        c6 = Scorio.Eval.pass_hat_k_ci(R, 3)
        @test all(isapprox.(collect(c5), collect(c6); atol=1e-12))

        @test Scorio.Eval.mg_pass_at_k_ci(R, 1) == (0.0, 0.0, 0.0, 0.0)
    end

    @testset "Validation and vector input" begin
        for fn in (
            Scorio.Eval.pass_at_k,
            Scorio.Eval.pass_hat_k,
            Scorio.Eval.g_pass_at_k,
            Scorio.Eval.mg_pass_at_k,
            Scorio.Eval.pass_at_k_ci,
            Scorio.Eval.pass_hat_k_ci,
            Scorio.Eval.g_pass_at_k_ci,
            Scorio.Eval.mg_pass_at_k_ci,
        )
            @test_throws ErrorException fn(R, 0)
        end

        @test_throws ErrorException Scorio.Eval.g_pass_at_k_tau(R, 2, 1.1)
        @test_throws ErrorException Scorio.Eval.g_pass_at_k_tau_ci(R, 2, 1.1)
        @test_throws ErrorException Scorio.Eval.pass_at_k(Int[0 1 2; 1 0 1], 1)

        v = Int[0, 1, 1, 0, 1]
        @test Scorio.Eval.pass_at_k(v, 1) ≈ 0.6 atol = 1e-12
        @test Scorio.Eval.pass_hat_k(v, 1) ≈ 0.6 atol = 1e-12
        @test Scorio.Eval.bayes(v)[1] ≈ Scorio.Eval.bayes(reshape(v, 1, :))[1] atol = 1e-12
    end
end
