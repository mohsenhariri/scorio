using Test
using Scorio

@testset "Scorio.jl Tests" begin
    
    @testset "bayes() function" begin
        # Test data from the Python docstring
        R = [0 1 2 2 1;
             1 1 0 2 2]
        w = [0.0, 0.5, 1.0]
        R0 = [0 2;
              1 2]
        
        # Test with prior (D=2 → T=10)
        mu, sigma = bayes(R, w, R0)
        @test mu ≈ 0.575 atol=0.001
        @test sigma ≈ 0.084275 atol=0.001
        
        # Test without prior (D=0 → T=8)
        mu2, sigma2 = bayes(R, w)
        @test mu2 ≈ 0.5625 atol=0.001
        @test sigma2 ≈ 0.091998 atol=0.001
        
        # Test with nothing as R0
        mu3, sigma3 = bayes(R, w, nothing)
        @test mu3 == mu2
        @test sigma3 == sigma2
        
        # Test error handling - invalid R values
        R_invalid = [0 1 2 3 1;
                     1 1 0 2 2]
        @test_throws ErrorException bayes(R_invalid, w, R0)
        
        # Test error handling - mismatched R0 rows
        R0_invalid = [0 2]  # 1×2 instead of 2×2
        @test_throws ErrorException bayes(R, w, R0_invalid)
    end
    
    @testset "avg() function" begin
        # Test simple average
        R = [0 1 1 0;
             1 1 1 1]
        result = avg(R)
        @test result == 0.75
        
        # Test with all zeros
        R2 = [0 0 0;
              0 0 0]
        @test avg(R2) == 0.0
        
        # Test with all ones
        R3 = [1 1 1;
              1 1 1]
        @test avg(R3) == 1.0
        
        # Test with 1D array
        R4 = [0.5, 0.7, 0.9, 0.3]
        @test avg(R4) ≈ 0.6
    end
    
    @testset "competition_ranks_from_scores() function" begin
        # Test basic ranking
        scores = [0.95, 0.87, 0.87, 0.72, 0.65]
        ranks = competition_ranks_from_scores(scores)
        @test ranks == [1, 2, 2, 4, 5]
        
        # Test all tied
        scores2 = [0.5, 0.5, 0.5, 0.5]
        ranks2 = competition_ranks_from_scores(scores2)
        @test ranks2 == [1, 1, 1, 1]
        
        # Test no ties
        scores3 = [1.0, 0.8, 0.6, 0.4, 0.2]
        ranks3 = competition_ranks_from_scores(scores3)
        @test ranks3 == [1, 2, 3, 4, 5]
        
        # Test with custom tolerance
        scores4 = [1.0, 0.9999999, 0.5]
        ranks4 = competition_ranks_from_scores(scores4, tol=1e-6)
        @test ranks4 == [1, 1, 3]
    end
    
    @testset "pass_at_k() function" begin
        R = [0 1 1 0 1;
             1 1 0 1 1]
        
        # k=1
        @test pass_at_k(R, 1) ≈ 0.7
        
        # k=2
        @test pass_at_k(R, 2) ≈ 0.95
        
        # Error handling
        @test_throws ErrorException pass_at_k(R, 0)
        @test_throws ErrorException pass_at_k(R, 6)
    end

    @testset "pass_hat_k() function" begin
        R = [0 1 1 0 1;
             1 1 0 1 1]
        
        # k=1
        @test pass_hat_k(R, 1) ≈ 0.7
        
        # k=2
        @test pass_hat_k(R, 2) ≈ 0.45
        
        # Alias check
        @test g_pass_at_k(R, 2) == pass_hat_k(R, 2)
        
        # Error handling
        @test_throws ErrorException pass_hat_k(R, 0)
        @test_throws ErrorException pass_hat_k(R, 6)
    end

    @testset "g_pass_at_k_tao() function" begin
        R = [0 1 1 0 1;
             1 1 0 1 1]
        
        # Test from Python docstring
        @test g_pass_at_k_tao(R, 2, 0.5) ≈ 0.95 atol=0.001
        @test g_pass_at_k_tao(R, 2, 1.0) ≈ 0.45 atol=0.001
        
        # tao=0 should be equivalent to pass_at_k
        @test g_pass_at_k_tao(R, 2, 0.0) ≈ pass_at_k(R, 2) atol=0.001
        
        # Error handling
        @test_throws ErrorException g_pass_at_k_tao(R, 2, -0.1)
        @test_throws ErrorException g_pass_at_k_tao(R, 2, 1.1)
        @test_throws ErrorException g_pass_at_k_tao(R, 0, 0.5)
        @test_throws ErrorException g_pass_at_k_tao(R, 6, 0.5)
    end

    @testset "mg_pass_at_k() function" begin
        R = [0 1 1 0 1;
             1 1 0 1 1]
        
        # Test from Python docstring
        @test mg_pass_at_k(R, 2) ≈ 0.45 atol=0.001
        @test mg_pass_at_k(R, 3) ≈ 0.166667 atol=0.001
        
        # Error handling
        @test_throws ErrorException mg_pass_at_k(R, 0)
        @test_throws ErrorException mg_pass_at_k(R, 6)
    end
    
    @testset "elo() placeholder" begin
        # Should raise an error
        @test_throws ErrorException elo()
    end
    
    @testset "Module exports" begin
        # Test that main functions are exported
        @test isdefined(Scorio, :bayes)
        @test isdefined(Scorio, :avg)
        @test isdefined(Scorio, :pass_at_k)
        @test isdefined(Scorio, :pass_hat_k)
        @test isdefined(Scorio, :g_pass_at_k)
        @test isdefined(Scorio, :g_pass_at_k_tao)
        @test isdefined(Scorio, :mg_pass_at_k)
        @test isdefined(Scorio, :elo)
        @test isdefined(Scorio, :competition_ranks_from_scores)
    end
    
    @testset "Type stability" begin
        R = [0 1 2 2 1;
             1 1 0 2 2]
        w = [0.0, 0.5, 1.0]
        
        # Test return types
        mu, sigma = bayes(R, w)
        @test mu isa Float64
        @test sigma isa Float64
        
        avg_result = avg(R)
        @test avg_result isa Float64
        
        scores = [0.95, 0.87, 0.87]
        ranks = competition_ranks_from_scores(scores)
        @test ranks isa Vector{Int}
    end
    
end
