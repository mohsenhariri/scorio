using Test, JSON, LinearAlgebra
const TP = Scorio.Eval
const TPW = TP.TailPassWeights

_tp_test_matrix(rows) = permutedims(hcat(collect.(rows)...))
function _tp_test_close(actual, expected; atol = 2e-10)
    if actual isa AbstractMatrix
        @test isapprox(actual, _tp_test_matrix(expected); atol = atol, rtol = 0)
    elseif actual isa Tuple || actual isa AbstractVector
        @test length(actual) == length(expected)
        for (a, b) in zip(actual, expected)
            _tp_test_close(a, b; atol = atol)
        end
    else
        @test isapprox(actual, expected; atol = atol, rtol = 0)
    end
end

@testset "TailPass Python reference parity" begin
    fixture = JSON.parsefile(joinpath(@__DIR__, "../fixtures/tailpass.json"))
    for (index, c) in enumerate(fixture["cases"])
        @testset "profile and utilities $index" begin
            opts = c["options"]
            p = TP.tailpass(
                c["R"],
                c["k"],
                c["w"],
                c["R0"];
                eta = opts["eta"],
                prior = opts["prior"],
                thresholds = opts["thresholds"],
            )
            _tp_test_close(p.mean, c["mean"])
            _tp_test_close(p.std, c["std"])
            _tp_test_close(p.covariance, c["covariance"])
            _tp_test_close(p.question_mean, c["question_mean"])
            weights = TPW.uniform_weights(length(p.thresholds))
            _tp_test_close(TP.linear(p, weights), c["linear"])
            _tp_test_close(TP.discovery(p), c["discovery"])
            _tp_test_close(TP.stability(p), c["stability"])
            for m in c["moments"]
                _tp_test_close(TP.moment(p, m["lam"]), m["result"])
            end
            _tp_test_close(
                TP.tailpass_empirical(
                    c["R"],
                    min(c["k"], length(c["R"][1])),
                    c["w"];
                    thresholds = p.thresholds,
                ),
                c["empirical"],
            )
            ref = c["draws"]
            raw = ref["probabilities"]
            probs = [
                raw[h][q][l] for
                h in eachindex(raw), q in eachindex(raw[1]), l in eachindex(raw[1][1])
            ]
            d = TP.TailPassDraws(p, probs)
            _tp_test_close(d.profile, ref["profile"])
            _tp_test_close(TP.linear(d, weights), ref["linear"])
            _tp_test_close(TP.moment(d, 2), ref["moment"])
            _tp_test_close(TP.qrs(d), ref["qrs"])
            _tp_test_close(TP.qrs(d; aggregation = "profile"), ref["qrs_profile"])
            _tp_test_close(TP.rollout(d, 3), ref["rollout"])
            _tp_test_close(TP.harmonic(d), ref["harmonic"])
            _tp_test_close(
                TP.shortfall(d, collect(range(1, 0.5; length = length(p.thresholds)))),
                ref["shortfall"],
            )
            _tp_test_close(TP.summary(d), ref["summary"])
            _tp_test_close(TP.moment(TP.at_k(d, p.k+1), 2), ref["at_k"])
            _tp_test_close(TP.ci(p; method = "normal")[1], c["mean"])
            _tp_test_close(TP.linear_ci(p, weights; method = "normal")[1:2], c["linear"])
        end
    end
    for c in fixture["weights"]
        _tp_test_close(getproperty(TPW, Symbol(c["name"]))(c["args"]...), c["expected"])
    end
    p = TP.tailpass([0 1 1; 1 1 1], 4)
    draws = TP.sample(p, 12000; rng = 42)
    _tp_test_close(TP.summary(draws)[1], p.mean; atol = 0.007)
    _tp_test_close(TP.summary(draws)[2], p.std; atol = 0.007)
    @test TP.sample(p, 8; rng = 42).profile == TP.sample(p, 8; rng = 42).profile
    @test TP.moment(TP.at_k(draws, 6), 1) ≈ TP.moment(draws, 1)
    @test TP.power_mean(draws, 1) ≈ TP.linear(draws, TPW.uniform_weights(4))
    large = TP.tailpass([0, 1, 2], 100000; w = [0, 0.2, 1], thresholds = [0.5])
    for lam in (1, 2, 4)
        @test all(isfinite, TP.moment(large, lam))
    end
    @test_throws ErrorException large.mean
    @test_throws ErrorException TP.tailpass([0, 1], 3000; thresholds = [1]).covariance
    for prior in ([1e20, 1], [1, 1e20])
        q = TP.tailpass([0, 1], 4; prior = prior)
        @test q.std[prior[1] > prior[2] ? 1 : 4] > 0
        @test TP.discovery(q)[2] > 0
        @test TP.stability(q)[2] > 0
    end
    for k in (0, 1.5, true, Inf)
        @test_throws ErrorException TP.tailpass([0, 1], k)
    end
    for R in ([], [0.5, 1], [NaN, 1], [0, 2])
        @test_throws ErrorException TP.tailpass(R, 2)
    end
    @test_throws ErrorException TP.tailpass([0, 1], 2; prior = 0)
    @test_throws ErrorException TP.tailpass([0, 1], 2; thresholds = [0.5, 0.5])
    @test_throws ErrorException TP.linear(p, [0.5, 0.5])
    @test_throws ErrorException TP.moment(p, 0)
    @test_throws ErrorException TP.sample(p, 1)
    @test_throws ErrorException TP.rollout(draws, 1)
    @test_throws ErrorException TP.tailpass_empirical([0, 1], 3)
end

@testset "synchronized numerical regressions" begin
    R = [0 1 2; 1 2 2]
    weights = [0, 0.5, 1]
    for score in (TP.bayes, TP.avg, (r, w) -> TP.max_at_k_ci(r, 2, w)[1:2])
        base = score(R, weights)
        scaled = score(R, weights .* 1e155)
        @test all(isfinite, scaled)
        @test collect(scaled) ./ 1e155 ≈ collect(base) rtol=1e-12
    end
    base = TP.max_at_k_ci(R, 2, weights)
    @test TP.max_at_k_ci(R, 2, weights .+ 1e12)[2] ≈ base[2] rtol=1e-12
    @test Scorio.normal_credible_interval(2, 0.1; bounds = (0, 1)) == (1.0, 1.0)
    @test Scorio.normal_credible_interval(-2, 0.1; bounds = (0, 1)) == (0.0, 0.0)
end
