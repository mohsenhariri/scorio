"""Shared validation and pairwise-count helpers for rank methods."""

function validate_input(R; binary_only::Bool=true)
    if !(R isa AbstractArray)
        error(
            "Input R must be a 2D array of shape (L, M) or 3D array of shape (L, M, N), got shape ()",
        )
    end

    A = Array(R)

    # Promote (L, M) to (L, M, 1)
    if ndims(A) == 2
        A = reshape(A, size(A, 1), size(A, 2), 1)
    elseif ndims(A) != 3
        error(
            "Input R must be a 2D array of shape (L, M) or 3D array of shape (L, M, N), got shape $(size(A))",
        )
    end

    if eltype(A) <: Bool
        A_int = Int.(A)
    else
        if !(eltype(A) <: Number)
            error("Input R must be numeric, got dtype $(eltype(A))")
        end

        if eltype(A) <: Complex
            error("Input R must contain real-valued outcomes")
        end

        if any(x -> !isfinite(x), A)
            error("Input R must not contain NaN or Inf values")
        end

        if eltype(A) <: AbstractFloat
            if any(x -> !(x == 0 || x == 1), A)
                error(
                    "Float inputs must be binary values (0.0 or 1.0). Use integer dtype for multiclass outcomes.",
                )
            end
        elseif binary_only
            if any(x -> !(x == 0 || x == 1), A)
                error("Input R must contain only binary values (0 or 1)")
            end
        end

        A_int = Int.(A)
    end

    L, M, N = size(A_int)
    if L < 2
        error("Need at least 2 models to rank, got L=$L")
    end
    if M < 1
        error("Need at least 1 question, got M=$M")
    end
    if N < 1
        error("Need at least 1 trial, got N=$N")
    end

    return A_int
end

function build_pairwise_wins(R)
    if ndims(R) != 3
        error("Input R must be 3D array of shape (L, M, N), got shape $(size(R))")
    end

    L, M, N = size(R)
    wins = zeros(Float64, L, L)

    @inbounds for i in 1:L
        for j in (i + 1):L
            i_wins = 0
            j_wins = 0
            for m in 1:M, n in 1:N
                ri = R[i, m, n]
                rj = R[j, m, n]
                if ri == 1 && rj == 0
                    i_wins += 1
                elseif rj == 1 && ri == 0
                    j_wins += 1
                end
            end

            wins[i, j] = i_wins
            wins[j, i] = j_wins
        end
    end

    return wins
end

function build_pairwise_counts(R)
    if ndims(R) != 3
        error("Input R must be 3D array of shape (L, M, N), got shape $(size(R))")
    end

    L, M, N = size(R)
    wins = zeros(Float64, L, L)
    ties = zeros(Float64, L, L)

    @inbounds for i in 1:L
        for j in (i + 1):L
            i_wins = 0
            j_wins = 0
            both_same = 0
            for m in 1:M, n in 1:N
                ri = R[i, m, n]
                rj = R[j, m, n]

                if ri == 1 && rj == 0
                    i_wins += 1
                elseif rj == 1 && ri == 0
                    j_wins += 1
                end

                if ri == rj
                    both_same += 1
                end
            end

            wins[i, j] = i_wins
            wins[j, i] = j_wins
            ties[i, j] = both_same
            ties[j, i] = both_same
        end
    end

    return wins, ties
end

function sigmoid(x)
    clipped = clamp.(x, -30.0, 30.0)
    return 1.0 ./ (1.0 .+ exp.(-clipped))
end
