# Shared simulation data loading for all test files.
# Loads NPZ data from tests/data/ once and provides accessor functions
# matching the Python conftest.py fixtures.
using NPZ

const _DATA_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "tests", "data"))

function _load_npz(path::String)::Dict{String, Array{Int}}
    raw = npzread(path)
    return Dict{String, Array{Int}}(k => Int.(v) for (k, v) in raw)
end

# Module-level cached data (loaded once)
const _TOP_P_DATA = _load_npz(joinpath(_DATA_DIR, "R_top_p.npz"))
const _GREEDY_DATA = _load_npz(joinpath(_DATA_DIR, "R_greedy.npz"))

function top_p_data()::Dict{String, Array{Int}}
    return _TOP_P_DATA
end

function greedy_data()::Dict{String, Array{Int}}
    return _GREEDY_DATA
end

function top_p_task_aime25()::Array{Int, 3}
    return _TOP_P_DATA["aime25"]
end

function greedy_task_aime25()::Array{Int, 3}
    return _GREEDY_DATA["aime25"]
end
