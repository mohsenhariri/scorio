"""Type aliases and enums for rank API options."""

const RankMethod = Union{String, Symbol}
const PairwiseTieHandling = Union{String, Symbol}

const RANK_METHOD_VALUES = (
    "competition",
    "competition_max",
    "dense",
    "avg",
)

const PAIRWISE_TIE_HANDLING_VALUES = (
    "skip",
    "draw",
    "correct_draw_only",
)
