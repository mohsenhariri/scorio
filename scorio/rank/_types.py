"""Shared type aliases for ranking and tie-handling policies."""

from typing import Literal, TypeAlias

import numpy as np

RankMethod: TypeAlias = Literal["competition", "competition_max", "dense", "avg"]
PairwiseTieHandling: TypeAlias = Literal["skip", "draw", "correct_draw_only"]
RankResult: TypeAlias = np.ndarray | tuple[np.ndarray, np.ndarray]
