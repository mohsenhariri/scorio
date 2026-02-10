"""Shared type aliases"""

from typing import Literal, TypeAlias

RankMethod: TypeAlias = Literal["competition", "competition_max", "dense", "avg"]
PairwiseTieHandling: TypeAlias = Literal["skip", "draw", "correct_draw_only"]
