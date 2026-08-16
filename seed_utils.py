"""
Shared seed utilities for all Minegym drivers.

--seed N        : same seed every episode (deterministic replay).
--seed_start N  : LCG anchor; episode i uses gen_seed(i, N) -- independent,
                  reproducible seeds, for statistical evaluation.

Both drivers (mGym_GymRun.py train/play and mGym_DefSchdRun.py) route
through resolve_episode_seeds() so the semantics match everywhere.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

# LCG constants (Numerical Recipes); must not vary between drivers.
_LCG_A = 1664525
_LCG_C = 1013904223
_LCG_M = 2 ** 32


def gen_seed(iteration: int,
             initial_seed: Optional[int] = None,
             ax: int = _LCG_A,
             cx: int = _LCG_C,
             mx: int = _LCG_M) -> int:
    """
    Return the LCG-derived seed for `iteration` (0-based; iteration=0
    returns initial_seed unchanged). If initial_seed is None, a random
    anchor is drawn -- callers needing reproducibility must pass an int.
    """
    if initial_seed is None:
        initial_seed = random.randint(0, mx - 1)

    epi_seed = initial_seed
    for _ in range(iteration):
        epi_seed = (ax * epi_seed + cx) % mx
    return epi_seed


def resolve_episode_seeds(
    *,
    seed: Optional[int],
    seed_start: Optional[int],
    num_episodes: int,
) -> Tuple[List[int], Optional[int], str]:
    """
    Resolve (seed, seed_start, num_episodes) into per-episode seeds.

    Returns (episode_seeds, anchor, mode): `anchor` is used for filename
    tagging (equals `seed` in fixed mode, `seed_start` in LCG mode) and
    `mode` is "fixed" or "lcg". Exactly one of seed/seed_start should be
    given -- the drivers' argparse enforces this; this is a direct guard
    against programmatic misuse.
    """
    if seed is not None and seed_start is not None:
        raise ValueError(
            "resolve_episode_seeds: seed and seed_start are mutually "
            "exclusive; pass exactly one."
        )
    if num_episodes < 1:
        raise ValueError(
            f"resolve_episode_seeds: num_episodes must be >= 1, got {num_episodes}"
        )

    if seed is not None:
        return [seed] * num_episodes, seed, "fixed"

    if seed_start is not None:
        return (
            [gen_seed(i, initial_seed=seed_start) for i in range(num_episodes)],
            seed_start,
            "lcg",
        )

    # Neither given: fall back to a random LCG anchor.
    random_anchor = random.randint(0, _LCG_M - 1)
    return (
        [gen_seed(i, initial_seed=random_anchor) for i in range(num_episodes)],
        random_anchor,
        "lcg",
    )
