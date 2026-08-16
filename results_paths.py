"""
Centralized output-path convention for RL and classical scheduling runs.

All artifacts of one experiment share a common tag:
    {algo}_scen{scenario}_seed{seed}

Directory layout under ./results/ (override with MGYM_RESULTS_ROOT):
    RL_baselines/        -- gymrun play outputs
    non-RL_baselines/    -- mGym_DefSchdRun.py outputs

Every caller should route file naming through this module so a rename
here propagates everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

RESULTS_ROOT = Path(os.environ.get("MGYM_RESULTS_ROOT", "results"))
RL_SUBDIR = "RL_baselines"
CLASSICAL_SUBDIR = "non-RL_baselines"

# Classical scheduler `--choice` ints -> short filename tag.
# Ground truth is mGym_DesEnv.scheduler_assign(): only choices 1..4 are
# wired, and choice 4 is MSWT (not SQF, despite scheduler.py's docstring).
# Keep in sync with scheduler_assign() if it ever changes.
CLASSICAL_ALGO_TAGS = {
    1: "rnd",     # random_sel
    2: "fixed",   # static round-robin LUT
    3: "sqf",     # shortest_queue_first
    4: "mswt",    # min_shovel_waiting_time
}


# ---- Tag derivation ----

def classical_algo_tag(choice: int) -> str:
    """Return the short filename tag for a classical scheduler choice int."""
    try:
        return CLASSICAL_ALGO_TAGS[choice]
    except KeyError:
        raise ValueError(
            f"Unknown classical scheduler choice: {choice}. "
            f"Known: {sorted(CLASSICAL_ALGO_TAGS)}"
        )


def rl_algo_tag(model_path: Optional[str] = None) -> str:
    """
    Derive an algo tag from a checkpoint path (e.g. 'PPO_scenB_2k.zip' ->
    'rl_ppo'). Falls back to 'rl' when nothing recognizable is present.
    """
    if not model_path:
        return "rl"
    stem = Path(model_path).stem.lower()
    for family in ("ppo", "sac", "dqn", "a2c", "td3"):
        if family in stem:
            return f"rl_{family}"
    return f"rl_{stem}"


def _slug(scenario: Optional[str], seed: Optional[int]) -> str:
    sc = f"scen{scenario}" if scenario else "scenNA"
    sd = f"seed{seed}" if seed is not None else "seedNA"
    return f"{sc}_{sd}"


def experiment_tag(algo: str, scenario: Optional[str], seed: Optional[int]) -> str:
    """The common tag prefix all artifacts of one experiment share."""
    return f"{algo}_{_slug(scenario, seed)}"


# ---- Directory + path builders ----

def results_dir(mode: str) -> Path:
    """Return (and create) the results subdir for a mode ("rl" | "classical")."""
    if mode == "rl":
        d = RESULTS_ROOT / RL_SUBDIR
    elif mode == "classical":
        d = RESULTS_ROOT / CLASSICAL_SUBDIR
    else:
        raise ValueError(f"mode must be 'rl' or 'classical', got {mode!r}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def kpi_log_path(mode: str, algo: str, scenario: Optional[str],
                 seed: Optional[int]) -> Path:
    """Per-tick KPI log written by KPICalculator; one file per experiment."""
    return results_dir(mode) / f"kpi_log_{experiment_tag(algo, scenario, seed)}.csv"


def episode_metrics_path(mode: str, algo: str, scenario: Optional[str],
                         seed: Optional[int]) -> Path:
    """Per-episode summary written by episode_metrics_logger."""
    return (results_dir(mode)
            / f"episode_metrics_{experiment_tag(algo, scenario, seed)}.csv")


def metrics_prefix(mode: str, algo: str, scenario: Optional[str],
                   seed: Optional[int]) -> Path:
    """
    Prefix passed to get_metrics.py --out-prefix. Produces
    {tag}_hourly.csv, {tag}_per_episode.csv, {tag}_composite.csv in
    the same results subdir as the kpi log.
    """
    return results_dir(mode) / experiment_tag(algo, scenario, seed)


def channel_csv_path(mode: str, algo: str, scenario: Optional[str],
                     seed: Optional[int]) -> Path:
    """
    StepChannel scratch file -- kept alongside the kpi log so a run's
    temp state is easy to garbage-collect.
    """
    return results_dir(mode) / f"channel_{experiment_tag(algo, scenario, seed)}.csv"
