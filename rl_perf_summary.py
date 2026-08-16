"""
Run each trained RL algorithm's "best" checkpoint for a chosen training
seed across scenarios A-F via `mGym_GymRun.py play`, organize the resulting
artifacts into results/plotting/RL_baselines/{ALGO}/, and aggregate them
into RL_perf_summ.csv -- the RL-side counterpart to non_RL_perf_summ.csv.
Also emits a long-format per-episode CSV (RL_perf_episodes.csv) for
downstream statistical tests.

The training seed of the checkpoint to load is --checkpoint-seed (default:
1013904223), distinct from --seed-start, the LCG anchor for the play run.

Uses a snapshot-diff of results/RL_baselines/ (rather than predicting
output filenames) because results_paths.rl_algo_tag() falls through to a
messy fallback tag for TRPO checkpoints (it doesn't match any of the
hardcoded "ppo"/"sac"/"dqn"/"a2c"/"td3" substrings), so filenames aren't
reliably predictable per algorithm.

Layout produced (seed{N} suffix reflects --seed-start, not --checkpoint-seed;
example with defaults, seed_start=0):

    results/plotting/RL_baselines/A2C/kpi_log_a2c_scenA_seed0.csv
    results/plotting/RL_baselines/A2C/channel_a2c_scenA_seed0_dispatch_log.csv
    results/plotting/RL_baselines/A2C/channel_a2c_scenA_seed0_episode_metrics.csv
    ... (same pattern for DQN, MASKABLE_PPO, TRPO)

    results/plotting/RL_perf_summ.csv       # aggregated mean/std per (scenario, algo)
    results/plotting/RL_perf_episodes.csv   # long-format, one row per episode

Usage
-----
    python rl_perf_summary.py                            # full sweep, all algos/scenarios
    python rl_perf_summary.py --dry-run -v                # preview the planned runs
    python rl_perf_summary.py --algos a2c dqn             # subset of algorithms
    python rl_perf_summary.py --scenarios A F             # subset of scenarios
    python rl_perf_summary.py --force                     # re-run even if output exists
    python rl_perf_summary.py --skip-run                  # aggregate only, no simulation
    python rl_perf_summary.py --checkpoint-seed 42        # load a different training seed
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from get_metrics import compute_per_episode, load_kpi_log

logger = logging.getLogger(__name__)

# Single source of truth for algo directory <-> CLI/display names.

ALGO_ORDER: list[str] = [
    "A2C_Models",
    "DQN_Models",
    "MASKABLE_PPO_Models",
    "TRPO_Models",
]
CLI_ALGO_NAME: dict[str, str] = {
    "A2C_Models": "a2c",
    "DQN_Models": "dqn",
    "MASKABLE_PPO_Models": "maskable_ppo",
    "TRPO_Models": "trpo",
}
DISPLAY_NAME: dict[str, str] = {
    "A2C_Models": "A2C",
    "DQN_Models": "DQN",
    "MASKABLE_PPO_Models": "PPO-masked",
    "TRPO_Models": "TRPO",
}
# Destination subdir name, e.g. "A2C/", "DQN/", "MASKABLE_PPO/", "TRPO/".
SUBDIR_NAME: dict[str, str] = {a: a.removesuffix("_Models") for a in ALGO_ORDER}
# Clean tag for renaming files during organize_run_files (avoids leaking
# the messy TRPO fallback tag -- see module docstring).
CLEAN_TAG: dict[str, str] = {a: CLI_ALGO_NAME[a] for a in ALGO_ORDER}

SCENARIOS_ALL = ["A", "B", "C", "D", "E", "F"]
RUN_DIR_GLOB = "run_*"


# Checkpoint discovery

def find_seed_best_dir(
    mgym_root: Path, algo_dir: str, seed: int
) -> Optional[Path]:
    """
    Locate the seed_{seed}/best directory for an algorithm, tolerating an
    optional timestamped run_* layer in between.
    """
    algo_root = mgym_root / algo_dir
    if not algo_root.is_dir():
        return None

    seed_names = (f"seed_{seed}", f"seed{seed}")

    for seed_name in seed_names:
        direct = algo_root / seed_name / "best"
        if direct.is_dir():
            return direct

    run_dirs = sorted(p for p in algo_root.glob(RUN_DIR_GLOB) if p.is_dir())
    for run_dir in reversed(run_dirs):  # most recent first
        for seed_name in seed_names:
            candidate = run_dir / seed_name / "best"
            if candidate.is_dir():
                if len(run_dirs) > 1:
                    logger.info(
                        "Multiple run_* directories under %s; using most recent: %s",
                        algo_root, run_dir.name,
                    )
                return candidate
    return None


def find_checkpoint_zip(best_dir: Path) -> Optional[Path]:
    """Find the model checkpoint .zip in a best/ dir, skipping VecNormalize pickles."""
    candidates = sorted(
        p for p in best_dir.glob("*.zip") if "vecnormalize" not in p.name.lower()
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        logger.warning(
            "Multiple checkpoint .zip files in %s; using %s",
            best_dir, candidates[0].name,
        )
    return candidates[0]


# Execution: snapshot-diff instead of predicting output filenames

@dataclass(frozen=True)
class RunResult:
    algo_dir: str
    scenario: str
    success: bool
    message: str
    files: list[Path] = field(default_factory=list)


def _snapshot(d: Path) -> dict[str, float]:
    """filename -> mtime for every file directly in d (non-recursive)."""
    if not d.is_dir():
        return {}
    return {p.name: p.stat().st_mtime for p in d.iterdir() if p.is_file()}


def _new_or_changed(before: dict[str, float], d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    after = {p.name: p.stat().st_mtime for p in d.iterdir() if p.is_file()}
    return [
        d / name
        for name, mtime in after.items()
        if name not in before or mtime > before[name] + 1e-6
    ]


def run_one_pair(
    mgym_root: Path,
    algo_dir: str,
    scenario: str,
    model_path: Path,
    num_episodes: int,
    seed_start: int,
    rl_baselines_src: Path,
    log_dir: Path,
) -> RunResult:
    """
    Invoke `mGym_GymRun.py play` for one (algorithm, scenario) pair and
    report which output files it produced. Exit code alone isn't
    sufficient -- success also requires a kpi_log_*.csv to appear.
    """
    cli_algo = CLI_ALGO_NAME[algo_dir]
    before = _snapshot(rl_baselines_src)

    cmd = [
        sys.executable, "mGym_GymRun.py", "play",
        "--num_episodes", str(num_episodes),
        "--model_path", str(model_path),
        "--scenario", scenario,
        "--seed_start", str(seed_start),
        "--algo", cli_algo,
    ]

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{algo_dir}_scen{scenario}.log"
    logger.info("Running %s / scenario %s -> %s", algo_dir, scenario, log_path.name)

    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=mgym_root, stdout=logf, stderr=subprocess.STDOUT)

    new_files = _new_or_changed(before, rl_baselines_src)
    kpi_files = [p for p in new_files if p.name.startswith("kpi_log_")]

    if proc.returncode != 0:
        return RunResult(
            algo_dir, scenario, False,
            f"exit code {proc.returncode}; see {log_path}", new_files,
        )
    if not kpi_files:
        return RunResult(
            algo_dir, scenario, False,
            f"no kpi_log_*.csv produced (exit 0 but likely a silent error); see {log_path}",
            new_files,
        )
    return RunResult(algo_dir, scenario, True, "ok", new_files)


# Canonicalize filenames on the way into results/plotting/

def organize_run_files(
    files: list[Path], dest_dir: Path, algo_dir: str, scenario: str, seed_start: int
) -> list[Path]:
    """
    Move a run's output files into dest_dir, renaming to a clean,
    consistent scheme so no tag-derivation quirk (e.g. TRPO's fallback
    tag) surfaces in the organized directory.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    clean_tag = CLEAN_TAG[algo_dir]
    tag = f"{clean_tag}_scen{scenario}_seed{seed_start}"

    moved: list[Path] = []
    for src in files:
        name_lower = src.name.lower()
        if name_lower.startswith("kpi_log_"):
            dest_name = f"kpi_log_{tag}.csv"
        elif name_lower.startswith("channel_") and "dispatch" in name_lower:
            dest_name = f"channel_{tag}_dispatch_log.csv"
        elif name_lower.startswith("channel_") and "episode_metric" in name_lower:
            dest_name = f"channel_{tag}_episode_metrics.csv"
        elif name_lower.startswith("channel_"):
            dest_name = f"channel_{tag}.csv"
        else:
            # Unexpected file type: keep it, tagged for traceability.
            dest_name = f"{tag}_{src.name}"

        dest_path = dest_dir / dest_name
        shutil.move(str(src), str(dest_path))
        moved.append(dest_path)
        logger.debug("Moved %s -> %s", src.name, dest_path)

    return moved


# Sweep orchestration

def run_sweep(
    mgym_root: Path,
    plotting_rl_root: Path,
    algos: list[str],
    scenarios: list[str],
    num_episodes: int,
    seed_start: int,
    checkpoint_seed: int,
    force: bool,
    dry_run: bool,
) -> list[RunResult]:
    rl_baselines_src = mgym_root / "results" / "RL_baselines"
    log_dir = plotting_rl_root / "logs"

    results: list[RunResult] = []
    interrupted = False

    for algo_dir in algos:
        dest_dir = plotting_rl_root / SUBDIR_NAME[algo_dir]

        best_dir = find_seed_best_dir(mgym_root, algo_dir, checkpoint_seed)
        if best_dir is None:
            logger.error(
                "No seed_%d/best directory found for %s", checkpoint_seed, algo_dir
            )
            results.extend(
                RunResult(
                    algo_dir, s, False,
                    f"no seed_{checkpoint_seed} checkpoint directory found",
                )
                for s in scenarios
            )
            continue

        model_path = find_checkpoint_zip(best_dir)
        if model_path is None:
            logger.error("No checkpoint .zip found in %s", best_dir)
            results.extend(
                RunResult(algo_dir, s, False, f"no checkpoint .zip in {best_dir}")
                for s in scenarios
            )
            continue

        for scenario in scenarios:
            if interrupted:
                results.append(RunResult(algo_dir, scenario, False, "skipped (interrupted)"))
                continue

            existing = sorted(dest_dir.glob(f"kpi_log_*_scen{scenario}_seed{seed_start}.csv"))
            if existing and not force:
                logger.info("SKIP %s/%s (already exists: %s)", algo_dir, scenario, existing[0].name)
                results.append(RunResult(algo_dir, scenario, True, "skipped (already exists)", existing))
                continue

            if dry_run:
                logger.info(
                    "[DRY RUN] %s / scenario %s -> model=%s", algo_dir, scenario, model_path
                )
                results.append(RunResult(algo_dir, scenario, True, "dry-run"))
                continue

            try:
                run_result = run_one_pair(
                    mgym_root, algo_dir, scenario, model_path,
                    num_episodes, seed_start, rl_baselines_src, log_dir,
                )
            except KeyboardInterrupt:
                logger.warning("Interrupted — finishing current pair, skipping the rest.")
                interrupted = True
                results.append(RunResult(algo_dir, scenario, False, "interrupted mid-run"))
                continue

            if run_result.success and run_result.files:
                organized = organize_run_files(
                    run_result.files, dest_dir, algo_dir, scenario, seed_start
                )
                run_result = RunResult(algo_dir, scenario, True, run_result.message, organized)
            results.append(run_result)

    return results


def print_sweep_summary(results: list[RunResult]) -> None:
    # Success determines bucket first: a "skipped (interrupted)" result
    # has success=False and belongs in Failed despite its message.
    skipped = [r for r in results if r.success and r.message.startswith("skipped")]
    passed = [r for r in results if r.success and not r.message.startswith("skipped")]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 60)
    print(f"RL sweep summary: {len(results)} pair(s)")
    print(f"  Passed:  {len(passed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Failed:  {len(failed)}")
    if failed:
        print("\nFailed pairs:")
        for r in failed:
            print(f"  {r.algo_dir} / {r.scenario}: {r.message}")
    print("=" * 60)


# Aggregation: reuses get_metrics.py's parsing, no duplicated logic.

def collect_per_episode(
    plotting_rl_root: Path, algos: list[str], scenarios: list[str]
) -> pd.DataFrame:
    """
    Parse every organized kpi_log_*.csv into per-episode rows and return
    a long-format DataFrame with one row per (Scenario, Algorithm, Episode):
    Scenario, Algorithm, Episode, Final_Production_Volume, Mean_Shovel_Queue.
    Feeds both downstream stat tests and build_rl_summary below.
    """
    frames: list[pd.DataFrame] = []

    for algo_dir in algos:
        dest_dir = plotting_rl_root / SUBDIR_NAME[algo_dir]
        if not dest_dir.is_dir():
            logger.warning("No output directory for %s at %s", algo_dir, dest_dir)
            continue

        for scenario in scenarios:
            matches = sorted(dest_dir.glob(f"kpi_log_*_scen{scenario}_seed*.csv"))
            if not matches:
                logger.warning(
                    "No kpi_log file for %s scenario=%s in %s",
                    algo_dir, scenario, dest_dir,
                )
                continue
            if len(matches) > 1:
                logger.warning(
                    "Multiple kpi_log files for %s scenario=%s; using %s",
                    algo_dir, scenario, matches[0].name,
                )
            path = matches[0]

            try:
                raw = load_kpi_log(path)
                per_episode = compute_per_episode(raw)
            except (ValueError, FileNotFoundError) as e:
                logger.warning("Failed to parse %s: %s", path, e)
                continue

            if per_episode.empty:
                logger.warning("No episodes parsed from %s", path)
                continue

            df = pd.DataFrame({
                "Scenario": scenario,
                "Algorithm": DISPLAY_NAME[algo_dir],
                "Episode": range(1, len(per_episode) + 1),
                "Final_Production_Volume": per_episode["Final_Production_Volume"].values,
                "Mean_Shovel_Queue": per_episode["Mean_Shovel_Queue"].values,
            })
            frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=["Scenario", "Algorithm", "Episode",
                     "Final_Production_Volume", "Mean_Shovel_Queue"]
        )
    return pd.concat(frames, ignore_index=True)


def build_rl_summary(episodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the long-format per-episode DataFrame to one row per
    (Scenario, Algorithm) — same shape as non_RL_perf_summ.csv.
    """
    if episodes_df.empty:
        return pd.DataFrame()

    grouped = episodes_df.groupby(["Scenario", "Algorithm"], sort=False)
    rows: list[dict[str, object]] = []
    for (scenario, algorithm), group in grouped:
        rows.append({
            "Scenario": scenario,
            "Algorithm": algorithm,
            "Mean_PVol": group["Final_Production_Volume"].mean(),
            "Std_PVol": group["Final_Production_Volume"].std(),
            "N_Episodes": len(group),
            "Mean_Queue_Length_Mean": group["Mean_Shovel_Queue"].mean(),
            "Mean_Queue_Length_Std": group["Mean_Shovel_Queue"].std(),
        })

    df = pd.DataFrame(rows)
    df[["Std_PVol", "Mean_Queue_Length_Std"]] = df[
        ["Std_PVol", "Mean_Queue_Length_Std"]
    ].fillna(0.0)
    return df


# CLI

def main(argv: Optional[list[str]] = None) -> int:
    script_dir = Path(__file__).resolve().parent
    default_mgym_root = script_dir.parent.parent  # results/plotting/ -> mgym/

    parser = argparse.ArgumentParser(
        description="Sweep RL algorithms across scenarios and summarize performance."
    )
    parser.add_argument("--mgym-root", type=Path, default=default_mgym_root,
                        help="Project root containing A2C_Models/, DQN_Models/, etc.")
    parser.add_argument("--algos", nargs="+",
                        choices=["a2c", "dqn", "maskable_ppo", "trpo"],
                        default=["a2c", "dqn", "maskable_ppo", "trpo"],
                        help="Which algorithms to sweep.")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS_ALL,
                        default=SCENARIOS_ALL, help="Which scenarios to sweep.")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Episodes per (algo, scenario) pair.")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="LCG seed anchor passed to --seed_start.")
    parser.add_argument("--checkpoint-seed", type=int, default=1013904223,
                        help="Training seed of the checkpoint to load "
                             "(selects the seed_{N}/best directory).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run pairs even if output already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs without executing them.")
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip the simulation sweep; aggregate existing output only.")
    parser.add_argument("--out-name", default="RL_perf_summ.csv",
                        help="Aggregated summary CSV filename.")
    parser.add_argument("--episodes-out-name", default="RL_perf_episodes.csv",
                        help="Per-episode long-format CSV filename (feeds stat tests).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable info-level logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    # Map short CLI algo names back to the *_Models directory keys that
    # drive discovery/organization internally.
    name_to_dir = {v: k for k, v in CLI_ALGO_NAME.items()}
    algo_dirs = [name_to_dir[a] for a in args.algos]

    plotting_rl_root = args.mgym_root / "results" / "RL_baselines"

    if not args.skip_run:
        start = time.time()
        results = run_sweep(
            args.mgym_root, plotting_rl_root, algo_dirs, args.scenarios,
            args.num_episodes, args.seed_start, args.checkpoint_seed,
            args.force, args.dry_run,
        )
        elapsed = time.time() - start
        print_sweep_summary(results)
        print(f"Sweep wall time: {elapsed:.1f}s")

        if not args.dry_run and any(not r.success for r in results):
            logger.error("One or more pairs failed — see logs under %s/logs/", plotting_rl_root)

    if args.dry_run:
        return 0

    episodes = collect_per_episode(plotting_rl_root, algo_dirs, args.scenarios)
    if episodes.empty:
        logger.error("No per-episode data collected — nothing to write.")
        return 1

    summary = build_rl_summary(episodes)
    if summary.empty:
        logger.error("Summary table is empty — nothing to write.")
        return 1

    episodes_out = script_dir / args.episodes_out_name
    summary_out = script_dir / args.out_name
    episodes.to_csv(episodes_out, index=False)
    summary.to_csv(summary_out, index=False)
    print(f"Wrote {summary_out}")
    print(f"Wrote {episodes_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())