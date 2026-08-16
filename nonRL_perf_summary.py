"""
Sweep classical (non-RL) scheduling baselines across scenarios via
`mGym_DefSchdRun.py`, then aggregate the resulting per-episode Pvol CSVs
into a scenario-wise summary -- the non-RL counterpart to
rl_perf_summary.py.

Produces, under --out-dir (default: script dir):
  1) non_RL_perf_summ.csv     -- one row per (Scenario, Algorithm), with
                                  mean/std of KPI0_PVol and Mean_Queue_Length.
  2) non_RL_perf_episodes.csv -- long format, one row per episode, for
                                  downstream statistical tests.

Input schema (per Pvol file, one row per episode):
    Episodes, Episode_Seed, KPI0_PVol, Mean_Queue_Length

Seeding mirrors seed_utils.resolve_episode_seeds: exactly one of
--seed / --seed-start. Default is --seed-start 0 (LCG anchor, distinct
seed per episode -- for mean/std baselines); --seed N fixes every
episode to the same seed (deterministic replay, for debugging).

The sweep is intentionally serial: mGym_DefSchdRun.py reads/writes
alloc.json at cwd, which would race under parallel invocations.

Usage
-----
    python nonRL_perf_summary.py                     # full sweep + aggregate
    python nonRL_perf_summary.py --dry-run -v         # preview planned runs
    python nonRL_perf_summary.py --algos sqf mswt     # subset of algos
    python nonRL_perf_summary.py --scenarios A F      # subset of scenarios
    python nonRL_perf_summary.py --force              # re-run even if output exists
    python nonRL_perf_summary.py --skip-run           # aggregate only, no sim
    python nonRL_perf_summary.py --seed 42            # fixed-seed replay mode
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Single source of truth for algo numeric choice <-> tag <-> display name.
# Kept in sync with results_paths.classical_algo_tag() (python) and
# ALGO_TAG_MAP in run_all_classic.sh (bash).

REQUIRED_COLUMNS = {"Episodes", "Episode_Seed", "KPI0_PVol", "Mean_Queue_Length"}

PVOL_FILENAME_RE = re.compile(
    r"^SchdSchm\d+_(?P<algo>[A-Za-z0-9]+)_scen(?P<scen>[A-F])_seed(?P<seed>\d+)_Pvol\.csv$"
)

# Matches SchdSchm numbering (1=rnd, 2=fixed, 3=sqf, 4=mswt).
ALGO_DISPLAY_ORDER = ["rnd", "fixed", "sqf", "mswt"]
ALGO_DISPLAY_NAMES = {
    "rnd": "Random",
    "fixed": "Fixed",
    "sqf": "SQF",
    "mswt": "MSWT",
}
# tag -> mGym_DefSchdRun.py --algo_choice int
ALGO_CHOICE: dict[str, int] = {"rnd": 1, "fixed": 2, "sqf": 3, "mswt": 4}

SCENARIOS_ALL = ["A", "B", "C", "D", "E", "F"]

SCRIPT_NAME = "mGym_DefSchdRun.py"


# Path helpers

def pvol_filename(algo: str, scenario: str, seed_value: int) -> str:
    """Canonical Pvol CSV filename produced by mGym_DefSchdRun.py."""
    return (
        f"SchdSchm{ALGO_CHOICE[algo]}_{algo}"
        f"_scen{scenario}_seed{seed_value}_Pvol.csv"
    )


# Execution: one subprocess per (algo, scenario) pair, serial.

@dataclass(frozen=True)
class RunResult:
    algo: str
    scenario: str
    success: bool
    message: str
    elapsed_s: float = 0.0


def run_one_pair(
    script_path: Path,
    mgym_root: Path,
    algo: str,
    scenario: str,
    num_episodes: int,
    seed_flag: str,
    seed_value: int,
    non_rl_dir: Path,
    log_dir: Path,
    force: bool,
) -> RunResult:
    """
    Invoke `mGym_DefSchdRun.py` for one (algorithm, scenario) pair. Uses
    the predictable Pvol CSV filename as the "did this pair complete?"
    proxy -- no snapshot-diff needed.
    """
    expected_pvol = non_rl_dir / pvol_filename(algo, scenario, seed_value)
    run_tag = f"algo{ALGO_CHOICE[algo]}_{algo}_scen{scenario}_seed{seed_value}"

    if expected_pvol.exists() and not force:
        return RunResult(
            algo=algo, scenario=scenario, success=True,
            message="skipped (already exists)",
        )

    log_file = log_dir / f"{run_tag}.log"
    cmd = [
        sys.executable, str(script_path),
        "--num_episodes", str(num_episodes),
        "--algo_choice", str(ALGO_CHOICE[algo]),
        "--scenario", scenario,
        seed_flag, str(seed_value),
    ]
    logger.info("Running: %s", " ".join(cmd))

    start = time.time()
    try:
        with log_file.open("w") as lf:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(mgym_root),  # config_extend_review.txt is a bare
                                     # relative path, so cwd must be project root.
                check=False,
            )
    except OSError as e:
        return RunResult(
            algo=algo, scenario=scenario, success=False,
            message=f"subprocess launch failed: {e}",
            elapsed_s=time.time() - start,
        )
    elapsed = time.time() - start

    # Exit code alone isn't sufficient -- verify the output artifact landed.
    if proc.returncode != 0:
        return RunResult(
            algo=algo, scenario=scenario, success=False,
            message=f"exit {proc.returncode} (see {log_file.name})",
            elapsed_s=elapsed,
        )
    if not expected_pvol.exists():
        return RunResult(
            algo=algo, scenario=scenario, success=False,
            message=f"rc=0 but no Pvol CSV at {expected_pvol.name}",
            elapsed_s=elapsed,
        )
    return RunResult(
        algo=algo, scenario=scenario, success=True,
        message="ok", elapsed_s=elapsed,
    )


def run_sweep(
    script_path: Path,
    mgym_root: Path,
    non_rl_dir: Path,
    algos: list[str],
    scenarios: list[str],
    num_episodes: int,
    seed_flag: str,
    seed_value: int,
    force: bool,
    dry_run: bool,
) -> list[RunResult]:
    """Sweep every (algo, scenario) pair serially, returning per-pair results."""
    log_dir = non_rl_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(a, s) for a in algos for s in scenarios]
    results: list[RunResult] = []

    for idx, (algo, scenario) in enumerate(pairs, start=1):
        run_tag = f"algo{ALGO_CHOICE[algo]}_{algo}_scen{scenario}_seed{seed_value}"
        print(f"\n[{idx}/{len(pairs)}] {run_tag}")

        if dry_run:
            print(f"  DRY-RUN would run: {SCRIPT_NAME} "
                  f"--algo_choice {ALGO_CHOICE[algo]} --scenario {scenario} "
                  f"--num_episodes {num_episodes} {seed_flag} {seed_value}")
            continue

        try:
            result = run_one_pair(
                script_path, mgym_root, algo, scenario, num_episodes,
                seed_flag, seed_value, non_rl_dir, log_dir, force,
            )
        except KeyboardInterrupt:
            print("\nSweep interrupted; writing partial summary...", file=sys.stderr)
            break

        status = "SKIP" if result.message.startswith("skipped") else (
            "PASS" if result.success else "FAIL"
        )
        print(f"  {status}  ({result.message}, {result.elapsed_s:.1f}s)")
        results.append(result)

    return results


def print_sweep_summary(results: list[RunResult]) -> None:
    passed = [r for r in results if r.success and not r.message.startswith("skipped")]
    skipped = [r for r in results if r.message.startswith("skipped")]
    failed = [r for r in results if not r.success]

    print("\n" + "=" * 60)
    print(f"Classical sweep summary: {len(results)} pair(s)")
    print(f"  Passed:  {len(passed)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"  Failed:  {len(failed)}")
    if failed:
        print("\nFailed pairs:")
        for r in failed:
            print(f"  {r.algo} / {r.scenario}: {r.message}")
    print("=" * 60)


# Discovery: used post-sweep to feed the aggregation step.

@dataclass(frozen=True)
class PvolFile:
    algo: str
    scenario: str
    seed: int
    path: Path


def discover_pvol_files(non_rl_dir: Path) -> list[PvolFile]:
    """Find every SchdSchm*_Pvol.csv under non_rl_dir."""
    found: list[PvolFile] = []
    for path in non_rl_dir.glob("SchdSchm*_Pvol.csv"):
        m = PVOL_FILENAME_RE.match(path.name)
        if not m:
            logger.debug("Skipping unrecognized filename: %s", path.name)
            continue
        found.append(
            PvolFile(
                algo=m["algo"], scenario=m["scen"],
                seed=int(m["seed"]), path=path,
            )
        )
    logger.info("Discovered %d Pvol files under %s", len(found), non_rl_dir)
    return found


# Loading & aggregation

def load_pvol_file(path: Path) -> pd.DataFrame:
    """Read and validate a single Pvol CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )
    return df


def build_tables(
    pvol_files: list[PvolFile], scenarios: list[str], algos: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the aggregated (Scenario, Algorithm) summary table and the
    per-episode long-format table in one pass. Multiple seed files for
    the same (algo, scenario) are pooled and episodes renumbered 1..N.
    Missing (algo, scenario) combinations are skipped with a warning.
    """
    summary_rows: list[dict[str, object]] = []
    episode_frames: list[pd.DataFrame] = []

    for scenario in scenarios:
        for algo in algos:
            matches = [
                f for f in pvol_files if f.algo == algo and f.scenario == scenario
            ]
            if not matches:
                logger.warning(
                    "No Pvol file for algo=%s scenario=%s — skipping", algo, scenario
                )
                continue

            frames = []
            for m in matches:
                try:
                    frames.append(load_pvol_file(m.path))
                except ValueError as e:
                    logger.warning("Failed to load %s: %s", m.path.name, e)
            if not frames:
                continue

            pooled = pd.concat(frames, ignore_index=True)
            display_name = ALGO_DISPLAY_NAMES[algo]

            summary_rows.append({
                "Scenario": scenario,
                "Algorithm": display_name,
                "Mean_PVol": pooled["KPI0_PVol"].mean(),
                "Std_PVol": pooled["KPI0_PVol"].std(),
                "N_Episodes": len(pooled),
                "Mean_Queue_Length_Mean": pooled["Mean_Queue_Length"].mean(),
                "Mean_Queue_Length_Std": pooled["Mean_Queue_Length"].std(),
            })

            episode_frames.append(pd.DataFrame({
                "Scenario": scenario,
                "Algorithm": display_name,
                "Episode": range(1, len(pooled) + 1),
                # Renamed to match the RL-side schema for concatenation.
                "Final_Production_Volume": pooled["KPI0_PVol"].values,
                "Mean_Shovel_Queue": pooled["Mean_Queue_Length"].values,
            }))

    summary_df = pd.DataFrame(summary_rows)
    episodes_df = (
        pd.concat(episode_frames, ignore_index=True)
        if episode_frames
        else pd.DataFrame(columns=["Scenario", "Algorithm", "Episode",
                                    "Final_Production_Volume", "Mean_Shovel_Queue"])
    )
    return summary_df, episodes_df


# CLI

def _resolve_seed_mode(seed: Optional[int], seed_start: Optional[int]) -> tuple[str, int]:
    """
    Mirror seed_utils.resolve_episode_seeds: exactly one of seed / seed_start.
    Returns (flag_for_subprocess, value) — e.g. ("--seed_start", 0).
    """
    if seed is not None and seed_start is not None:
        raise ValueError("--seed and --seed-start are mutually exclusive.")
    if seed is not None:
        return "--seed", seed
    # Default: seed_start=0 (matches mGym_DefSchdRun.py's default).
    return "--seed_start", seed_start if seed_start is not None else 0


def main(argv: Optional[list[str]] = None) -> int:
    script_dir = Path(__file__).resolve().parent
    default_results_dir = script_dir.parent  # plotting/ -> results/
    default_mgym_root = default_results_dir.parent  # results/ -> mgym/

    parser = argparse.ArgumentParser(
        description="Sweep classical schedulers across scenarios and summarize performance.",
    )
    # --- Paths ---
    parser.add_argument("--mgym-root", type=Path, default=default_mgym_root,
                        help=f"Project root containing {SCRIPT_NAME} (run cwd).")
    parser.add_argument("--results-dir", type=Path, default=default_results_dir,
                        help="Root results directory containing non-RL_baselines/.")
    parser.add_argument("--out-dir", type=Path, default=script_dir,
                        help="Directory to write output CSVs into.")
    parser.add_argument("--out-name", default="non_RL_perf_summ.csv",
                        help="Aggregated summary CSV filename.")
    parser.add_argument("--episodes-out-name", default="non_RL_perf_episodes.csv",
                        help="Per-episode long-format CSV filename (feeds stat tests).")

    # --- Sweep selection ---
    parser.add_argument("--algos", nargs="+", choices=ALGO_DISPLAY_ORDER,
                        default=ALGO_DISPLAY_ORDER,
                        help="Which classical algorithms to sweep.")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS_ALL,
                        default=SCENARIOS_ALL, help="Which scenarios to sweep.")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Episodes per (algo, scenario) pair.")

    # --- Seeding (mutually exclusive) ---
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None,
                            help="Fixed seed for every episode (deterministic replay).")
    seed_group.add_argument("--seed-start", type=int, default=None,
                            help="LCG anchor for per-episode seeds. Default: 0.")

    # --- Sweep control ---
    parser.add_argument("--force", action="store_true",
                        help="Re-run pairs even if the Pvol CSV already exists.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs without executing them.")
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip the simulation sweep; aggregate existing output only.")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable info-level logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    try:
        seed_flag, seed_value = _resolve_seed_mode(args.seed, args.seed_start)
    except ValueError as e:
        parser.error(str(e))

    non_rl_dir = args.results_dir / "non-RL_baselines"
    non_rl_dir.mkdir(parents=True, exist_ok=True)

    # --- Sweep phase ---------------------------------------------------
    if not args.skip_run:
        script_path = args.mgym_root / SCRIPT_NAME
        if not script_path.is_file():
            logger.error("%s not found at %s", SCRIPT_NAME, script_path)
            return 2

        start = time.time()
        results = run_sweep(
            script_path=script_path,
            mgym_root=args.mgym_root,
            non_rl_dir=non_rl_dir,
            algos=args.algos,
            scenarios=args.scenarios,
            num_episodes=args.num_episodes,
            seed_flag=seed_flag,
            seed_value=seed_value,
            force=args.force,
            dry_run=args.dry_run,
        )
        elapsed = time.time() - start

        if not args.dry_run:
            print_sweep_summary(results)
            print(f"Sweep wall time: {elapsed:.1f}s")
            if any(not r.success for r in results):
                logger.error(
                    "One or more pairs failed — see logs under %s/logs/",
                    non_rl_dir,
                )

    if args.dry_run:
        return 0

    # --- Aggregation phase --------------------------------------------
    pvol_files = discover_pvol_files(non_rl_dir)
    if not pvol_files:
        logger.error("No Pvol files found under %s", non_rl_dir)
        return 1

    summary, episodes = build_tables(pvol_files, args.scenarios, args.algos)
    if summary.empty:
        logger.error("Summary table is empty — nothing to write.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_out = args.out_dir / args.out_name
    episodes_out = args.out_dir / args.episodes_out_name
    summary.to_csv(summary_out, index=False)
    episodes.to_csv(episodes_out, index=False)
    print(f"Wrote {summary_out}")
    print(f"Wrote {episodes_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())