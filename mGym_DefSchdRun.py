"""
Driver for classical (non-RL) scheduling policies.

Seeding (matches mGym_GymRun.py):
    --seed N        Same integer N for every episode (deterministic replay).
    --seed_start N  LCG anchor; episode i uses gen_seed(i, N). Use for
                    baseline evaluation (mean +- std over episodes).
    (default)       --seed_start 0

Outputs are routed through results_paths.py:

    results/non-RL_baselines/
        kpi_log_{algo}_scen{X}_seed{Y}.csv          -- per-tick KPI log
        SchdSchm{choice}_{algo}_scen{X}_seed{Y}_Pvol.csv
                                                    -- per-episode summary
        channel_{algo}_scen{X}_seed{Y}.csv          -- placeholder path passed to runDes

Y in the filename is whichever integer was passed (--seed OR --seed_start);
the CSV's Episode_Seed column is the ground truth for per-episode seeds.
"""

import argparse
import ast
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np

import mGym_DesEnv as denv
from scenario_loader import load_scenario
from results_paths import (
    classical_algo_tag,
    kpi_log_path,
    channel_csv_path,
    results_dir,
    experiment_tag,
)
from seed_utils import resolve_episode_seeds


def save_temp_data(cfg_seed_info, data):
    with open(cfg_seed_info, 'w') as file:
        json.dump(data, file)


def _compute_mean_queue_per_episode(kpi_path):
    """
    Compute mean total shovel queue length per episode from the
    multi-episode KPI log ("total" = sum across shovels at a tick,
    "mean" = average of that sum across the episode's ticks).

    Returns {episode_idx: mean_queue}; interrupted/missing episodes are
    simply absent. Uses ast.literal_eval (not eval) on the list-string
    column so a malformed cell fails loudly instead of executing code.
    """
    if not (kpi_path.exists() and kpi_path.stat().st_size > 0):
        return {}

    sums = defaultdict(float)
    counts = defaultdict(int)

    with open(kpi_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queue_str = row.get('Shovel_Queue_Lengths', '')
            if not (isinstance(queue_str, str) and queue_str.strip().startswith('[')):
                continue
            try:
                total = sum(ast.literal_eval(queue_str))
            except (ValueError, SyntaxError):
                continue
            try:
                ep = int(row['Episode'])
            except (KeyError, ValueError, TypeError):
                continue
            sums[ep] += total
            counts[ep] += 1

    return {ep: sums[ep] / counts[ep] for ep in sums if counts[ep] > 0}


def _build_parser():
    # allow_abbrev=False: a typo like `--sed 0` fails loudly instead of
    # silently binding to whatever unique prefix it happens to match.
    parser = argparse.ArgumentParser(
        description="Run the simulation with a classical scheduling policy.",
        allow_abbrev=False,
    )
    parser.add_argument('--num_episodes', type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument('--algo_choice', type=int, required=True,
                        help="Scheduler choice (1=rnd, 2=fixed, 3=sqf, 4=mswt). "
                             "See mGym_DesEnv.scheduler_assign() for the "
                             "dispatch table (ground truth).")
    parser.add_argument('--scenario', type=str, default=None,
                        help="Test scenario (A, B, C, D, E, F)")

    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        '--seed', type=int, default=None,
        help="Fixed seed for every episode (deterministic replay). "
             "Mutually exclusive with --seed_start."
    )
    seed_group.add_argument(
        '--seed_start', type=int, default=None,
        help="LCG anchor: episode i uses gen_seed(i, seed_start). "
             "Mutually exclusive with --seed."
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # Default: LCG anchored at 0 (matches mGym_GymRun.py train's default).
    if args.seed is None and args.seed_start is None:
        args.seed_start = 0

    num_episodes = args.num_episodes
    algo_choice = args.algo_choice
    scenario_overrides = load_scenario(args.scenario)

    episode_seeds, anchor, mode = resolve_episode_seeds(
        seed=args.seed,
        seed_start=args.seed_start,
        num_episodes=num_episodes,
    )

    algo = classical_algo_tag(algo_choice)
    kpi_path = kpi_log_path("classical", algo, args.scenario, anchor)
    channel_path = channel_csv_path("classical", algo, args.scenario, anchor)

    # Fresh KPI log per invocation -- otherwise rows from a prior run
    # under the same (algo, scenario, anchor) would concatenate.
    if kpi_path.exists():
        kpi_path.unlink()

    print(f"--- Classical scheduler run ---")
    print(f"    algo tag:    {algo}  (choice={algo_choice})")
    print(f"    scenario:    {args.scenario}")
    print(f"    seed mode:   {mode}  (anchor={anchor})")
    print(f"    episodes:    {num_episodes}")
    print(f"    KPI log:     {kpi_path}")

    pvols = []
    for epsd, episode_seed in enumerate(episode_seeds):
        # Static round-robin LUT is state-on-disk from a previous run;
        # clear it so shovel/truck count changes don't feed a stale
        # allocation table into the fixed() policy.
        if os.path.exists('alloc.json'):
            os.remove('alloc.json')

        random.seed(episode_seed)
        np.random.seed(episode_seed)

        kpi_01 = denv.runDes(
            fsim=False,
            flag_RL_sched=False,
            fdef_schdlr_choice=algo_choice,
            episode_seed=episode_seed,
            scenario_overrides=scenario_overrides,
            csv_path=str(channel_path),
            scenario_name=args.scenario,
            play_seed=anchor,
            episode_idx=epsd,
            kpi_log_path=str(kpi_path),
        )
        print(f"Episode {epsd+1} Seed: {episode_seed}, "
              f"Value of KPI01-PVol: {kpi_01}")
        pvols.append(kpi_01)

    mean_kpi01 = float(np.mean(pvols)) if pvols else 0.0
    print(f"Average KPI01-PVol: {mean_kpi01}, over {num_episodes} repeats")

    # Legacy per-episode summary CSV. Episode_Seed is the seed actually
    # used (see header); Mean_Queue_Length comes from the KPI log.
    mean_queues = _compute_mean_queue_per_episode(kpi_path)

    tag = experiment_tag(algo, args.scenario, anchor)
    legacy_pvol_path = (
        results_dir("classical") / f"SchdSchm{algo_choice}_{tag}_Pvol.csv"
    )
    with open(legacy_pvol_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Episodes", "Episode_Seed", "KPI0_PVol", "Mean_Queue_Length"]
        )
        for idx, (pvol, ep_seed) in enumerate(zip(pvols, episode_seeds), 1):
            mean_q = mean_queues.get(idx - 1, "")  # KPI log uses 0-based Episode
            writer.writerow([idx, ep_seed, pvol, mean_q])
    print(f"Legacy PVOL CSV: {legacy_pvol_path}")


if __name__ == "__main__":
    main()
