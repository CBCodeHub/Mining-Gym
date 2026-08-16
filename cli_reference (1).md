## Setup

```bash
conda env create -f environment.yml
conda activate <env-name>
```

---

## 1. Train an RL policy

```bash
python mGym_GymRun.py train --num_episodes 10
```

| Flag | Default | Description |
|---|---|---|
| `--num_episodes` | `5000` | Total training episodes. |
| `--algo` | `maskable_ppo` | `maskable_ppo` \| `ppo` \| `a2c` \| `dqn` \| `trpo`. |
| `--repeat` | `1` | Number of from-scratch training runs (one per seed). |
| `--seed` | — | Fixed seed for every repeat. Mutually exclusive with `--seed_start`. |
| `--seed_start` | `0` | LCG anchor: repeat `i` uses `gen_seed(i, seed_start)`. |
| `--no_early_stop` | off | Disable patience-based early stopping; run the full `--num_episodes`. |
| `--eval_every_episodes` | `2000` | Run a deterministic eval batch every N episodes. |
| `--num_eval_episodes` | `3` | Eval episodes per checkpoint, per scenario (eval runs scenarios A + F). |
| `--patience` | `10` | Evals without improvement before early stopping. |
| `--improvement_margin` | `0.01` | Relative composite-score improvement required for new-best. |
| `--model_path` | — | Resume training from a checkpoint. Requires `--repeat 1`. |

```bash
# Different algorithm
python mGym_GymRun.py train --algo a2c --num_episodes 5000

# 3 independent seeds in one sweep
python mGym_GymRun.py train --repeat 3 --seed_start 0 --num_episodes 10000

# Full budget, no early stopping
python mGym_GymRun.py train --no_early_stop --num_episodes 20000

# Resume from a checkpoint
python mGym_GymRun.py train --model_path <checkpoint>.zip --num_episodes 5000
```

---

## 2. Play / evaluate a trained RL policy

```bash
python mGym_GymRun.py play --num_episodes 5 --model_path <path_to_saved_model.zip>
```

| Flag | Default | Description |
|---|---|---|
| `--num_episodes` | `5000` | Episodes to play. |
| `--model_path` | — | Required. Path to the saved `.zip` checkpoint. |
| `--algo` | `maskable_ppo` | Must match the algorithm the checkpoint was trained with. |
| `--scenario` | `None` | Fixed test-time scenario `A`–`F`. |
| `--seed` | `49` | Fixed seed for every episode. Mutually exclusive with `--seed_start`. |
| `--seed_start` | — | LCG anchor: episode `i` uses `gen_seed(i, seed_start)`. |

```bash
python mGym_GymRun.py play \
  --num_episodes 10 \
  --model_path MASKABLE_PPO_Models/run_2026-07-08_07-56-45/seed_1013904223/best/ppo_minegym_best_scenarioAF.zip \
  --scenario F \
  --seed_start 0
```

---

## 3. Classical (non-RL) scheduler baselines

```bash
python mGym_DefSchdRun.py --num_episodes 10 --algo_choice 1
python mGym_DefSchdRun.py --num_episodes 10 --algo_choice 1 --scenario D
```

| `--algo_choice` | Algorithm |
|---|---|
| `1` | Random selection |
| `2` | Fixed (static round-robin) |
| `3` | Shortest-queue-first (SQF) |
| `4` | Min-shovel-waiting-time (MSWT) |

| Flag | Default | Description |
|---|---|---|
| `--num_episodes` | `10` | Episodes to simulate. |
| `--algo_choice` | — | Required. See table above. |
| `--scenario` | `None` | Fixed test-time scenario `A`–`F`. |
| `--seed` | — | Fixed seed for every episode. Mutually exclusive with `--seed_start`. |
| `--seed_start` | `0` | LCG anchor: episode `i` uses `gen_seed(i, seed_start)`. |

```bash
python mGym_DefSchdRun.py --num_episodes 10 --algo_choice 3 --scenario A --seed_start 0
python mGym_DefSchdRun.py --num_episodes 10 --algo_choice 3 --seed 42
```

---

## 4. Batch sweeps + aggregation

### RL sweep

```bash
python rl_perf_summary.py -v
```

| Flag | Default | Description |
|---|---|---|
| `--algos` | `a2c dqn maskable_ppo trpo` | Algorithms to sweep. |
| `--scenarios` | `A B C D E F` | Scenarios to sweep. |
| `--num-episodes` | `10` | Episodes per (algo, scenario) pair. |
| `--seed-start` | `0` | LCG seed anchor passed through as `--seed_start`. |
| `--checkpoint-seed` | `1013904223` | Training seed of the checkpoint to load (`seed_{N}/best`). |
| `--force` | off | Re-run pairs even if output already exists. |
| `--dry-run` | off | Print planned runs without executing. |
| `--skip-run` | off | Skip the sweep; aggregate existing output only. |
| `--out-name` | `RL_perf_summ.csv` | Aggregated summary CSV filename. |
| `--episodes-out-name` | `RL_perf_episodes.csv` | Per-episode long-format CSV filename. |
| `-v`, `--verbose` | off | Info-level logging. |

```bash
python rl_perf_summary.py --dry-run -v
python rl_perf_summary.py --algos maskable_ppo --scenarios F
python rl_perf_summary.py --checkpoint-seed 42
python rl_perf_summary.py --skip-run
```

Writes `RL_perf_summ.csv` and `RL_perf_episodes.csv` under `results/plotting/`.

### Classical scheduler sweep

```bash
python nonRL_perf_summary.py -v
```

| Flag | Default | Description |
|---|---|---|
| `--algos` | `rnd fixed sqf mswt` | Algorithms to sweep. |
| `--scenarios` | `A B C D E F` | Scenarios to sweep. |
| `--num-episodes` | `10` | Episodes per (algo, scenario) pair. |
| `--seed` | — | Fixed seed for every episode. Mutually exclusive with `--seed-start`. |
| `--seed-start` | `0` | LCG anchor for per-episode seeds. |
| `--force` | off | Re-run pairs even if output already exists. |
| `--dry-run` | off | Print planned runs without executing. |
| `--skip-run` | off | Skip the sweep; aggregate existing output only. |
| `--out-name` | `non_RL_perf_summ.csv` | Aggregated summary CSV filename. |
| `--episodes-out-name` | `non_RL_perf_episodes.csv` | Per-episode long-format CSV filename. |
| `-v`, `--verbose` | off | Info-level logging. |

```bash
python nonRL_perf_summary.py --dry-run -v
python nonRL_perf_summary.py --algos sqf mswt --scenarios A F
python nonRL_perf_summary.py --seed 42
```

Writes `non_RL_perf_summ.csv` and `non_RL_perf_episodes.csv` in the same schema as the RL sweep.

### Post-process a single KPI log

```bash
python get_metrics.py kpi_log_test_shared.csv --out-prefix rl_ckpt2k -v
```

| Flag | Default | Description |
|---|---|---|
| `kpi_log` | — | Required. Path to the multi-episode KPI log CSV. |
| `--out-prefix` | input filename stem | Prefix for output files. |
| `-v`, `--verbose` | off | Info-level logging. |

Writes `<prefix>_hourly.csv`, `<prefix>_per_episode.csv`, `<prefix>_composite.csv`.

---

## 5. Configuration

| File / variable | Purpose |
|---|---|
| `config_extend_review.txt` | Main simulation config — fleet size, distributions, scheduler settings. |
| `T_scene_config.txt` | Scenario `A`–`F` overrides. |
| `MGYM_RESULTS_ROOT` (env var) | Overrides where `results/` is written. Default: `./results`. |

```bash
MGYM_RESULTS_ROOT=/data/mgym_runs python mGym_DefSchdRun.py --num_episodes 10 --algo_choice 1
```
