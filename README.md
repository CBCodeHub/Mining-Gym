# Mining-Gym

Mining-Gym is an open-source, configurable benchmarking environment for optimizing truck dispatch scheduling in open-pit mining using Reinforcement Learning (RL).

Paper Link: https://doi.org/10.1016/j.asoc.2026.116153

## Repository Files

**Simulator**
- `mGym_DesEnv.py` — discrete-event mining site simulator (built on Salabim)
- `mGym_GymEnv.py` — Gymnasium-compatible wrapper around the simulator
- `shared_channel.py` — connects the RL environment to the simulator each step

**Run a policy**
- `mGym_GymRun.py` — train or play RL scheduling agents
- `mGym_DefSchdRun.py` — run classical (non-RL) schedulers
- `scheduler.py` — classical scheduling algorithm implementations

**Support modules**
- `read_config.py` — samples simulation parameters from the config file
- `scenario_loader.py` — loads fixed test scenarios (A–F)
- `seed_utils.py` — shared seeding policy for reproducible episodes
- `kpi_calc.py`, `episode_metrics_logger.py` — KPI and training-metrics logging
- `results_paths.py` — centralized output file-naming convention

**Batch evaluation**
- `rl_perf_summary.py` — runs every RL algorithm × scenario and aggregates results
- `nonRL_perf_summary.py` — same, for classical schedulers
- `get_metrics.py` — turns one KPI log into plotting-ready summary CSVs

**Configuration**
- `config_extend_review.txt` — main simulation settings
- `T_scene_config.txt` — fixed scenario parameters (A–F)
- `environment.yml` — conda environment definition

## Setup

```bash
conda env create -f environment.yml
conda activate <env-name>   # name is set in environment.yml
```

## 1. Train a new RL policy

```bash
python mGym_GymRun.py train --num_episodes 10
```

- `--num_episodes`: number of training episodes
- `--algo`: which RL algorithm to train — `maskable_ppo` (default, the paper baseline), `ppo`, `a2c`, `dqn`, or `trpo`

## 2. Run a classical scheduler

```bash
python mGym_DefSchdRun.py --num_episodes 10 --algo_choice 1
python mGym_DefSchdRun.py --num_episodes 10 --algo_choice 1 --scenario D
```

- `--num_episodes`: number of episodes to simulate
- `--algo_choice`: `1` = random, `2` = fixed round-robin, `3` = shortest-queue-first, `4` = min-shovel-wait-time
- `--scenario`: fixed test-time configuration, `A`–`F`

## 3. Play a pretrained model

```bash
python mGym_GymRun.py play --num_episodes 5 --model_path <path_to_saved_model.zip>
```

Replace `<path_to_saved_model.zip>` with your saved checkpoint. If it wasn't trained with the default algorithm, also pass `--algo` so it loads correctly (e.g. `--algo a2c`).

## 4. Evaluate everything at once

To run every algorithm against every scenario and get summary CSVs instead of calling the commands above by hand:

```bash
python rl_perf_summary.py -v       # sweeps all trained RL checkpoints x scenarios A-F
python nonRL_perf_summary.py -v    # sweeps all classical schedulers x scenarios A-F
```

Both write a `*_perf_summ.csv` (mean/std per scenario) and a `*_perf_episodes.csv` (one row per episode) for downstream comparison and plotting.

## 5. Change configuration data

Edit `config_extend_review.txt` for general simulation settings (fleet size, distributions, and other simulation parameters), or `T_scene_config.txt` for the fixed `A`–`F` scenario overrides used by `--scenario`.

Refer to `cli_reference.md` for details on cli options.
