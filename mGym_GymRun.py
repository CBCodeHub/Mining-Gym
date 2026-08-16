"""
Train or play RL scheduling policies for the Minegym mining-dispatch
environment (TOP of 3: GymRun -> GymEnv -> DesEnv).
"""

import argparse
import csv
import glob
import inspect
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Type

import gymnasium as gym
import numpy as np
import pandas as pd
import tensorboard
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation
from sb3_contrib import MaskablePPO, TRPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from read_config import ConfigSampler
from scenario_loader import load_scenario
from seed_utils import resolve_episode_seeds
# Training scenario pool's source of truth lives in mGym_GymEnv.
from mGym_GymEnv import TRAIN_SCENARIO_POOL as TRAIN_SCENARIO_POOL_FOR_META

# ALGO_REGISTRY: single source of truth for per-algorithm configuration.
# Adding a new sb3/sb3_contrib algorithm is a one-dict-entry change --
# everything downstream (env wrapping, model construction, checkpoint
# load, play-mode predict, argparse choices) is driven by the capability
# flags on this spec; no isinstance() branches anywhere else.
#
#   * `default_kwargs` holds only algorithm-specific hyperparameters;
#     universal kwargs (env, seed, verbose, tensorboard_log) are supplied
#     by make_model() at call time.
#   * `uses_n_steps` distinguishes on-policy rollout algos (PPO, A2C,
#     TRPO, MaskablePPO) that consume a runtime-computed n_steps from
#     off-policy algos like DQN that don't.
#   * `supports_masking` and `needs_flat_obs` drive wrap_env_for_algo();
#     independent flags so any combination is expressible.
#   * The MaskablePPO entry is byte-for-byte the paper's published
#     configuration -- do NOT edit those numbers.


@dataclass(frozen=True)
class AlgoSpec:
    """Per-algorithm capabilities and defaults. Frozen so a typo at the
    call site (e.g. spec.policy = ...) fails loudly rather than silently
    mutating the registry mid-run."""
    cls: Type
    policy: str
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    supports_masking: bool = False
    needs_flat_obs: bool = False
    uses_n_steps: bool = False  # True iff on-policy with rollout buffer


ALGO_REGISTRY: Dict[str, AlgoSpec] = {
    # Paper baseline -- DO NOT MODIFY without versioning the algo name.
    "maskable_ppo": AlgoSpec(
        cls=MaskablePPO,
        policy="MultiInputPolicy",
        default_kwargs=dict(
            learning_rate=3e-4,
            batch_size=64,
            n_epochs=5,
            gamma=0.995,
            clip_range=0.25,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.02,
            vf_coef=0.7,
            max_grad_norm=0.5,
            target_kl=0.03,
        ),
        supports_masking=True,
        needs_flat_obs=False,
        uses_n_steps=True,
    ),

    # Comparison algorithms. gamma=0.995 is unified across algos so the
    # discount factor isn't a confound; otherwise sb3 documented defaults.
    "ppo": AlgoSpec(
        cls=PPO,
        policy="MultiInputPolicy",
        default_kwargs=dict(
            learning_rate=3e-4,
            batch_size=64,
            n_epochs=5,
            gamma=0.995,
            clip_range=0.25,
            ent_coef=0.02,
            vf_coef=0.7,
            max_grad_norm=0.5,
        ),
        supports_masking=False,
        needs_flat_obs=False,
        uses_n_steps=True,
    ),
    "a2c": AlgoSpec(
        cls=A2C,
        policy="MultiInputPolicy",
        default_kwargs=dict(
            learning_rate=7e-4,
            n_steps=5,  # A2C convention: short rollouts; overrides the
                        # runtime n_steps_calculated injection (see make_model).
            gamma=0.995,
            ent_coef=0.02,
            vf_coef=0.7,
            max_grad_norm=0.5,
        ),
        supports_masking=False,
        needs_flat_obs=False,
        uses_n_steps=False,           # has its own fixed n_steps in defaults
    ),
    "dqn": AlgoSpec(
        cls=DQN,
        policy="MlpPolicy",           # DQN's standard recipe: flat obs.
        default_kwargs=dict(
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.995,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
        ),
        supports_masking=False,
        needs_flat_obs=True,
        uses_n_steps=False,
    ),
    "trpo": AlgoSpec(
        cls=TRPO,
        policy="MultiInputPolicy",
        default_kwargs=dict(
            learning_rate=1e-3,
            batch_size=128,
            gamma=0.995,
            target_kl=0.01,
        ),
        supports_masking=False,
        needs_flat_obs=False,
        uses_n_steps=True,
    ),
}


def get_spec(algo: str) -> AlgoSpec:
    """Look up an algo spec, with a friendly error listing valid names."""
    key = algo.lower()
    if key not in ALGO_REGISTRY:
        valid = ", ".join(sorted(ALGO_REGISTRY.keys()))
        raise ValueError(f"Unknown algo {algo!r}. Valid: {valid}")
    return ALGO_REGISTRY[key]


def wrap_env_for_algo(env: gym.Env, spec: AlgoSpec) -> gym.Env:
    """
    Apply algorithm-conditional env wrappers. Used by the training env
    (train_one_seed), eval env (run_deterministic_eval), and play env, so
    wrapping rules can't drift between paths.

    Order: FlattenObservation first (if needed; masking doesn't care
    about obs structure), then ActionMasker (if supported; must be
    outermost so the model's predict()/rollout collector sees the mask
    interface).
    """
    if spec.needs_flat_obs:
        env = FlattenObservation(env)
    if spec.supports_masking:
        if not hasattr(env.unwrapped, "action_masks"):
            raise RuntimeError(
                f"Algo requires masking but env {type(env.unwrapped).__name__} "
                "does not expose action_masks(). Did you forget to implement it?"
            )

        def _mask_fn(e):
            return e.unwrapped.action_masks()
        env = ActionMasker(env, _mask_fn)
    return env


def make_model(
    algo: str,
    env,
    *,
    seed: int,
    tb_log: str,
    n_steps_calculated: Optional[int] = None,
    model_path: Optional[str] = None,
):
    """
    Construct or load a model using the registry spec. `model_path`
    triggers cls.load(); otherwise a fresh instance is built from
    policy + default_kwargs + (n_steps if applicable). Universal kwargs
    (env, seed, verbose, tensorboard_log) are merged here.

    `n_steps_calculated` is injected only for algos with uses_n_steps=True
    that don't already pin n_steps in default_kwargs (A2C pins n_steps=5,
    so its entry has uses_n_steps=False to skip injection).
    """
    spec = get_spec(algo)

    if model_path:
        # cls.load's signature is uniform across sb3 algos.
        return spec.cls.load(
            model_path, env=env, verbose=2, tensorboard_log=tb_log
        )

    kwargs = dict(spec.default_kwargs)
    if spec.uses_n_steps and "n_steps" not in kwargs:
        if n_steps_calculated is None:
            raise ValueError(
                f"Algo {algo!r} requires n_steps but none was provided."
            )
        kwargs["n_steps"] = n_steps_calculated

    return spec.cls(
        spec.policy,
        env,
        verbose=2,
        tensorboard_log=tb_log,
        seed=seed,
        device="auto",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Algorithm-agnostic prediction helper
# ---------------------------------------------------------------------------
# The composite-score early stopping + best-model selection is intentionally
# decoupled from the trainer: it only needs model.predict() and the env's
# terminal info dict. Mask support is detected via signature inspection so
# vanilla SB3 algorithms (which raise TypeError on action_masks kwarg) and
# MaskablePPO both work without isinstance checks.
def _supports_action_masks(model) -> bool:
    """True iff model.predict accepts an action_masks kwarg."""
    try:
        sig = inspect.signature(model.predict)
        return "action_masks" in sig.parameters
    except (TypeError, ValueError):
        return False


def _predict_action(model, obs, env, *, deterministic: bool):
    """Single predict() call site for eval/play.

    Passes action_masks only if (a) the model supports it AND (b) the env
    exposes action_masks(). Algorithm-agnostic: plug in PPO/SAC/A2C/DQN/TRPO
    and it Just Works; plug in MaskablePPO with a masked env and masking is
    honored.
    """
    if _supports_action_masks(model):
        try:
            masks = env.unwrapped.action_masks()
            action, state = model.predict(obs, deterministic=deterministic,
                                          action_masks=masks)
            return action, state
        except AttributeError:
            pass
    action, state = model.predict(obs, deterministic=deterministic)
    return action, state


# Define evaluation parameters directly in the script.
DEFAULT_EVAL_EPISODES = 0
DEFAULT_EVAL_INTERVAL = 0


def register_minegym():
    """Register the Minegym environment with Gymnasium."""
    try:
        register(
            id='Minegym-v0',
            entry_point='mGym_GymEnv:Minegym',
        )
        print("Environment registered successfully!")
    except Exception as e:
        print(f"Failed to register environment: {e}")


def gen_seed(iteration, initial_seed=44, ax=1664525, cx=1013904223, mx=2**32):
    """Generate a seed via LCG."""
    epi_seed = initial_seed
    for tx in range(iteration):
        epi_seed = (ax * epi_seed + cx) % mx
    return epi_seed


class EntropyScheduleCallback(BaseCallback):
    """
    Linearly decays ``model.ent_coef`` from ``start`` to ``end`` over
    ``total_episodes`` training episodes, then holds at ``end``.

    Algorithm-agnostic: silently no-ops for algorithms without an
    ent_coef attribute (DQN, TRPO). Writes to model.ent_coef only, no
    PPO-specific state touched.
    """

    def __init__(
        self,
        start: float = 0.02,
        end: float = 0.003,
        total_episodes: int = 15000,
        episode_counter_cb=None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start = float(start)
        self.end = float(end)
        self.total_episodes = max(1, int(total_episodes))
        self._episode_counter_cb = episode_counter_cb
        self._last_logged_coef = None
        self._warned_no_ent_coef = False

    def _current_episode(self) -> int:
        cb = self._episode_counter_cb
        if cb is not None and hasattr(cb, "episode_count"):
            try:
                return int(cb.episode_count)
            except (TypeError, ValueError):
                pass
        return int(self.num_timesteps / 250)

    def _on_step(self) -> bool:
        # Capability guard: DQN and TRPO have no ent_coef. No-op cleanly.
        if not hasattr(self.model, "ent_coef"):
            if not self._warned_no_ent_coef and self.verbose >= 1:
                print(f"[EntropySchedule] model {type(self.model).__name__} "
                      "has no ent_coef; schedule is a no-op for this algo.")
                self._warned_no_ent_coef = True
            return True

        ep = self._current_episode()
        frac = min(1.0, max(0.0, ep / self.total_episodes))
        new_coef = self.start + (self.end - self.start) * frac
        self.model.ent_coef = new_coef

        if self.verbose >= 1 and (
            self._last_logged_coef is None
            or abs(new_coef - self._last_logged_coef) > 1e-4
        ):
            print(f"[EntropySchedule] ep={ep} ent_coef={new_coef:.5f}")
            self._last_logged_coef = new_coef
        return True


class MultiScenarioEvalEarlyStop(BaseCallback):
    """
    Best-so-far + patience early stopping on a multi-scenario ratio composite:

        C = (1/|S|) * sum_s [ PVOL_WEIGHT * (PVol_s / PVol*_s)
                             + QUEUE_WEIGHT * (Q*_s / Q_s) ]

    where PVol_s/Q_s are this eval round's mean PVol / mean total shovel
    queue on scenario s (over num_eval_episodes), and PVol*_s/Q*_s are the
    running best (max PVol / min queue) on s across the run so far.


    Algorithm-agnostic: reads only model.predict() through
    run_deterministic_eval; mask support is auto-detected.
    """

    # |S|-normalization comes from averaging over scenarios, so weights
    # need not sum to 1 -- but doing so keeps the composite in [0, 1].
    PVOL_WEIGHT = 0.5
    QUEUE_WEIGHT = 0.5
    QUEUE_EPS = 1e-6  # guards divide-by-zero if mean queue is ever exactly 0

    def __init__(
        self,
        model: Any,
        eval_env_id: str,
        eval_scenarios: Dict[str, dict],   # ordered: scen_name -> overrides
        save_dir: str,
        algo: str,
        eval_every_episodes: int = 2000,
        num_eval_episodes: int = 3,
        improvement_margin: float = 0.01,
        patience: int = 10,
        seed: int = -1,
        verbose: int = 1,
        early_stop_enabled: bool = True,
    ):
        super().__init__(verbose)
        if not eval_scenarios:
            raise ValueError(
                "MultiScenarioEvalEarlyStop requires at least one eval scenario."
            )
        for name, overrides in eval_scenarios.items():
            if not overrides:
                raise ValueError(
                    f"Eval scenario {name!r} has empty overrides -- likely a "
                    "missing [SCENARIO_X] section in T_scene_config.txt."
                )

        self.model = model
        self.eval_env_id = eval_env_id
        # Insertion order (preserved on Py3.7+) is used as both the
        # composite's scenario iteration order and the filename tag order.
        self.eval_scenarios: Dict[str, dict] = dict(eval_scenarios)
        self.save_dir = save_dir
        self.algo = algo
        self.eval_every_episodes = eval_every_episodes
        self.num_eval_episodes = num_eval_episodes
        self.improvement_margin = improvement_margin
        self.patience = patience
        self.seed = seed
        self.early_stop_enabled = early_stop_enabled

        # Sorted scenario letters so the filename is deterministic
        # regardless of dict insertion order.
        self._scenario_tag = "".join(sorted(self.eval_scenarios.keys()))
        if algo.lower() == "maskable_ppo":
            self._best_model_filename = (
                f"ppo_minegym_best_scenario{self._scenario_tag}.zip"
            )
            self._best_vecnorm_filename = (
                f"vecnormalize_best_scenario{self._scenario_tag}.pkl"
            )
        else:
            self._best_model_filename = (
                f"{algo}_best_scenario{self._scenario_tag}.zip"
            )
            self._best_vecnorm_filename = (
                f"{algo}_vecnormalize_best_scenario{self._scenario_tag}.pkl"
            )

        # Per-scenario running bests. Initialize with sentinels that any
        # observed value will beat: PVol max starts at -inf, queue min at
        # +inf. First eval round becomes the reference.
        self._running_best_pvol: Dict[str, float] = {
            s: -np.inf for s in self.eval_scenarios
        }
        self._running_best_queue: Dict[str, float] = {
            s: np.inf for s in self.eval_scenarios
        }

        # Incumbent's raw per-scenario (pvol, queue); None until the first
        # eval fires. Raw (not composite) numbers enable self-consistent
        # re-scoring against drifting denominators -- see class docstring.
        self._incumbent_raw: Optional[Dict[str, Dict[str, float]]] = None

        # best_composite is the incumbent's composite re-scored against
        # the current running bests (updated each round for TB), not a
        # frozen historical value.
        self.best_composite: float = -np.inf
        self.best_checkpoint_episode: Optional[int] = None
        self.evals_without_improvement: int = 0

        self.episode_count: int = 0
        self._last_eval_episode: int = 0
        self.eval_history: List[Dict[str, Any]] = []

    def _composite_from_raw(
        self, raw: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Compute the composite from a {scen: {'pvol': .., 'queue': ..}} dict
        against the current running-best denominators. Idempotent: two
        calls with the same raw dict and running-best state agree, which
        the re-scoring approach relies on.
        """
        parts: List[float] = []
        for scen in self.eval_scenarios:
            pvol = float(raw[scen]['pvol'])
            queue = max(float(raw[scen]['queue']), self.QUEUE_EPS)
            pvol_best = self._running_best_pvol[scen]
            queue_best = self._running_best_queue[scen]

            # pvol_best == -inf only before any update; treat ratio as 0
            # (defensive -- _update_running_bests always runs first).
            pvol_ratio = (pvol / pvol_best) if pvol_best > 0 else 0.0
            queue_ratio = (queue_best / queue) if np.isfinite(queue_best) else 0.0

            parts.append(
                self.PVOL_WEIGHT * pvol_ratio
                + self.QUEUE_WEIGHT * queue_ratio
            )
        return float(np.mean(parts))

    def _update_running_bests(
        self, current_raw: Dict[str, Dict[str, float]],
    ) -> None:
        """In-place update of PVol*_s (max) and Q*_s (min) per scenario."""
        for scen, comp in current_raw.items():
            pvol = float(comp['pvol'])
            queue = float(comp['queue'])
            if pvol > self._running_best_pvol[scen]:
                self._running_best_pvol[scen] = pvol
            if queue < self._running_best_queue[scen]:
                self._running_best_queue[scen] = queue

    def _is_new_best(
        self, candidate_composite: float, incumbent_composite: float,
    ) -> bool:
        """
        Relative-margin comparison: candidate must beat incumbent by
        improvement_margin (fraction of incumbent's magnitude). The first
        successful eval (incumbent = -inf) always wins if candidate is finite.
        """
        if not np.isfinite(incumbent_composite):
            return np.isfinite(candidate_composite)
        required = (
            incumbent_composite
            + abs(incumbent_composite) * self.improvement_margin
        )
        return candidate_composite > required

    # Eval CSV
    _RAW_FIELDS = [
        "seed", "checkpoint_episode", "scenario", "eval_idx", "eval_seed",
        "pvol", "prod_ratio", "mean_queue", "div_score",
        "reward", "steps",
        # Per-scenario ratios against this round's running bests (diagnostic).
        "running_best_pvol", "running_best_queue",
        "pvol_ratio", "queue_ratio",
        # Round-level combined values (repeated per row for join-free plotting).
        "combined_composite", "incumbent_composite_rescored",
        "is_new_best", "evals_without_improvement",
    ]

    def _append_eval_logs(
        self,
        per_scenario_results: Dict[str, Dict[str, Any]],
        combined_composite: float,
        incumbent_composite_rescored: float,
        is_new_best: bool,
    ) -> None:
        """
        One row per (scenario, eval_episode) plus denominator + ratio
        columns, appended to <save_dir>/eval_episodes_raw.csv (header
        written on first call).
        """
        raw_path = os.path.join(self.save_dir, "eval_episodes_raw.csv")
        raw_rows: List[Dict[str, Any]] = []
        for scen, eval_result in per_scenario_results.items():
            runs = eval_result.get('individual_runs', [])
            pvol_best = self._running_best_pvol[scen]
            queue_best = self._running_best_queue[scen]
            for run in runs:
                pvol = float(run['PVOL'])
                queue = max(float(run['MeanQueue']), self.QUEUE_EPS)
                pvol_ratio = (pvol / pvol_best) if pvol_best > 0 else 0.0
                queue_ratio = (queue_best / queue) if np.isfinite(queue_best) else 0.0
                raw_rows.append({
                    "seed": self.seed,
                    "checkpoint_episode": self.episode_count,
                    "scenario": scen,
                    "eval_idx": run['Eval_Run'],
                    "eval_seed": run['Seed'],
                    "pvol": pvol,
                    "prod_ratio": run.get('ProdRatio', ''),
                    "mean_queue": run['MeanQueue'],
                    "div_score": run.get('DivScore', ''),
                    "reward": run.get('Total_Reward', ''),
                    "steps": run.get('Total_Steps', ''),
                    "running_best_pvol": pvol_best,
                    "running_best_queue": queue_best,
                    "pvol_ratio": pvol_ratio,
                    "queue_ratio": queue_ratio,
                    "combined_composite": combined_composite,
                    "incumbent_composite_rescored": incumbent_composite_rescored,
                    "is_new_best": int(bool(is_new_best)),
                    "evals_without_improvement": self.evals_without_improvement,
                })
        if not raw_rows:
            return
        raw_exists = os.path.isfile(raw_path) and os.path.getsize(raw_path) > 0
        with open(raw_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._RAW_FIELDS)
            if not raw_exists:
                w.writeheader()
            w.writerows(raw_rows)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        if any(self.locals.get('dones', [])):
            self.episode_count += 1

        if self.episode_count - self._last_eval_episode < self.eval_every_episodes:
            return True
        if self.episode_count == self._last_eval_episode:
            return True
        self._last_eval_episode = self.episode_count

        scenario_names = list(self.eval_scenarios.keys())
        if self.verbose > 0:
            print(
                f"\n=== Multi-scenario eval at training episode "
                f"{self.episode_count} "
                f"[scenarios={scenario_names}, "
                f"{self.num_eval_episodes} deterministic runs each] ==="
            )

        eval_initial_seed = (
            self.seed if self.seed is not None and self.seed >= 0 else 44
        )

        # 1. Run eval on each scenario
        per_scenario_results: Dict[str, Dict[str, Any]] = {}
        for scen_name, overrides in self.eval_scenarios.items():
            if self.verbose > 0:
                print(f"\n  -- Scenario {scen_name} --")
            per_scenario_results[scen_name] = run_deterministic_eval(
                model=self.model,
                algo=self.algo,
                env_id=self.eval_env_id,
                num_eval_episodes=self.num_eval_episodes,
                scenario_overrides=overrides,
                eval_initial_seed=eval_initial_seed,
                scenario_name=scen_name,
            )

        # 2. Aggregate per-scenario raw (pvol, queue)
        current_raw: Dict[str, Dict[str, float]] = {
            scen: {
                'pvol': float(res['mean_pvol']),
                'queue': float(res['mean_queue']),
            }
            for scen, res in per_scenario_results.items()
        }

        # 3. Update running bests BEFORE scoring, so both candidate and
        # incumbent are scored against the same fresh denominators.
        self._update_running_bests(current_raw)

        # 4. Score candidate and (re-)score incumbent
        candidate_composite = self._composite_from_raw(current_raw)
        if self._incumbent_raw is not None:
            incumbent_composite_rescored = self._composite_from_raw(
                self._incumbent_raw
            )
        else:
            incumbent_composite_rescored = -np.inf

        is_new_best = self._is_new_best(
            candidate_composite, incumbent_composite_rescored,
        )

        # 5. Best-model save / patience state
        if is_new_best:
            self._incumbent_raw = current_raw
            self.best_composite = candidate_composite
            self.best_checkpoint_episode = self.episode_count
            self.evals_without_improvement = 0
            best_path = os.path.join(self.save_dir, self._best_model_filename)
            try:
                self.model.save(best_path)
                vec_env = self.model.get_vec_normalize_env()
                if vec_env is not None:
                    vec_env.save(
                        os.path.join(self.save_dir, self._best_vecnorm_filename)
                    )
                if self.verbose > 0:
                    print(
                        f"  -> new best composite ({candidate_composite:.4f}), "
                        f"checkpoint saved to {best_path}"
                    )
            except Exception as e:
                print(f"  -> WARNING: could not save best-so-far checkpoint: {e}")
        else:
            self.evals_without_improvement += 1
            # Keep best_composite synced to the re-scored incumbent, not
            # a stale scalar from an earlier round.
            if np.isfinite(incumbent_composite_rescored):
                self.best_composite = incumbent_composite_rescored

        # 6. Per-round history record
        record: Dict[str, Any] = {
            'training_episode': self.episode_count,
            'combined_composite': candidate_composite,
            'incumbent_composite_rescored': incumbent_composite_rescored,
            'is_new_best': is_new_best,
            'best_composite_so_far': self.best_composite,
            'best_checkpoint_episode': self.best_checkpoint_episode,
            'evals_without_improvement': self.evals_without_improvement,
        }
        for scen, res in per_scenario_results.items():
            record[f'{scen}_mean_pvol'] = res['mean_pvol']
            record[f'{scen}_mean_queue'] = res['mean_queue']
            record[f'{scen}_mean_prod_ratio'] = res['mean_prod_ratio']
            record[f'{scen}_mean_div_score'] = res['mean_div_score']
            record[f'{scen}_running_best_pvol'] = self._running_best_pvol[scen]
            record[f'{scen}_running_best_queue'] = self._running_best_queue[scen]
        self.eval_history.append(record)

        # 7. Console summary
        if self.verbose > 0:
            per_scen_str = "  ".join(
                f"[{scen}] pvol={res['mean_pvol']:.1f}"
                f" (best={self._running_best_pvol[scen]:.1f})"
                f" queue={res['mean_queue']:.2f}"
                f" (best={self._running_best_queue[scen]:.2f})"
                for scen, res in per_scenario_results.items()
            )
            print(
                f"\n  composite={candidate_composite:.4f}  "
                f"incumbent(rescored)={incumbent_composite_rescored:.4f}  "
                f"best_so_far={self.best_composite:.4f}"
                f"@ep{self.best_checkpoint_episode}  "
                f"no_improve_evals={self.evals_without_improvement}/{self.patience}\n"
                f"  {per_scen_str}"
            )

        # 8. TensorBoard: round-level scalars under eval_combined/.
        self.logger.record("eval_combined/composite", candidate_composite)
        self.logger.record(
            "eval_combined/incumbent_composite_rescored",
            incumbent_composite_rescored,
        )
        self.logger.record(
            "eval_combined/best_composite_so_far", self.best_composite,
        )
        self.logger.record(
            "eval_combined/evals_without_improvement",
            self.evals_without_improvement,
        )
        self.logger.record("eval_combined/is_new_best", float(is_new_best))
        # Per-scenario diagnostics under eval_scenario{X}/.
        for scen, res in per_scenario_results.items():
            prefix = f"eval_scenario{scen}"
            self.logger.record(f"{prefix}/mean_pvol", res['mean_pvol'])
            self.logger.record(f"{prefix}/mean_queue", res['mean_queue'])
            self.logger.record(f"{prefix}/mean_prod_ratio", res['mean_prod_ratio'])
            self.logger.record(f"{prefix}/mean_div_score", res['mean_div_score'])
            self.logger.record(
                f"{prefix}/running_best_pvol", self._running_best_pvol[scen],
            )
            self.logger.record(
                f"{prefix}/running_best_queue", self._running_best_queue[scen],
            )
            pvol_ratio = (
                res['mean_pvol'] / self._running_best_pvol[scen]
                if self._running_best_pvol[scen] > 0 else 0.0
            )
            queue_ratio = (
                self._running_best_queue[scen]
                / max(res['mean_queue'], self.QUEUE_EPS)
                if np.isfinite(self._running_best_queue[scen]) else 0.0
            )
            self.logger.record(f"{prefix}/pvol_ratio", pvol_ratio)
            self.logger.record(f"{prefix}/queue_ratio", queue_ratio)
        self.logger.dump(step=self.num_timesteps)

        # 9. CSV
        try:
            self._append_eval_logs(
                per_scenario_results,
                candidate_composite,
                incumbent_composite_rescored,
                is_new_best,
            )
        except Exception as e:
            print(f"  -> WARNING: could not write eval logs: {e}")

        # 10. Early stopping
        if (
            self.early_stop_enabled
            and self.evals_without_improvement >= self.patience
        ):
            print(
                f"\nEarly stopping: no composite improvement "
                f">{self.improvement_margin*100:.1f}% in "
                f"{self.patience} consecutive multi-scenario evals "
                f"({self.patience * self.eval_every_episodes} episodes). "
                f"Best composite={self.best_composite:.4f} @ episode "
                f"{self.best_checkpoint_episode} "
                f"(checkpoint: {self._best_model_filename})."
            )
            return False

        return True


# Backward-compat alias for the old class name.
ScenarioEvalEarlyStop = MultiScenarioEvalEarlyStop


def log_evaluation_csv(episode_count: int, eval_data: List[Dict[str, Any]]):
    EVAL_DIR = 'interim_test_data'
    os.makedirs(EVAL_DIR, exist_ok=True)
    for row in eval_data:
        row['Training_Episode'] = episode_count
    df = pd.DataFrame(eval_data)
    summary_row = {
        'Training_Episode': episode_count,
        'Eval_Run': 'Mean',
        'Seed': '-',
        'Total_Reward': df['Total_Reward'].mean(),
        'PVOL': df['PVOL'].iloc[0],
        'DivScore': df['DivScore'].mean(),
        'Total_Steps': df['Total_Steps'].mean()
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    filename = os.path.join(EVAL_DIR, 'Evaluation_Results.csv')
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
        print(f"Evaluation data appended to {filename}")
    else:
        df.to_csv(filename, mode='w', header=True, index=False)
        print(f"Evaluation data saved to new file: {filename}")


def run_deterministic_eval(model: Any, env_id: str, num_eval_episodes: int,
                           scenario_overrides: Optional[dict] = None,
                           eval_initial_seed: int = 44,
                           algo: str = "maskable_ppo",
                           scenario_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Run num_eval_episodes deterministic episodes and return aggregated
    results. Env wrapping is delegated to wrap_env_for_algo so the rules
    can't drift from the training-time env. scenario_name is a cosmetic
    tag surfaced in print lines and on the returned dict; optional.
    """
    results = []
    if num_eval_episodes <= 0:
        return {'individual_runs': [], 'mean_pvol': 0.0, 'mean_div_score': 0.0,
                'mean_prod_ratio': 0.0, 'mean_queue': 0.0,
                'scenario_name': scenario_name}

    spec = get_spec(algo)
    scen_tag = f" [scenario={scenario_name}]" if scenario_name else ""

    for eval_num in range(num_eval_episodes):
        print(f"\n--- Starting Deterministic Evaluation Episode "
              f"{eval_num + 1}/{num_eval_episodes}{scen_tag} ---")

        eval_seed = gen_seed(eval_num, initial_seed=eval_initial_seed)
        eval_csv_path = "./eval_shared.csv"

        eval_env = gym.make(env_id, render_mode="console",
                            scenario_overrides=scenario_overrides,
                            csv_path=eval_csv_path)
        eval_env = wrap_env_for_algo(eval_env, spec)

        obs, info = eval_env.reset(seed=eval_seed)
        done = False
        cumulative_reward = 0
        final_pvol = 0.0
        final_div_score = 0.0
        final_prod_ratio = 0.0
        final_mean_queue = 0.0
        step_count = 0

        while not done:
            action, _ = _predict_action(model, obs, eval_env, deterministic=True)
            action_scalar = action.item()
            obs, reward, done, truncated, info = eval_env.step(action_scalar)
            cumulative_reward += reward
            step_count += 1
            if done:
                final_pvol = info.get('PVOL', 0.0)
                final_div_score = info.get('DivScore', 0.0)
                final_prod_ratio = info.get('prod_ratio', 0.0)
                final_mean_queue = info.get('mean_queue', 0.0)
                break

        results.append({
            'Eval_Run': eval_num + 1,
            'Seed': eval_seed,
            'Total_Reward': float(cumulative_reward),
            'PVOL': final_pvol,
            'DivScore': final_div_score,
            'ProdRatio': final_prod_ratio,
            'MeanQueue': final_mean_queue,
            'Total_Steps': step_count
        })

        print(f"--- Evaluation Run {eval_num + 1} Finished. PVOL: {final_pvol:.2f}, "
              f"DivScore: {final_div_score:.4f}, ProdRatio: {final_prod_ratio:.4f}, "
              f"MeanQueue: {final_mean_queue:.2f} ---")
        eval_env.close()

    mean_pvol = np.mean([r['PVOL'] for r in results])
    mean_div_score = np.mean([r['DivScore'] for r in results])
    mean_prod_ratio = np.mean([r['ProdRatio'] for r in results])
    mean_queue = np.mean([r['MeanQueue'] for r in results])

    return {'individual_runs': results, 'mean_pvol': mean_pvol, 'mean_div_score': mean_div_score,
            'mean_prod_ratio': mean_prod_ratio, 'mean_queue': mean_queue,
            'scenario_name': scenario_name}


class TrainingLoggerCallback(BaseCallback):
    """Custom callback for saving model checkpoints and managing episode limits."""
    def __init__(
        self,
        model,
        save_dir,
        eval_interval: int,
        verbose=1,
        max_timesteps=10000000,
        max_episodes=2,
        save_interval=2,
        eval_env_id: str = 'Minegym-v0',
        scenario_overrides: Optional[dict] = None,
        num_eval_episodes: int = 0,
        training_env=None,
        algo: str = "maskable_ppo",  # stamps the algo into checkpoint names
    ):
        super().__init__(verbose)
        self.model = model
        self.save_dir = save_dir
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.save_interval = save_interval
        self.episode_count = 0
        self.eval_interval = eval_interval
        self.eval_env_id = eval_env_id
        self.scenario_overrides = scenario_overrides
        self.num_eval_episodes = num_eval_episodes
        self.last_eval_episode = 0
        # "ppo_minegym_checkpoint_*" preserved for maskable_ppo so paper
        # artifacts and resume-by-name tooling are unaffected.
        if algo.lower() == "maskable_ppo":
            self._ckpt_prefix = "ppo_minegym_checkpoint"
        else:
            self._ckpt_prefix = f"{algo}_checkpoint"

    def _on_step(self) -> bool:
        if any(self.locals['dones']):
            self.episode_count += 1
            if self.verbose > 0:
                print(f"--- Training Episode {self.episode_count} finished at Timestep {self.num_timesteps} ---")
            if self.episode_count % self.save_interval == 0:
                model_path = os.path.join(
                    self.save_dir,
                    f"{self._ckpt_prefix}_{self.episode_count}.zip"
                )
                try:
                    self.model.save(model_path)
                    if self.verbose > 0:
                        print(f"Checkpoint saved at episode {self.episode_count} to {model_path}")
                except Exception as e:
                    print(f"Error saving model checkpoint: {e}")

        if self._should_stop_training():
            return False
        return True

    def _should_stop_training(self) -> bool:
        if self.num_timesteps >= self.max_timesteps:
            print(f"Reached maximum timesteps of {self.max_timesteps}")
            return True
        if self.episode_count >= self.max_episodes:
            print(f"Reached maximum episode count of {self.max_episodes}")
            return True
        return False

    def close(self):
        pass


@dataclass
class SeedPaths:
    seed_dir: str
    checkpoints_dir: str
    best_dir: str
    final_dir: str
    logs_dir: str
    tb_dir: str
    train_csv: str
    meta_json: str
    config_snapshot: str
    scenario_snapshot: str


def build_seed_paths(algo: str, parent_dir: str, seed: int) -> SeedPaths:
    seed_dir = os.path.join(parent_dir, f"seed_{seed}")
    paths = SeedPaths(
        seed_dir=seed_dir,
        checkpoints_dir=os.path.join(seed_dir, "checkpoints"),
        best_dir=os.path.join(seed_dir, "best"),
        final_dir=os.path.join(seed_dir, "final"),
        logs_dir=os.path.join(seed_dir, "logs"),
        tb_dir=os.path.join(seed_dir, "tb"),
        train_csv=os.path.join(seed_dir, "logs", "train_shared.csv"),
        meta_json=os.path.join(seed_dir, "run_meta.json"),
        config_snapshot=os.path.join(seed_dir, "config_snapshot.txt"),
        scenario_snapshot=os.path.join(seed_dir, "T_scene_config_snapshot.txt"),
    )
    for d in (paths.seed_dir, paths.checkpoints_dir, paths.best_dir,
              paths.final_dir, paths.logs_dir, paths.tb_dir):
        os.makedirs(d, exist_ok=True)
    return paths


def snapshot_configs(paths: SeedPaths) -> None:
    try:
        shutil.copy2("config_extend_review.txt", paths.config_snapshot)
    except FileNotFoundError:
        print("WARNING: config_extend_review.txt not found in cwd; not snapshotted.")
    if os.path.isfile("T_scene_config.txt"):
        shutil.copy2("T_scene_config.txt", paths.scenario_snapshot)


def write_run_meta(paths: SeedPaths, meta: dict) -> None:
    try:
        with open(paths.meta_json, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"WARNING: could not write run_meta.json: {e}")


def write_sweep_summary(parent_dir: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    summary_path = os.path.join(parent_dir, "sweep_summary.csv")
    try:
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\nSweep summary written to: {summary_path}")
    except Exception as e:
        print(f"WARNING: could not write sweep summary: {e}")


def aggregate_eval_summary(parent_dir: str) -> None:
    """
    Roll per-episode raw eval CSVs (one per seed) up into two summary CSVs
    at the sweep parent:

      * ``eval_summary_all_seeds.csv``  -- per (seed, checkpoint_episode,
        scenario) aggregate: mean/std of pvol, mean_queue, prod_ratio,
        div_score, reward, and the per-scenario ratios that feed the
        composite.

      * ``eval_summary_combined.csv``  -- per (seed, checkpoint_episode)
        round-level: mean combined_composite, incumbent_composite_rescored,
        and is_new_best; adds an ``is_seed_best`` flag on the argmax
        combined_composite row per seed.

    Split into two files because the raw CSV now interleaves per-scenario
    rows and per-round scalars; keeping them collapsed into a single
    output CSV would either duplicate the combined scalars across
    scenarios or hide the per-scenario detail.
    """
    raw_paths = sorted(glob.glob(
        os.path.join(parent_dir, "seed_*", "best", "eval_episodes_raw.csv"),
    ))
    if not raw_paths:
        return
    try:
        df = pd.concat([pd.read_csv(p) for p in raw_paths], ignore_index=True)
    except Exception as e:
        print(f"WARNING: could not read raw eval CSVs for aggregation: {e}")
        return

    # Per-scenario aggregate
    per_scen = df.groupby(
        ["seed", "checkpoint_episode", "scenario"], sort=True,
    ).agg(
        num_eval_episodes=("eval_idx", "count"),
        mean_pvol=("pvol", "mean"),               std_pvol=("pvol", "std"),
        mean_prod_ratio=("prod_ratio", "mean"),   std_prod_ratio=("prod_ratio", "std"),
        mean_queue=("mean_queue", "mean"),        std_queue=("mean_queue", "std"),
        mean_div_score=("div_score", "mean"),     std_div_score=("div_score", "std"),
        mean_reward=("reward", "mean"),           std_reward=("reward", "std"),
        pvol_ratio=("pvol_ratio", "first"),
        queue_ratio=("queue_ratio", "first"),
        running_best_pvol=("running_best_pvol", "first"),
        running_best_queue=("running_best_queue", "first"),
    ).reset_index()

    per_scen_path = os.path.join(parent_dir, "eval_summary_all_seeds.csv")
    try:
        per_scen.to_csv(per_scen_path, index=False)
        print(
            f"Cross-seed per-scenario eval summary updated: {per_scen_path} "
            f"({len(per_scen)} rows, {len(raw_paths)} seed file(s))"
        )
    except Exception as e:
        print(f"WARNING: could not write per-scenario eval summary: {e}")

    # Round-level (combined) aggregate
    combined = df.groupby(["seed", "checkpoint_episode"], sort=True).agg(
        combined_composite=("combined_composite", "first"),
        incumbent_composite_rescored=("incumbent_composite_rescored", "first"),
        is_new_best=("is_new_best", "first"),
        evals_without_improvement=("evals_without_improvement", "first"),
    ).reset_index()
    combined["is_seed_best"] = 0
    if not combined.empty:
        idx_best_per_seed = combined.groupby("seed")["combined_composite"].idxmax()
        combined.loc[idx_best_per_seed, "is_seed_best"] = 1
    combined_path = os.path.join(parent_dir, "eval_summary_combined.csv")
    try:
        combined.to_csv(combined_path, index=False)
        print(
            f"Cross-seed combined eval summary updated: {combined_path} "
            f"({len(combined)} rows)"
        )
    except Exception as e:
        print(f"WARNING: could not write combined eval summary: {e}")


def train_one_seed(
    seed: int,
    algo: str,
    parent_dir: str,
    num_episodes: int,
    scenario: Optional[str],
    scenario_overrides: dict,
    n_steps_calculated: int,
    model_path: Optional[str] = None,
    no_early_stop: bool = False,
    improvement_margin: float = 0.01,
    patience: int = 10,
    eval_every_episodes: int = 500,
    num_eval_episodes: int = 3,
) -> Dict[str, Any]:
    """Run a single training to completion for one seed."""
    random.seed(seed)
    np.random.seed(seed)

    # Resolved once, before any directory creation, so an unknown algo
    # errors out immediately.
    spec = get_spec(algo)

    paths = build_seed_paths(algo, parent_dir, seed)
    snapshot_configs(paths)

    meta = {
        "seed": seed,
        "algo": algo,
        "algo_class": spec.cls.__name__,
        "algo_policy": spec.policy,
        "algo_supports_masking": spec.supports_masking,
        "algo_needs_flat_obs": spec.needs_flat_obs,
        "algo_default_kwargs": dict(spec.default_kwargs),
        "scenario_cli": scenario,
        "train_randomized_scenarios": True,
        "train_scenario_pool": list(TRAIN_SCENARIO_POOL_FOR_META),
        "eval_scenarios": ["A", "F"],
        "num_episodes_requested": num_episodes,
        "resumed_from": model_path,
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "early_stop": {
            "enabled": not no_early_stop,
            "improvement_margin": improvement_margin,
            "patience": patience,
            "eval_every_episodes": eval_every_episodes,
            "num_eval_episodes": num_eval_episodes,
        },
    }
    write_run_meta(paths, meta)

    if os.path.exists("alloc.json"):
        try:
            os.remove("alloc.json")
        except OSError:
            pass

    # Wrapping is algo-driven via wrap_env_for_algo(env, spec); reused by
    # run_deterministic_eval so train/eval envs have matching wrappers.
    env = gym.make("Minegym-v0", render_mode="console",
                   scenario_overrides=None,
                   csv_path=paths.train_csv,
                   randomize_scenarios=True)
    env = wrap_env_for_algo(env, spec)

    # norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.995 are
    # the published paper settings, applied to every algo for fairness.
    VECNORM_PATH = os.path.join(paths.final_dir, "vecnormalize.pkl")
    env = DummyVecEnv([lambda: env])
    if model_path and os.path.isfile(VECNORM_PATH):
        print(f"Loading VecNormalize stats from {VECNORM_PATH}")
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True,
                           clip_reward=10.0, gamma=0.995)

    # ---- Model construction via the registry ------------------------------
    if model_path:
        print(f"Loading {algo} model from checkpoint: {model_path}")
        model = make_model(algo, env, seed=seed, tb_log=paths.tb_dir,
                           n_steps_calculated=n_steps_calculated,
                           model_path=model_path)
        match = re.search(r'_(\d+)\.zip$', model_path)
        start_episode = int(match.group(1)) if match else 0
        print(f"Resuming training from episode: {start_episode}")
    else:
        print(f"Initializing a new {spec.cls.__name__} model (algo={algo}, seed={seed}).")
        model = make_model(algo, env, seed=seed, tb_log=paths.tb_dir,
                           n_steps_calculated=n_steps_calculated)
        start_episode = 0

    raw_env = env
    logger_callback = TrainingLoggerCallback(
        model=model,
        save_dir=paths.checkpoints_dir,
        verbose=1,
        max_episodes=num_episodes + start_episode,
        save_interval=5000,
        eval_interval=DEFAULT_EVAL_INTERVAL,
        num_eval_episodes=DEFAULT_EVAL_EPISODES,
        eval_env_id='Minegym-v0',
        scenario_overrides=scenario_overrides,
        training_env=raw_env,
        algo=algo,
    )
    logger_callback.episode_count = start_episode

    # Baseline (A, no failures) + critical bottleneck (F, 37.5% shovel
    # loss); composite is averaged over the pair. See
    # MultiScenarioEvalEarlyStop docstring.
    EVAL_SCENARIO_NAMES = ("A", "F")
    eval_scenarios: Dict[str, dict] = {}
    for scen_name in EVAL_SCENARIO_NAMES:
        overrides = load_scenario(scen_name)
        if not overrides:
            raise RuntimeError(
                f"load_scenario('{scen_name}') returned no overrides -- "
                "T_scene_config.txt is likely missing from the working directory, "
                f"or [SCENARIO_{scen_name}] was not found. Refusing to proceed."
            )
        eval_scenarios[scen_name] = overrides

    multi_eval_callback = MultiScenarioEvalEarlyStop(
        model=model,
        eval_env_id='Minegym-v0',
        eval_scenarios=eval_scenarios,
        save_dir=paths.best_dir,
        algo=algo,
        eval_every_episodes=eval_every_episodes,
        num_eval_episodes=num_eval_episodes,
        improvement_margin=improvement_margin,
        patience=patience,
        early_stop_enabled=not no_early_stop,
        seed=seed,
        verbose=1,
    )
    multi_eval_callback.episode_count = start_episode

    entropy_schedule_cb = EntropyScheduleCallback(
        start=0.02,
        end=0.007,
        total_episodes=max(1, num_episodes),
        episode_counter_cb=logger_callback,
        verbose=1,
    )
    combined_callback = CallbackList([multi_eval_callback, logger_callback, entropy_schedule_cb])

    final_status = "completed"
    error_msg = None
    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=combined_callback,
            tb_log_name=f"{algo}_seed{seed}",
            log_interval=1,
        )
        # Historical final-model name preserved for the paper baseline.
        final_name = "ppo_minegym_final.zip" if algo.lower() == "maskable_ppo" else f"{algo}_final.zip"
        final_model_path = os.path.join(paths.final_dir, final_name)
        model.save(final_model_path)
        try:
            vec_env = model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(VECNORM_PATH)
        except Exception as e:
            print(f"WARNING: could not save VecNormalize stats: {e}")
    except KeyboardInterrupt:
        final_status = "interrupted"
        raise
    except Exception as e:
        final_status = "failed"
        error_msg = repr(e)
        print(f"ERROR in seed {seed}: {error_msg}")
    finally:
        logger_callback.close()
        env.close()

    summary = {
        "seed": seed,
        "algo": algo,
        "scenario": scenario or "",
        "status": final_status,
        "final_episode": getattr(logger_callback, "episode_count", start_episode),
        "best_composite": getattr(multi_eval_callback, "best_composite", None),
        "best_checkpoint_episode": getattr(multi_eval_callback, "best_checkpoint_episode", None),
        "num_evals_done": len(getattr(multi_eval_callback, "eval_history", [])),
        "error": error_msg or "",
    }
    meta.update({
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "status": final_status,
        "error": error_msg,
        "final_episode": summary["final_episode"],
        "best_composite": summary["best_composite"],
        "best_checkpoint_episode": summary["best_checkpoint_episode"],
    })
    write_run_meta(paths, meta)
    return summary


def main(choice, num_episodes, model_path=None, scenario=None, *,
         seed=None, seed_start=None, repeat=1, algo="maskable_ppo",
         no_early_stop=False, improvement_margin=0.01, patience=10,
         eval_every_episodes=2000, num_eval_episodes=3):

    # Resolve the algo before any directory creation or env work.
    spec = get_spec(algo)

    scenario_overrides = load_scenario(scenario)
    register_minegym()

    cfg_samp = ConfigSampler('config_extend_review.txt')
    shift_duration = cfg_samp.get_sampled_value('Sdur')
    decisions_per_episode = 210
    n_steps_calculated = decisions_per_episode * 2
    print(f"Calculated n_steps: {n_steps_calculated} (~2 episodes of {decisions_per_episode} RL decisions each)")

    if choice == 'train':
        if model_path:
            ckpt_dir = os.path.dirname(os.path.abspath(model_path))
            seed_dir_guess = os.path.dirname(ckpt_dir)
            parent_dir = os.path.dirname(seed_dir_guess)
            print(f"Resuming under existing parent dir: {parent_dir}")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            parent_dir = os.path.join(f"{algo.upper()}_Models", f"run_{timestamp}")
            os.makedirs(parent_dir, exist_ok=True)
            print(f"Sweep parent dir: {parent_dir}  (repeat={repeat}, "
                  f"seed={seed}, seed_start={seed_start})")

        # In train mode, produces `repeat` seeds (one per from-scratch run).
        seeds, anchor, seed_mode = resolve_episode_seeds(
            seed=seed, seed_start=seed_start, num_episodes=repeat,
        )
        print(f"Seed mode: {seed_mode}  (anchor={anchor})")
        print(f"Seed sequence: {seeds}")

        sweep_rows: List[Dict[str, Any]] = []
        for idx, seed in enumerate(seeds):
            print(f"\n{'='*70}\n[{idx+1}/{repeat}] Training algo={algo} seed={seed}\n{'='*70}")
            try:
                row = train_one_seed(
                    seed=seed,
                    algo=algo,
                    parent_dir=parent_dir,
                    num_episodes=num_episodes,
                    scenario=scenario,
                    scenario_overrides=scenario_overrides,
                    n_steps_calculated=n_steps_calculated,
                    model_path=model_path if idx == 0 else None,
                    no_early_stop=no_early_stop,
                    improvement_margin=improvement_margin,
                    patience=patience,
                    eval_every_episodes=eval_every_episodes,
                    num_eval_episodes=num_eval_episodes,
                )
            except KeyboardInterrupt:
                print(f"\nInterrupted during seed {seed}. Writing partial sweep summary and exiting.")
                write_sweep_summary(parent_dir, sweep_rows)
                aggregate_eval_summary(parent_dir)
                raise
            except Exception as e:
                print(f"FATAL setup error for seed {seed}: {e!r}")
                sweep_rows.append({
                    "seed": seed, "algo": algo, "scenario": scenario or "",
                    "status": "setup_failed", "final_episode": 0,
                    "best_composite": None, "best_checkpoint_episode": None,
                    "num_evals_done": 0, "error": repr(e),
                })
                continue
            sweep_rows.append(row)
            write_sweep_summary(parent_dir, sweep_rows)
            aggregate_eval_summary(parent_dir)

        print(f"\nSweep complete: {len(sweep_rows)} seed(s) processed under {parent_dir}")
        return

    elif choice == 'play':
        # Play-mode artifacts route through results_paths (consistent
        # {algo}_scen{X}_seed{Y} tag under results/RL_baselines/), mirroring
        # mGym_DefSchdRun.py's results/non-RL_baselines/ convention.
        from results_paths import (
            rl_algo_tag, kpi_log_path, channel_csv_path,
        )

        play_episode_seeds, anchor, seed_mode = resolve_episode_seeds(
            seed=seed, seed_start=seed_start, num_episodes=num_episodes,
        )

        algo = rl_algo_tag(model_path)  # e.g. "rl_ppo" from a PPO checkpoint
        kpi_path = kpi_log_path("rl", algo, scenario, anchor)
        channel_path = channel_csv_path("rl", algo, scenario, anchor)

        # Fresh KPI log per play invocation.
        if kpi_path.exists():
            kpi_path.unlink()

        TESTING_CSV_PATH = str(channel_path)

        env_test = gym.make("Minegym-v0", render_mode="console",
                            scenario_overrides=scenario_overrides,
                            csv_path=TESTING_CSV_PATH,
                            scenario_name=scenario,
                            play_seed=anchor,
                            kpi_log_path=str(kpi_path))
        # Same wrapping rules as train/eval, driven by the spec.
        env_test = wrap_env_for_algo(env_test, spec)

        all_results = []
        try:
            if model_path is None:
                print("Error: Model path must be provided for playing.")
                return

            # spec.cls.load works for every registered algorithm (all
            # expose .load(path, env=...) with identical signatures).
            model = spec.cls.load(model_path, env=env_test)

            print(f"\n--- Starting Play Mode for {num_episodes} Episodes ---")
            print(f"    algo tag: {algo} | scenario: {scenario} | "
                  f"seed mode: {seed_mode} (anchor={anchor})")
            print(f"    KPI log:  {kpi_path}")

            for episode in range(num_episodes):
                episode_seed = play_episode_seeds[episode]
                obs, info = env_test.reset(seed=episode_seed)
                done = False
                cumulative_reward = 0
                step_count = 0

                while not done:
                    action, _states = _predict_action(
                        model, obs, env_test, deterministic=True)
                    action_scalar = action.item()
                    obs, reward, done, truncated, info = env_test.step(action_scalar)
                    cumulative_reward += reward
                    step_count += 1

                    if done or truncated:
                        final_pvol = info.get('PVOL', 0.0)
                        final_div_score = info.get('DivScore', 0.0)
                        all_results.append({
                            'Episode': episode + 1,
                            'Reward': cumulative_reward,
                            'PVOL': final_pvol,
                            'DivScore': final_div_score,
                            'Steps': step_count
                        })
                        print(f"Episode {episode + 1} finished. Reward: {cumulative_reward:.2f} | PVOL: {final_pvol:.2f}")
                        break

            if all_results:
                df = pd.DataFrame(all_results)
                mean_reward = df['Reward'].mean()
                mean_pvol = df['PVOL'].mean()
                mean_div_score = df['DivScore'].mean()
                mean_steps = df['Steps'].mean()

                print("\n======================================")
                print(f"    Average Play Metrics ({num_episodes} Runs)   ")
                print("======================================")
                print(f"Mean Reward:        {mean_reward:.2f}")
                print(f"Mean PVOL:          {mean_pvol:.2f}")
                print(f"Mean DivScore:      {mean_div_score:.4f}")
                print(f"Mean Steps:         {mean_steps:.2f}")
                print("======================================")
        finally:
            env_test.close()


if __name__ == "__main__":
    # allow_abbrev=False: a typo or stale alias fails loudly instead of
    # silently binding to a unique prefix (previously `--seed 0` could
    # silently bind to `--seed_start` in play mode).
    parser = argparse.ArgumentParser(
        description="Train or play an RL model for the Minegym environment.",
        allow_abbrev=False,
    )
    parser.add_argument(
        'choice',
        type=str,
        choices=['train', 'play'],
        help="Choose 'train' to train a new model or 'play' to load and play an existing model."
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
        help="Number of episodes to train/play. Default 5000."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help="Path to the pre-trained model for playing. Required for 'play'."
    )
    parser.add_argument(
        '--scenario',
        type=str,
        default=None,
        help="Test scenario (A, B, C, D, E, F)")

    # --seed / --seed_start mean the same thing across train, play, and
    # the classical driver -- see seed_utils.resolve_episode_seeds.
    # play:  --seed N -> every episode reset(seed=N); --seed_start N ->
    #        episode i uses gen_seed(i, N).
    # train: --seed N -> every repeat gets seed N; --seed_start N ->
    #        repeat i gets gen_seed(i, N). Default: --seed_start 0.
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        '--seed', type=int, default=None,
        help="Fixed seed for every episode/repeat (deterministic replay). "
             "Mutually exclusive with --seed_start."
    )
    seed_group.add_argument(
        '--seed_start', type=int, default=None,
        help="LCG anchor: episode/repeat i uses gen_seed(i, seed_start). "
             "Mutually exclusive with --seed."
    )
    seed_group.add_argument(
        '--play_seed', type=int, default=None,
        # Hidden legacy alias, kept so old scripts/cron jobs don't break;
        # emits a deprecation notice after parse.
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--no_early_stop',
        action='store_true',
        default=False,
        help="Disable patience-based early stopping; run for the full --num_episodes."
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=1,
        help="Number of from-scratch training runs (one per seed). Default 1."
    )
    # --algo choices are derived from the registry; adding a new entry
    # there automatically makes it selectable from the CLI.
    parser.add_argument(
        '--algo',
        type=str,
        default='maskable_ppo',
        choices=sorted(ALGO_REGISTRY.keys()),
        help="RL algorithm. Default 'maskable_ppo' reproduces the paper baseline. "
             "Other choices: " + ", ".join(sorted(ALGO_REGISTRY.keys())) + "."
    )
    parser.add_argument(
        '--improvement_margin', type=float, default=0.01,
        help="Relative composite-score improvement required for new-best."
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help="Consecutive evals without improvement before early stopping."
    )
    parser.add_argument(
        '--eval_every_episodes', type=int, default=2000,
        help="Run a deterministic eval batch every N training episodes."
    )
    parser.add_argument(
        '--num_eval_episodes', type=int, default=3,
        help="Number of deterministic eval episodes per checkpoint PER "
             "SCENARIO. With the current multi-scenario eval (A + F), a "
             "value of 3 means 6 eval rollouts per eval round."
    )

    args = parser.parse_args()

    # Legacy alias: warn and remap.
    if args.play_seed is not None:
        print("WARNING: --play_seed is deprecated; use --seed instead. "
              f"Treating --play_seed={args.play_seed} as --seed={args.play_seed}.")
        args.seed = args.play_seed
    # Per-mode default when neither was given (matches historical defaults):
    # play -> --seed 49; train -> --seed_start 0.
    if args.seed is None and args.seed_start is None:
        if args.choice == 'play':
            args.seed = 49
        else:
            args.seed_start = 0

    if args.model_path and args.choice == 'train' and args.repeat != 1:
        parser.error("--model_path (resume) is only supported with --repeat 1.")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1.")

    main(args.choice, args.num_episodes, args.model_path, args.scenario,
         seed=args.seed, seed_start=args.seed_start, repeat=args.repeat,
         algo=args.algo, no_early_stop=args.no_early_stop,
         improvement_margin=args.improvement_margin, patience=args.patience,
         eval_every_episodes=args.eval_every_episodes,
         num_eval_episodes=args.num_eval_episodes)