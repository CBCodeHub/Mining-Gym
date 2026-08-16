"""
Gymnasium-compatible environment wrapping the DES mining-site simulator
"""

import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces import Discrete, Dict, MultiBinary, Box
import time
import os
import json
import csv
import random
import multiprocessing
import threading
import traceback
import sys
from read_config import ConfigSampler
from scenario_loader import load_scenario

# In-process channel replacing the old shared-CSV IPC between this Gym
# environment (main thread) and the DES simulator (worker thread).
from shared_channel import StepChannel, ChannelStopped

# Scenarios sampled from during training randomization. The eval path
# (mGym_GymRun.run_deterministic_eval / ScenarioEvalEarlyStop) hard-codes
# its own fixed scenario and never reads this list.
TRAIN_SCENARIO_POOL = ("A", "B", "C", "D", "E", "F")

from gymnasium.envs.registration import register

# Verbosity gate for per-episode prints; MGYM_VERBOSE=1 restores them.
_MGYM_VERBOSE = int(os.environ.get('MGYM_VERBOSE', '0'))



class Minegym(Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self, render_mode="console", scenario_overrides=None, csv_path=None,
                 scenario_name=None, play_seed=None, randomize_scenarios=False,
                 scenario_pool=None, kpi_log_path=None):
        super(Minegym, self).__init__()

        self.render_mode = render_mode
        self.scenario_overrides = scenario_overrides
        self.scenario_name = scenario_name
        self.play_seed = play_seed
        # Explicit KPI-log path (results_paths.py convention); if None,
        # KPICalculator derives a name from csv_path instead.
        self.kpi_log_path = kpi_log_path

        # Per-episode scenario randomization (training only): when True,
        # reset() draws a scenario uniformly from scenario_pool before
        # starting the DES thread. Eval never sets this, so it stays on a
        # single fixed scenario. Uses a dedicated Generator (not the global
        # random module) so the scenario draw doesn't depend on other
        # random calls earlier in the episode.
        self.randomize_scenarios = bool(randomize_scenarios)
        self.scenario_pool = tuple(scenario_pool) if scenario_pool is not None else TRAIN_SCENARIO_POOL
        self._scenario_rng = np.random.default_rng()
        self._scenario_rng_seeded = False

        # Per-episode seed from reset(seed=...), forwarded to the DES
        # worker as `episode_seed` so runDes seeds its global random/
        # np.random state deterministically per episode (not just the
        # scenario choice, but the full episode dynamics).
        self._current_episode_seed = None

        if csv_path is None:
            raise ValueError("csv_path must be explicitly provided when creating Minegym environment")

        # Kept only as a stable label, still forwarded for KPI-file naming.
        self.file_path = csv_path

        cfg_samplr = ConfigSampler('config_extend_review.txt')
        self.NumTrucks = int(cfg_samplr.get_sampled_value('TR'))
        self.NumShovels = int(cfg_samplr.get_sampled_value('SH'))
        self.id_counter = 0
        self.tender_mode = render_mode

        # Queue-cooldown action-mask thresholds (see action_masks below).
        # Operate on Queue_length's normalized space, not raw truck counts.
        # MULTIPLIER: how far above the best alternative counts as
        # avoidable congestion. MIN_GAP: absolute floor so the rule doesn't
        # fire on near-equal small queues (e.g. shift start).
        self.QUEUE_MASK_MULTIPLIER = 1.5
        self.QUEUE_MASK_MIN_GAP = 0.05

        # Define action and observation spaces
        self.action_space = Discrete(self.NumShovels, start=0)
        self.observation_space = Dict({
            "ShovelID": MultiBinary(self.NumShovels * 4),
            "Queue_length": Box(low=0, high=float('inf'), shape=(self.NumShovels,), dtype=np.float32),
            "SH_Status": MultiBinary(self.NumShovels),

            "TruckID_Active": MultiBinary(1 * 6),  # 1 truck * 6 bits
            "Trips_complete_Active": Box(low=0, high=float('inf'), shape=(1,), dtype=np.float32),
            "TR_Status_Active": MultiBinary(1 * 3),  # 1 truck * 3 bits

            "Fleet_Avg_Trips": Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
            "Recent_Shovel_Usage": Box(low=0, high=1.0, shape=(self.NumShovels,), dtype=np.float32),
            "Fleet_Diversity": Box(low=0, high=1.0, shape=(1,), dtype=np.float32),

        })

        # Single conduit for the (state, action, reward, next-state) signal
        # between this env and the DES; self-checks the action/result
        # alternation and logs any violation.
        self.channel = StepChannel(
            name=os.path.basename(str(self.file_path)),
            log_every=200,
            raise_on_break=False,
            debug_each_step=False,
        )
        self._des_thread = None
        self._episode_counter = 0
        self.step_timeout = 120.0
        self.join_timeout = 30.0

        self.des_process = None  # unused; kept for backward compatibility
        self.done = False

    # ----- in-process DES worker management -----
    def is_des_alive(self):
        """Check whether the DES worker thread is running."""
        return self._des_thread is not None and self._des_thread.is_alive()

    def _start_des_thread(self, fsim, scenario_name=None, play_seed=None,
                          episode_seed=None):
        """
        Start the DES simulator on a daemon background thread, sharing the
        StepChannel directly so each decision point is a microsecond hand-off.
        episode_seed is forwarded to runDes to seed its global RNG state
        deterministically per episode.
        """
        from mGym_DesEnv import runDes

        self._stop_des_thread()

        if _MGYM_VERBOSE:
            print(f"Starting DES worker thread (fsim={fsim}) for episode "
                  f"{self._episode_counter}")
        self._des_thread = threading.Thread(
            target=runDes,
            kwargs={
                'fsim': fsim,
                'flag_RL_sched': True,
                'channel': self.channel,
                'scenario_overrides': self.scenario_overrides,
                'csv_path': self.file_path,  # only used for KPI naming now
                'scenario_name': scenario_name,
                'play_seed': play_seed,
                'episode_seed': episode_seed,
                'episode_idx': self._episode_counter,
                'kpi_log_path': self.kpi_log_path,
            },
            name=f"DES-worker-ep{self._episode_counter}",
            daemon=True,
        )
        self._des_thread.start()

    def _stop_des_thread(self):
        """Cleanly stop the DES worker thread (used between episodes and on
        close). A naturally-finished episode is usually an instant join; if
        still blocked, request_stop() unblocks it via ChannelStopped."""
        if self._des_thread is not None and self._des_thread.is_alive():
            self.channel.request_stop()
            self._des_thread.join(timeout=self.join_timeout)
            if self._des_thread.is_alive():
                print("WARNING: DES worker did not stop within "
                      f"{self.join_timeout}s; it is a daemon and will be "
                      "reaped with the process.")
        self._des_thread = None

    # Backward-compatible aliases (the old API surface, now thread-based).
    def cleanup_resources(self):
        self._stop_des_thread()

    def terminate_DES(self):
        self._stop_des_thread()

    def start_DES(self, fsim, scenario_name=None, play_seed=None,
                  episode_seed=None):
        self._start_des_thread(fsim, scenario_name=scenario_name,
                               play_seed=play_seed,
                               episode_seed=episode_seed)

    def convert_obs_to_numpy(self, observation):
        """Convert observation from JSON to numpy arrays with correct dtypes"""
        observation["ShovelID"] = np.array(observation["ShovelID"], dtype=np.int8)
        observation["Queue_length"] = np.array(observation["Queue_length"], dtype=np.float32)
        observation["SH_Status"] = np.array(observation["SH_Status"], dtype=np.int8)

        # Active truck
        observation["TruckID_Active"] = np.array(observation["TruckID_Active"], dtype=np.int8)
        observation["Trips_complete_Active"] = np.array(observation["Trips_complete_Active"], dtype=np.float32)
        observation["TR_Status_Active"] = np.array(observation["TR_Status_Active"], dtype=np.int8)
    
        # Fleet context features
        observation["Fleet_Avg_Trips"] = np.array(observation["Fleet_Avg_Trips"], dtype=np.float32)
        observation["Recent_Shovel_Usage"] = np.array(observation["Recent_Shovel_Usage"], dtype=np.float32)
        observation["Fleet_Diversity"] = np.array(observation["Fleet_Diversity"], dtype=np.float32)

        return observation



    def action_masks(self):
        """
        Boolean mask over the action space, True where dispatching to that
        shovel is currently legal. SB3-contrib's MaskablePPO calls this
        automatically before sampling an action.

        Two layers, both from the most recent observation:
        1. Operational mask: SH_Status == 1 (broken shovels are illegal;
           otherwise the agent could dispatch to one and just queue).
        2. Queue-cooldown mask: a shovel is masked out if its queue exceeds
           QUEUE_MASK_MULTIPLIER x the best available alternative, and that
           alternative has real slack (gap >= QUEUE_MASK_MIN_GAP) -- makes
           clearly-bad choices illegal rather than merely costly, to curb
           shovel hugging that reward shaping alone didn't eliminate. Falls
           back to "all operational shovels legal" if the rule would mask
           out every option.
        """
        sh_status = self.mine_state["SH_Status"].astype(bool)
        operational = sh_status.copy()

        queue_length = self.mine_state["Queue_length"]  # normalized [0,1]
        op_queues = queue_length[operational]
        if op_queues.size > 1:
            best_q = op_queues.min()
            cooldown = (
                operational
                & (queue_length > self.QUEUE_MASK_MULTIPLIER * max(best_q, 1e-6))
                & (queue_length - best_q >= self.QUEUE_MASK_MIN_GAP)
            )
            candidate_mask = operational & ~cooldown
            if candidate_mask.any():
                return candidate_mask
            return operational

        return operational

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Forwarded to runDes as-is: an explicit seed makes the DES's
        # global RNGs deterministic; None falls back to whatever RNG
        # state already exists (early training resets).
        self._current_episode_seed = seed

        self._stop_des_thread()
        self.channel.prepare_for_episode(episode_index=self._episode_counter)

        # Must happen after super().reset() (needs self.np_random) and
        # before _start_des_thread (which snapshots scenario_overrides).
        # No-op when randomize_scenarios is False (eval path).
        if self.randomize_scenarios:
            if not self._scenario_rng_seeded:
                self._scenario_rng = np.random.default_rng(
                    self.np_random.integers(0, 2**32 - 1))
                self._scenario_rng_seeded = True
            chosen = str(self._scenario_rng.choice(self.scenario_pool))
            overrides = load_scenario(chosen)
            if not overrides:
                # Fail loudly rather than silently falling back to baseline.
                raise RuntimeError(
                    f"randomize_scenarios=True chose '{chosen}' but "
                    f"load_scenario('{chosen}') returned no overrides. "
                    "Check T_scene_config.txt is in the working directory "
                    "and contains the matching [SCENARIO_*] section."
                )
            self.scenario_overrides = overrides
            self.scenario_name = chosen
            if _MGYM_VERBOSE:
                print(f"[reset] randomized scenario for episode "
                      f"{self._episode_counter}: {chosen}")

        self.mine_state = {
            "ShovelID": np.zeros(self.NumShovels * 4, dtype=np.int8),
            "Queue_length": np.zeros(self.NumShovels, dtype=np.float32),
            "SH_Status": np.ones(self.NumShovels, dtype=np.int8),

            "TruckID_Active": np.zeros(1 * 6, dtype=np.int8),
            "Trips_complete_Active": np.zeros(1, dtype=np.float32),
            "TR_Status_Active": np.ones(1 * 3, dtype=np.int8),

            "Fleet_Avg_Trips": np.zeros(1, dtype=np.float32),
            "Recent_Shovel_Usage": np.zeros(self.NumShovels, dtype=np.float32),
            "Fleet_Diversity": np.ones(1, dtype=np.float32),
        }
        self.info = None
        self.terminated = False
        self.done = False

        self.steps_this_episode = 0

        if self.render_mode == "human":
            self._start_des_thread(fsim=True, scenario_name=self.scenario_name,
                                   play_seed=self.play_seed,
                                   episode_seed=self._current_episode_seed)
        elif self.render_mode == "console":
            self._start_des_thread(fsim=False, scenario_name=self.scenario_name,
                                   play_seed=self.play_seed,
                                   episode_seed=self._current_episode_seed)
        else:
            raise ValueError(f"Invalid render_mode: {self.render_mode}. Please choose either 'human' or 'console'.")

        # Must increment after both prepare_for_episode() and
        # _start_des_thread() have captured the same pre-increment value,
        # so the channel's episode index and the DES's episode_idx agree.
        self._episode_counter += 1
        return self.mine_state, {}


    def step(self, action):
        if self.done:
            print("WARNING: step() called on terminated environment. Call reset() first.")
            return self.mine_state, 0.0, True, True, {"error": "step_after_done"}

        self.steps_this_episode += 1

        # DES liveness check: a normal episode end sends terminated=True
        # through the channel and the thread exits naturally, so there's a
        # narrow window where the result is already in the channel slot but
        # the thread has exited. Check for that pending result before
        # declaring a crash, or the real terminal PVOL gets discarded.
        if not self.is_des_alive() and not self.channel.has_pending_result():
            print(f"ERROR: DES worker not alive at step {self.steps_this_episode}")
            self.done = True
            return self.mine_state, 0.0, True, True, {"error": "DES_not_alive"}

        truncated = False
        self.done = False

        # Microsecond hand-off (no file I/O, no sleeps). The channel
        # guarantees this result matches this action, else logs SYNC BREAK.
        try:
            self.channel.send_action(int(action))
            result = self.channel.receive_result(timeout=self.step_timeout)
        except (TimeoutError, ChannelStopped) as e:
            print(f"mGym: no result for step {self.steps_this_episode} ({e}). "
                  "Ending episode.")
            self.done = True
            self._stop_des_thread()
            return self.mine_state, 0.0, True, True, {"error": "des_no_response"}

        observation = self.convert_obs_to_numpy(result["observation"])
        reward = float(result["reward"])
        terminate = bool(result["terminated"])
        info = result.get("info", {}) or {}

        self.mine_state = observation
        self.done = terminate

        if terminate:
            if _MGYM_VERBOSE:
                print("mGym: terminal result received. Episode complete.")
            self._stop_des_thread()   # join the finished worker cleanly

        return observation, reward, self.done, truncated, info


    def render(self):
        # Implementation for rendering (done in DES)
        pass

    def close(self):
        self.terminate_DES()
        print("Environment Closed")