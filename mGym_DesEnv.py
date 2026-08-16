"""
Mining Load and Haul Cycle discrete-event simulation, built on Salabim
"""

import salabim as sim
sim.yieldless(False)
import gymnasium as gym
from gymnasium.spaces import Dict, MultiBinary, Box
from multiprocessing import Process
import random
import csv
import time
import re
import json
import scheduler as sch 
import kpi_calc as kcal
from read_config import ConfigSampler
import threading
import tempfile
import shutil
import queue
import functools
from collections import deque, Counter
import numpy as np
import sys
import os
from datetime import datetime

# In-process channel to the Gym/RL side (this module runs on a worker
# thread). ChannelStopped lets a blocked decision point unwind cleanly
# when the Gym requests an episode reset/shutdown.
from shared_channel import ChannelStopped

# Append-only per-episode metrics log (separate from kpi_calc.py's
# single-snapshot JSON), so training progress can be tracked live.
from episode_metrics_logger import log_episode_metrics, log_dispatch_decision


_RUN_DES_LOCK = threading.Lock()
_thread_locals = threading.local()
_DES_VERBOSE = int(os.environ.get('DES_VERBOSE', '0'))

# Every module global the truck process (or anything it calls during an
# episode) reads. The snapshot/restore helpers below preserve the
# calling thread's view of these across a wait_for_action gap.
_DES_GLOBAL_NAMES = (
    'env', 'shovels', 'truck', 'dumps', 'crushers', 'shovel_animations',
    'shovel_idle_times', 'shovel_last_check',
    '_channel', 'cfg_samp',
    'RL_sched', 'def_schdlr_choice', 'all_trk_shv_dec', 'epsilon',
    'rl_decision_index',
    'Num_trucks', 'Num_shovels', 'Num_crushers', 'Num_dumps',
    'shift_dura', 'targ_pvol', 'load_per_trip', 'choice',
    'total_trips', 'total_crush_trips',
    'broken_shovel_dispatch_count',
    'episode_shovel_choices', 'episode_queue_sum', 'episode_queue_count',
    'truck_trip_counts', 'truck_phases', 'truck_last_trip_times',
    'shovel_wait_time_tracker',
    'trip_times', 'shovel_queues',
    'r_imm_d_pt', 'terminated', 'pvol', 'sim_exit',
    'file_path',
)


def _snapshot_des_globals():
    """Return a dict of the current values of the per-episode module globals."""
    g = globals()
    return {name: g.get(name) for name in _DES_GLOBAL_NAMES}


def _restore_des_globals(snap):
    """Restore module globals from a snapshot dict produced by _snapshot_des_globals."""
    g = globals()
    for name, value in snap.items():
        g[name] = value


def _with_runDes_lock(fn):
    """
    Decorator that acquires _RUN_DES_LOCK for the lifetime of a runDes() call.
    Combined with the release-around-wait_for_action pattern in Truck.process(),
    this lets multiple DES threads coexist in the same Python process without
    racing on the module globals.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _RUN_DES_LOCK.acquire()
        try:
            return fn(*args, **kwargs)
        finally:
            # Balances the acquire above; wait_for_action's own release/
            # reacquire around the blocking wait leaves this safe even if
            # it raised mid-wait (its finally clause reacquires first).
            _RUN_DES_LOCK.release()
    return wrapper


def _vprint(level, *args, **kwargs):
    """
    Print only when DES_VERBOSE is at or above the given level. Used to gate
    the per-decision / per-20-step debug prints that otherwise flood the
    terminal during long runs.
    """
    if _DES_VERBOSE >= level:
        print(*args, **kwargs)


def _dispatch_log_path():
    """
    Build the per-decision dispatch-diagnostics CSV path from the same
    file_path/csv_path used for the episode-metrics log, so train/test/eval
    runs each get their own file. Falls back to a fixed name if file_path
    isn't set yet, since this is diagnostic logging only.
    """
    fp = globals().get('file_path', None)
    return (os.path.splitext(str(fp))[0] + "_dispatch_log.csv") if fp else "dispatch_log.csv"

csv_lock = threading.Lock()

update_freq = 20  # time units between rate-keeping prints
shift_start_time = 0
all_trk_shv_dec = None 
# Monotonic RL-decision counter for the current episode, reset in runDes.
# all_trk_shv_dec (a maxlen deque for the diversity score) saturates at
# its cap, so it isn't usable as a decision index.
rl_decision_index = 0

cfg_samp = ConfigSampler('config_extend_review.txt')
cfg_samp.episode_count = 0

# Fixed parameters loaded from config (cached, same across episodes)
Num_trucks = None 
Num_shovels = None
Num_crushers = None 
Num_dumps = None 
shift_dura = None 
targ_pvol = None 
load_per_trip = None
choice = None 

epsilon = 0.35  # episodic parameter; updated in runDes() per episode

RL_sched = True
def_schdlr_choice = None

# In-process channel to the Gym/RL side, set once per episode at the top
# of runDes(). None for classical/rule-based runs (mGym_DefSchdRun.py),
# in which case every channel hook below is a no-op.
_channel = None

r_optimal = (1-epsilon)/epsilon  # used for calculating waste trip balance

xs_init = 450  # x init value of shovel queue
xd_init = 100  # x init value of dump queue

# Truck state codes as bytes
phase_shovel = '000'
phase_crusher = '001'
phase_dump = '010'
phase_travel_shovel_crusher = '011'
phase_travel_shovel_dump = '100'
phase_travel_crusher_shovel = '101'
phase_travel_dump_shovel = '110'
phase_broken_down = '111'

k  = 5 # Sliding window length for reward calculation
alpha = 0.5 # Decay rate for exponential weight used in sliding window
trip_times = deque(maxlen=k)
shovel_queues = deque(maxlen=k)
r_imm_d_pt = None
terminated = False
pvol = 0


total_trips = 0
avg_idle_orig_time = 0
sim_exit = False
total_crush_trips = 0
broken_shovel_dispatch_count = 0  # sanity counter; should stay 0 once action masking is active

# Per-episode histogram of RL shovel choices {shovel_idx: count}: the
# standing shovel-hugging monitor, reset each episode and summarized
# (max share / selection entropy / unused-shovel count) at shift end.
episode_shovel_choices = {}

# Per-episode running sum/count of queue length at the chosen shovel per
# RL decision, feeding mean_queue into the terminal info dict so an
# external eval callback can read operational queue quality directly.
episode_queue_sum = 0.0
episode_queue_count = 0

truck_trip_counts = {}
truck_phases = {}


def add_item(item):
    global all_trk_shv_dec
    all_trk_shv_dec.append(item)

def diversity_score() -> float:
    """
    Calculate diversity using Shannon entropy (ORIGINAL version).
    NO recency weighting - that made it worse!
    """
    global all_trk_shv_dec, Num_shovels, Num_trucks
    
    window_size = min(30, len(all_trk_shv_dec)) 
    trk_shv_dec = deque(list(all_trk_shv_dec)[-window_size:])
    
    if len(trk_shv_dec) < Num_shovels:
        return 1.0
    
    # Simple Shannon entropy over full window
    counts = Counter(trk_shv_dec)
    total = len(trk_shv_dec)
    entropy = 0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p + 1e-10)
    
    max_entropy = np.log2(Num_shovels)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return normalized_entropy


def calculate_shovel_imbalance(shovels):
    global all_trk_shv_dec
    shovel_counts = Counter(all_trk_shv_dec)

    for shovel in shovels:
        if shovel not in shovel_counts:
            shovel_counts[shovel] = 0

    total_shovels = len(all_trk_shv_dec)
    if total_shovels == 0:
        return 0.0

    max_count = max(shovel_counts.values())
    min_count = min(shovel_counts.values())
    imbalance = max_count - min_count

    zero_usage_count = sum(1 for count in shovel_counts.values() if count == 0)
    starvation_penalty = zero_usage_count * (total_shovels / Num_shovels)
    
    return imbalance + starvation_penalty


def calculate_streak_penalty() -> float:
    """
    Direct penalty for consecutive same-shovel decisions.
    Returns value in [0, 1] where higher = worse streaking.
    """
    global all_trk_shv_dec
    
    if len(all_trk_shv_dec) < 15:
        return 0.0

    recent = list(all_trk_shv_dec)[-15:]
    same_count = sum(1 for i in range(1, len(recent)) if recent[i] == recent[i-1])
    streak_ratio = same_count / (len(recent) - 1)
    
    return streak_ratio

def scheduler_assign(choice, truck_id=None):
    '''
    Reads user choice of scheduler and queries it.
 
    Algorithm registry:
        1. Random
        2. Fixed (round-robin LUT)
        3. Shortest queue 
        4. Shortest Queue First
        5. Minimum Shovel Waiting Time 
    '''
    def_scheduler = sch.DefaultScheduler()
 
    if choice == 1:
        scheduled_equip = def_scheduler.random_sel(shovels)
    elif choice == 2:
        scheduled_equip = def_scheduler.fixed(truck_id, Num_trucks, shovels)
    elif choice == 3:
        scheduled_equip = def_scheduler.shortest_queue_first(shovels)
    elif choice == 4:
        # MSWT reads cumulative per-shovel idle time maintained by the
        # TrackIdleTime salabim component (see class TrackIdleTime in this
        # file). The dict is keyed by shovel.name() and ticks every 5 sim
        # units, which is finer-grained than the truck loading cycle so
        # dispatch decisions always see a recent value.
        scheduled_equip = def_scheduler.min_shovel_waiting_time(
            shovels, shovel_idle_times
        )
    else:
        raise ValueError("Choice must be one of {1, 2, 3, 4}")
 
    return scheduled_equip

def print_episode_summary():
    """Print brief episode summary for debugging"""
    try:
        print(f"Episode {cfg_samp.episode_count}: epsilon={cfg_samp.get_sampled_value('Eps'):.3f}")
        
        # Show if we have shovel classes (indicates heterogeneous setup)
        try:
            classes = cfg_samp.get_sampled_value('shovel_performance_class')
            clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
            print(f"  Shovel classes: {classes}")
            print(f"  Location clusters: {clusters}")
        except KeyError:
            print("  Using homogeneous shovel configuration")
        
        # Infrastructure-based parameters
        try:
            sz1c = cfg_samp.get_sampled_value('SZ1C')
            sz1d = cfg_samp.get_sampled_value('SZ1D')
            sz2c = cfg_samp.get_sampled_value('SZ2C')
            sz2d = cfg_samp.get_sampled_value('SZ2D')
            sz3c = cfg_samp.get_sampled_value('SZ3C')
            sz3d = cfg_samp.get_sampled_value('SZ3D')
            print(f"  Infrastructure (SZ): Zone1-Crush: {sz1c:.2f}, Zone1-Dump: {sz1d:.2f}")
            print(f"    Zone2-Crush: {sz2c:.2f}, Zone2-Dump: {sz2d:.2f}")
            print(f"    Zone3-Crush: {sz3c:.2f}, Zone3-Dump: {sz3d:.2f}")
        except KeyError:
            pass  # not in config

        try:
            trdm = cfg_samp.get_sampled_value('TRDM')
            trcr = cfg_samp.get_sampled_value('TRCR')
            print(f"  Equipment Dump Times: Truck-Dump: {trdm:.2f}, Truck-Crush: {trcr:.2f}")
        except KeyError:
            pass

        try:
            rsh = cfg_samp.get_sampled_value('RSH')
            rtr = cfg_samp.get_sampled_value('RTR')
            rcr = cfg_samp.get_sampled_value('RCR')
            rds = cfg_samp.get_sampled_value('RDS')
            print(f"  Base MTTR: Shovel: {rsh:.2f}, Truck: {rtr:.2f}, Crusher: {rcr:.2f}, Dump: {rds:.2f}")
        except KeyError:
            pass

        try:
            known_cost = cfg_samp.get_sampled_value('known_cost')
            estimated_cost = cfg_samp.get_sampled_value('estimated_cost')
            print(f"  Cost Parameters: Known: {known_cost:.2f}, Estimated: {estimated_cost:.2f}")
        except KeyError:
            pass

        try:
            trl_c1 = cfg_samp.get_sampled_value('TRL_C1')
            trl_c2 = cfg_samp.get_sampled_value('TRL_C2')
            trl_c3 = cfg_samp.get_sampled_value('TRL_C3')
            print(f"  Loading Times: Class1: {trl_c1:.2f}, Class2: {trl_c2:.2f}, Class3: {trl_c3:.2f}")
        except KeyError:
            pass

        try:
            fw = cfg_samp.get_sampled_value('FW')
            fo = cfg_samp.get_sampled_value('FO')
            fe = cfg_samp.get_sampled_value('FE')
            te = cfg_samp.get_sampled_value('TE')
            tl = cfg_samp.get_sampled_value('TL')
            print(f"  Fuel Consumptions: Waste: {fw:.2f}, Ore: {fo:.2f}, Empty: {fe:.2f}")
            print(f"  Truck Speeds: Empty: {te:.2f}, Loaded: {tl:.2f}")
        except KeyError:
            pass

        try:
            fsh = cfg_samp.get_sampled_value('FSH')
            ftr = cfg_samp.get_sampled_value('FTR')
            fcr = cfg_samp.get_sampled_value('FCR')
            fds = cfg_samp.get_sampled_value('FDS')
            print(f"  Equipment MTBF: Shovel: {fsh:.2f}, Truck: {ftr:.2f}, Crusher: {fcr:.2f}, Dump: {fds:.2f}")
        except KeyError:
            pass
            
    except KeyError as e:
        print(f"Episode summary error: {e}")

def update_csv_action(seq_id, action):
    """
    Legacy no-op kept for backward compatibility. 
    """
    return


def update_empty_obs_rew_row(observ, r_imm_d_pt, info, max_retries=10, retry_delay=1):
    """
    Deliver the (observation, immediate reward) for the current decision
    point to the Gym/RL side via the in-process channel (terminated=False)
    -- the 'next state + reward' half of one (s, a, r, s') transition. The
    matching action is consumed right after in Truck.process().
    """
    global RL_sched, sim_exit, _channel

    if RL_sched == False or sim_exit or _channel is None:
        return

    _channel.send_result(observ, r_imm_d_pt, terminated=False, info=info)
    _vprint(1, "\n *Code DES: sent observation + immediate reward to RL (in-process)")
    


def final_step_update(observ, r_epi, info, max_retries=10, retry_delay=1):
    """
    Deliver the terminal (observation, episode reward) to the Gym/RL side
    and flag the episode as done, via the in-process channel
    (terminated=True) -- answers the Gym's one pending action, since the
    Gym is always exactly one action ahead and blocked awaiting a result.
    """
    global RL_sched, _channel

    if RL_sched == False or _channel is None:
        return

    _channel.send_result(observ, r_epi, terminated=True, info=info)
    print("\n *Code DES: sent terminal observation + episode reward to RL (in-process)")


def get_shovel_from_integer(shovel_list, index):
    """Return the shovel object at `index` in `shovel_list`, or None if
    out of bounds."""
    if 0 <= index < len(shovel_list):
        return shovel_list[index]
    else:
        print("\n Shovel number out of range ")
        return None


def create_observation(num_shovels, num_trucks, shovel_data, active_truck_data, fleet_summary):
    """
    Build the Gym observation dict from shovel_data (list of
    (queue_len, status) tuples), active_truck_data (single truck's
    {'trip_count', 'phase'}), and fleet_summary ({'avg_trips',
    'recent_decisions', 'diversity_score'}).
    """
    shovel_id = np.zeros(num_shovels * 4, dtype=int)
    queue_length = np.zeros(num_shovels, dtype=float)
    sh_status = np.zeros(num_shovels, dtype=int)

    for i, (queue_len, status) in enumerate(shovel_data):
        # Bounded by a few trucks-per-shovel (not 2*num_trucks) so real
        # queue spreads (0-50) land in the middle of [0,1] instead of the
        # bottom ~0-0.42, where small differences are hard to discriminate.
        # Still clipped to 1.0 against rare congestion spikes.
        max_queue = max((num_trucks / max(num_shovels, 1)) * 3.0, 10.0)
        queue_length[i] = round(min(queue_len / max_queue, 1.0), 4)
        sh_status[i] = status

        bin_str = format(i + 1, '04b')  # 4 bits supports up to 16 shovels
        shovel_id[i * 4:i * 4 + 4] = list(map(int, bin_str))

    truck_id_active = np.zeros(1 * 6, dtype=int) 
    trips_complete_active = np.zeros(1, dtype=float)
    tr_status_active = np.zeros(1 * 3, dtype=int)

    truck_key = list(active_truck_data.keys())[0]
    truck_info = active_truck_data[truck_key]
    
    # Parse truck information
    trip_count = truck_info['trip_count']
    phase = truck_info['phase']
    truck_index = int(truck_key.split('.')[1]) - 1

    binary_index = f'{truck_index + 1:06b}'  # 6 bits supports up to 64 trucks
    truck_id_active[0:6] = list(map(int, binary_index))
    trips_complete_active[0] = round(trip_count / 500, 4)

    phase_list = [int(char) for char in phase]
    tr_status_active[0:3] = phase_list

    # Fleet context: average trip progress, recent shovel usage
    # distribution, and a diversity score, all for coordination.
    fleet_avg_trips = np.zeros(1, dtype=float)
    fleet_avg_trips[0] = round(fleet_summary['avg_trips'] / 500, 4)

    recent_shovel_usage = np.zeros(num_shovels, dtype=float)
    if len(fleet_summary['recent_decisions']) >= 5:
        recent_window = list(fleet_summary['recent_decisions'])[-10:]
        recent_counts = Counter(recent_window)
        
        for shovel_name, count in recent_counts.items():
            idx = int(shovel_name.split('_')[1])  # e.g. "Shovel_3" -> 3
            recent_shovel_usage[idx] = count / len(recent_window)

    fleet_diversity = np.zeros(1, dtype=float)
    fleet_diversity[0] = round(fleet_summary['diversity_score'], 4)

    observation = {
        "ShovelID": shovel_id.tolist(),
        "Queue_length": queue_length.tolist(),
        "SH_Status": sh_status.tolist(),

        "TruckID_Active": truck_id_active.tolist(),
        "Trips_complete_Active": trips_complete_active.tolist(),
        "TR_Status_Active": tr_status_active.tolist(),

        "Fleet_Avg_Trips": fleet_avg_trips.tolist(),
        "Recent_Shovel_Usage": recent_shovel_usage.tolist(),
        "Fleet_Diversity": fleet_diversity.tolist()
    }

    return observation


class RewardCalculator:
    def __init__(self, k, alpha=0.5):
        """
        k: sliding-window length for averaging. alpha: exponential
        weighting factor controlling how fast the weights increase.
        """
        self.k = k
        self.alpha = alpha
        self.lamda = 0.1
        self.trip_times = trip_times
        self.shovel_queues = shovel_queues

        self.PROD_W = 15.0

        cycle_len = self._estimate_cycle_length()
        self.TT_Avg_min = 0.0
        self.TT_Avg_max = max(3.0 * cycle_len, 30.0)   # ~3 full truck cycles
        self.Q_Avg_d_min = 0.0
        self.Q_Avg_d_max = max(2.0 * cycle_len, 20.0)  # ~2 cycles worth of wait

    @staticmethod
    def _estimate_cycle_length():
        """
        Estimate a representative single-truck cycle length (load + haul to
        crusher/dump + dump + haul back) from the sampled config parameters,
        instead of hardcoding it. Falls back to a conservative default if the
        config isn't available yet (e.g. at import time).
        """
        try:
            load_times = [cfg_samp.get_sampled_value(f'TRL_C{c}') for c in (1, 2, 3)]
            avg_load = sum(load_times) / len(load_times)

            haul_keys = ['SZ1C', 'SZ1D', 'SZ2C', 'SZ2D', 'SZ3C', 'SZ3D']
            haul_times = [cfg_samp.get_sampled_value(k) for k in haul_keys]
            avg_haul_leg = sum(haul_times) / len(haul_times)

            avg_dump = (cfg_samp.get_sampled_value('TRDM')
                        + cfg_samp.get_sampled_value('TRCR')) / 2.0

            # load + haul-out + dump + haul-back (haul-back approximated by the
            # same average leg time, since empty/loaded speed differ but the
            # distance is the same).
            return avg_load + 2 * avg_haul_leg + avg_dump
        except Exception:
            return 30.0  # conservative fallback; only used if config sampling fails

    def min_max_normalize(self, current_value, min_value, max_value):
        """
        Normalize a value using Min-Max normalization, clipped to [0, 1] so an
        occasional outlier (e.g. a queue spike during a multi-shovel breakdown)
        cannot push the normalized value outside its intended range.
        """
        if max_value == min_value:
            raise ValueError("max_value and min_value cannot be the same")

        normalized_value = (current_value - min_value) / (max_value - min_value)
        return max(0.0, min(1.0, normalized_value))

    def update(self, tau_d, Q_SH_d):
        """ Update the sliding window with the current decision point data """

        self.trip_times.append(tau_d)
        self.shovel_queues.append(Q_SH_d)
    
    def compute_weighted_average(self, data):
        """ Compute the weighted (exponential) average for the sliding window data """
        n = len(data)
        if n == 0:
            return 0
        
        # Calculate exponential weights
        weights = np.exp(self.alpha * np.arange(1, n + 1))
        
        # Normalize the weights so they sum to 1
        normalized_weights = weights / np.sum(weights)
        
        # Compute the weighted average
        weighted_avg = np.dot(normalized_weights, data)
        
        return weighted_avg

    @staticmethod
    def compute_dispatch_advantage(chosen_queue_len, all_queue_lens):
        """
        Action-consequential signal in [-1, +1]: +1 if the chosen shovel had
        the shortest operational queue, -1 if it had the longest, linear in
        between (symmetric, so "good dispatch" and "less bad dispatch" both
        have a real gradient -- an earlier [-1, 0] form was structurally
        pessimistic and taught the policy to spread uniformly rather than
        pick well).

        `all_queue_lens` should already be filtered to operational
        (non-broken) shovels by the caller, so a broken shovel never sets
        the bar an operational choice is judged against.
        """
        if not all_queue_lens:
            return 0.0
        best_alt = min(all_queue_lens)
        spread = max(all_queue_lens) - best_alt
        if spread <= 1e-9:
            return 0.0  # every alternative was equally good; no signal here
        advantage = 1.0 - 2.0 * (chosen_queue_len - best_alt) / spread
        return max(-1.0, min(1.0, advantage))

    def compute_r_imm_d(self, chosen_queue_len=None, all_queue_lens=None,
                        trips_delta=0, chosen_shovel_id=None):
        """
        Compute the immediate, action-consequential reward at the current
        decision point, as a dense, mostly-positive signal tied to the
        action just taken:
          - dispatch_term: how the chosen shovel compared to the best
            available alternative (the core action-consequential signal --
            load-spreading should emerge as a consequence of good
            dispatching, not be forced by a separate regularizer).
          - production_reward: a positive reward for each trip THIS truck
            completed since its last decision (trips_delta, crusher AND
            dump), credited here since under the lockstep (s, a, r, s')
            protocol that trip is a consequence of the truck's previous
            action. Crediting every completed trip (not just epsilon-routed
            crusher trips) keeps the signal action-correlated and
            low-variance.

        diversity_penalty and streak_penalty terms are intentionally
        absent from certain paths: once dispatch quality is rewarded
        directly, spreading load is what a queue-aware policy does anyway.
        """
        if len(self.trip_times) == 0 or len(self.shovel_queues) == 0:
            return {'r_imm_d': 0.0, 'trip_times': [], 'shovel_queues': []}


        if chosen_queue_len is not None and all_queue_lens is not None:
            dispatch_term = 1.0 * self.compute_dispatch_advantage(chosen_queue_len, all_queue_lens)
            _spread = (max(all_queue_lens) - min(all_queue_lens)) if all_queue_lens else 0.0
            _degenerate = _spread <= 1e-9
        else:
            dispatch_term = 0.0  # no dispatch info (truck's first decision this episode)
            _spread = None
            _degenerate = None

        try:
            _streak_ratio = calculate_streak_penalty()
            streak_term = -0.30 * _streak_ratio
        except Exception:
            _streak_ratio = None
            streak_term = 0.0

        try:
            crush_so_far = globals().get('total_crush_trips', 0) or 0
            pvol_now = crush_so_far * load_per_trip
            progress = min(1.0, pvol_now / targ_pvol) if targ_pvol else 0.0
            per_trip_reward = self.PROD_W * (load_per_trip / targ_pvol) * (1.0 + progress)
        except Exception:
            per_trip_reward = 0.0
        production_reward = per_trip_reward * max(0, trips_delta)

        r_imm_d = dispatch_term + production_reward + streak_term

        # Per-decision dispatch diagnostics (separate CSV from the per-episode
        # metrics log) -- lets the spread-zero / degenerate-case theory be
        # checked empirically against real training data instead of guessed
        # at. See episode_metrics_logger.log_dispatch_decision.
        global rl_decision_index
        rl_decision_index += 1
        try:
            log_dispatch_decision(
                csv_path=_dispatch_log_path(),
                episode=cfg_samp.episode_count if hasattr(cfg_samp, 'episode_count') else None,
                decision_index=rl_decision_index,
                chosen_queue_len=chosen_queue_len,
                all_queue_lens=all_queue_lens,
                spread=_spread,
                degenerate=_degenerate,
                dispatch_term=dispatch_term,
                chosen_shovel_id=chosen_shovel_id,
                num_operational=(len(all_queue_lens) if all_queue_lens else None),
                streak_ratio=_streak_ratio,
                streak_term=streak_term,
                production_reward=production_reward,
            )
        except Exception:
            pass  # never let diagnostic logging interrupt the simulation

        if _DES_VERBOSE >= 2 and rl_decision_index % 20 == 0:
            episode_num = cfg_samp.episode_count if hasattr(cfg_samp, 'episode_count') else '?'
            print(f"[Episode {episode_num}] Step {rl_decision_index}: "
                  f"Dispatch: {dispatch_term:+.3f}, Streak: {streak_term:+.3f}, "
                  f"ProdRew: {production_reward:+.4f}, Reward: {r_imm_d:+.4f}")


        return {
            'r_imm_d': r_imm_d,
            'trip_times': list(self.trip_times),
            'shovel_queues': list(self.shovel_queues)
        }


class ShovelWaitTimeTracker:
    def __init__(self, num_shovels):
        self.num_shovels = num_shovels
        self.shovel_waiting_times = {f"Shovel_{i}": 0 for i in range(num_shovels)}
        self.shovel_request_counts = {f"Shovel_{i}": 0 for i in range(num_shovels)}
        self.shovel_truck_waiting_times = {f"Shovel_{i}": {} for i in range(num_shovels)}  # Track individual truck waiting times

    def add_truck_to_queue(self, shovel_name, arrival_time, truck_id):
        """
        Called when a truck joins the shovel queue.
        Increments the request count for the shovel.
        """
        self.shovel_request_counts[shovel_name] += 1
        self.shovel_truck_waiting_times[shovel_name][truck_id] = arrival_time

    def remove_truck_from_queue(self, shovel_name, departure_time, truck_id):
        """
        Called when a truck leaves the shovel queue.
        Decrements the request count for the shovel and removes the last max waiting time of the truck.
        """
        if truck_id in self.shovel_truck_waiting_times[shovel_name]:
            arrival_time = self.shovel_truck_waiting_times[shovel_name].pop(truck_id)
            waiting_time = departure_time - arrival_time
            self.shovel_waiting_times[shovel_name] += waiting_time
            if self.shovel_request_counts[shovel_name] > 0:
                self.shovel_request_counts[shovel_name] -= 1

    def get_average_waiting_times(self):
        """
        Returns the average waiting time for each shovel.
        """
        average_times = {}
        for shovel, total_waiting_time in self.shovel_waiting_times.items():
            request_count = self.shovel_request_counts[shovel]
            average_times[shovel] = total_waiting_time / request_count if request_count > 0 else 0
        return average_times

class Truck(sim.Component):
    scheduler_assigned = False
    trucks_failed = 0     # count of trucks that have experienced breakdown
    trucks_to_fail = None  # IDs of trucks that will fail

    def setup(self):
        global RL_sched, def_schdlr_choice, truck_breakdown_manager, Num_trucks
        global truck_last_trip_times
        global truck_trip_counts  
        global truck_phases       

        self.truck_id = ''.join(filter(str.isdigit, self.name()))
        self.trip_count = 0
        self.breakdown_display = None

        if Truck.trucks_to_fail is None:
            num_trucks_to_fail = min(int(cfg_samp.get_sampled_value('TTF')), Num_trucks)
            Truck.trucks_to_fail = random.sample(range(1, Num_trucks + 1), num_trucks_to_fail)

        if int(self.truck_id) in Truck.trucks_to_fail:
            self.next_breakdown_time = cfg_samp.get_sampled_value('TIB')
        else:
            self.next_breakdown_time = float('inf')

        self.has_failed = False
        self.time_to_repair = cfg_samp.get_sampled_value('RTR')
        self.phase = phase_shovel
        self.last_phase = None
        self.shovel_name = None
        truck_trip_counts[self.name()] = self.trip_count
        truck_last_trip_times[self.truck_id] = 0

        self.reward_calculator = RewardCalculator(k, alpha)

        # (chosen_queue_len, alt_queue_lens) from this truck's most recent
        # dispatch, consumed by compute_r_imm_d() at its next decision point.
        self._pending_chosen_queue = None
        self._pending_alt_queues = None
        # Shovel index chosen at the most recent dispatch (for the
        # dispatch-log diagnostic), consumed at the next decision point.
        self._pending_chosen_shovel = None
        # Per-truck (not global) crush-trip bookkeeping: a trip completed
        # since this truck's last decision is a consequence of its own
        # previous action, credited at its next decision point.
        self.crush_trips_total = 0
        self._crush_trips_at_last_decision = 0
        # Per-truck ALL-trips bookkeeping (crusher + dump). This is what the
        # production-progress reward is now credited on (see compute_r_imm_d),
        # since every completed trip -- not just the epsilon-routed crusher
        # ones -- is a direct consequence of dispatch quality. self.trip_count
        # already increments once per completed cycle in process(); we only
        # need to remember its value at the last decision to form the delta.
        self._trips_at_last_decision = 0

    def handle_breakdown(self):

        if int(self.truck_id) not in Truck.trucks_to_fail:
            return False

        # Check if we haven't reached the total failure target
        max_failures = len(Truck.trucks_to_fail)
        if Truck.trucks_failed >= max_failures:
            return False

        if not self.has_failed:
            if env.now() >= self.next_breakdown_time:
                self.has_failed = True
                Truck.trucks_failed += 1
                return True
            return False
            


    def update_phase(self, new_phase): 
        global truck_phases
        self.phase = new_phase
        truck_phases[self.name()] = self.phase  # Update truck phase dictionary

        # Add breakdown display when entering breakdown phase
        if new_phase == phase_broken_down:
            # Calculate even spacing using actual screen dimensions
            min_x = 400
            max_x = 600
            spacing=10
            x_pos = min_x + (int(self.truck_id) - 1) * spacing
        
            # Show broken truck image with ID
            self.breakdown_display = sim.AnimateImage(
                "dump_truck_broken.png", 
                width=70,
                x=x_pos,
                y=50,
                text=str(self.truck_id),
                text_anchor="c",
                fontsize=20,
                textcolor="yellow"
            )
        # Remove breakdown display when leaving breakdown phase
        elif self.breakdown_display is not None:
            self.breakdown_display.remove()
            self.breakdown_display = None


    def process(self):
        global total_trips
        global total_crush_trips
        global r_imm_d_pt
        first_trip = True
        processed_items = set()
        seq_id=None
        action = None
                        
        while True:
            while True:
                curr_truck_id = ''.join(filter(str.isdigit, self.name()))
                elapsed_time = env.now() - shift_start_time
                time_left = shift_dura - elapsed_time

                # First scheduling round uses the default method (shortest
                # queue length) regardless of RL/classical mode.
                if (first_trip == True and RL_sched == False):
                    selected_shovel = scheduler_assign(def_schdlr_choice, truck_id=int(curr_truck_id))
                    first_trip = False
                    add_item(selected_shovel.name())

                elif (first_trip == True and RL_sched == True):
                    selected_shovel = scheduler_assign(choice, truck_id=int(curr_truck_id))
                    first_trip = False
                    add_item(selected_shovel.name())

                elif (first_trip == False and RL_sched == True):
                    tau_d = truck_last_trip_times[curr_truck_id]

                    average_waiting_times = shovel_wait_time_tracker.get_average_waiting_times()
                    Q_SH_d = sum(average_waiting_times.values()) / len(average_waiting_times)
                    self.reward_calculator.update(tau_d, Q_SH_d)

                    # This truck's own throughput since its last decision --
                    # a direct consequence of that action.
                    trips_delta = self.trip_count - self._trips_at_last_decision
                    self._trips_at_last_decision = self.trip_count
                    r_dict = self.reward_calculator.compute_r_imm_d(
                        chosen_queue_len=getattr(self, '_pending_chosen_queue', None),
                        all_queue_lens=getattr(self, '_pending_alt_queues', None),
                        trips_delta=trips_delta,
                        chosen_shovel_id=getattr(self, '_pending_chosen_shovel', None),
                    )
                    r_imm_d_pt = r_dict['r_imm_d']
                    self._pending_chosen_queue = None
                    self._pending_alt_queues = None
                    self._pending_chosen_shovel = None

                    # print_trip_counts returns ALL truck data; observation
                    # needs only the active truck's.
                    all_truck_data = print_trip_counts(ppflag=0)
                    active_truck_data = {self.name(): all_truck_data[self.name()]}

                    all_trips = list(truck_trip_counts.values())
                    fleet_summary = {
                        'avg_trips': sum(all_trips) / len(all_trips) if all_trips else 0,
                        'recent_decisions': all_trk_shv_dec,
                        'diversity_score': diversity_score()
                    }

                    shovel_status_snapshot = print_resource_status(pflag=0)['Shovels']
                    observ = create_observation(Num_shovels, Num_trucks, shovel_status_snapshot, active_truck_data,fleet_summary)

                    info = {'truck_id': curr_truck_id}

                    global sim_exit
                    if sim_exit:
                        break
                    else:
                        # Sends the (next-state, reward) half of this
                        # (s, a, r, s') transition over the channel.
                        update_empty_obs_rew_row(observ, r_imm_d_pt, info)

                    _vprint(1, "\n Querying RL Policy \n ")
                    # Block for the action that answers this decision. The
                    # channel verifies it matches the result just sent.
                    #
                    # THREAD-SAFETY: _RUN_DES_LOCK is held throughout runDes
                    # so concurrent DES threads (eval episodes during
                    # training) can't race on the module globals, but
                    # blocking here while holding it would deadlock the eval
                    # callback -- so this thread's globals are snapshotted,
                    # the lock released for the wait, then reacquired and
                    # the snapshot restored on wake-up, regardless of what
                    # eval did to the globals during the gap.
                    _snap_for_wait = _snapshot_des_globals()
                    _RUN_DES_LOCK.release()
                    try:
                        action = _channel.wait_for_action()
                    except ChannelStopped:
                        # Reset/shutdown was requested; unwind cleanly.
                        _RUN_DES_LOCK.acquire()
                        _restore_des_globals(_snap_for_wait)
                        sim_exit = True
                        return
                    else:
                        _RUN_DES_LOCK.acquire()
                        _restore_des_globals(_snap_for_wait)

                    if action is None:
                        sim_exit = True
                        return

                    selected_shovel = get_shovel_from_integer(shovels, int(action))
                    add_item(selected_shovel.name())
                    _vprint(1, f"\n\nCode DES: Received Action {action}.\n\n")

                    # Sanity check: with action masking active, this should
                    # stay at 0 -- the agent should never be offered a
                    # broken shovel. A live signal that masking is working
                    # (jumps off zero if the mask wrapper is ever removed).
                    global broken_shovel_dispatch_count
                    chosen_idx_check = int(action)
                    if (0 <= chosen_idx_check < len(shovel_status_snapshot)
                            and shovel_status_snapshot[chosen_idx_check][1] == 0):
                        broken_shovel_dispatch_count += 1
                        print(f"WARNING: dispatched to broken shovel (action={action}) "
                              f"-- action masking may not be active.")

                    # Score the chosen shovel against operational
                    # alternatives available at the instant of choice (only
                    # operational shovels set the comparison bar). Stashed
                    # here and consumed at this truck's next decision point,
                    # per the lockstep (s, a, r, s') protocol.
                    operational_queues = [q for (q, status) in shovel_status_snapshot if status == 1]
                    chosen_idx = int(action)
                    chosen_queue_len = (shovel_status_snapshot[chosen_idx][0]
                                         if 0 <= chosen_idx < len(shovel_status_snapshot) else 0)
                    self._pending_chosen_queue = chosen_queue_len
                    self._pending_alt_queues = operational_queues
                    self._pending_chosen_shovel = chosen_idx

                    # Shovel-hugging monitor: record every RL choice, cheap
                    # per-decision, summarized into concentration stats once
                    # at episode end.
                    global episode_shovel_choices, episode_queue_sum, episode_queue_count
                    episode_shovel_choices[chosen_idx] = episode_shovel_choices.get(chosen_idx, 0) + 1
                    episode_queue_sum += chosen_queue_len
                    episode_queue_count += 1

                elif (RL_sched == False):
                    selected_shovel = scheduler_assign(def_schdlr_choice, truck_id=int(curr_truck_id))  
                    add_item(selected_shovel.name())

                               
                self.shovel_name = selected_shovel.name()  # Set the shovel name
                shovel_wait_time_tracker.add_truck_to_queue(self.shovel_name, env.now(), self.truck_id)

                yield self.request((selected_shovel,1,4))
                start_time = env.now()

                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    shovel_classes = cfg_samp.get_sampled_value('shovel_performance_class')
                    performance_class = shovel_classes[shovel_id]
                    trk_load = cfg_samp.get_sampled_value(f'TRL_C{performance_class}')
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get loading time for shovel {selected_shovel.name()}: {e}")


                yield self.hold(trk_load)

                if self.isbumped():
                    trk_load -= env.now() - start_time
                    continue
                break

            shovel_wait_time_tracker.remove_truck_from_queue(self.shovel_name, env.now(), self.truck_id)
            self.release()
     
            # Epsilon-greedy: epsilon=1 routes to crusher only.
            if random.random() < epsilon:
                self.update_phase(phase_travel_shovel_crusher) 
                selected_crusher = random.choice(crushers)  
                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    location_clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
                    location_cluster = location_clusters[shovel_id]
                    norm_time_cr = cfg_samp.get_sampled_value(f'SZ{location_cluster}C')
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get crusher travel time for shovel {selected_shovel.name()}: {e}")

                vmax_rnd_cr = 300 / (norm_time_cr)
                traj_c01 = sim.TrajectoryCircle(radius=200, x_center=750, y_center=600, angle0=230, angle1=360, v0=0, vmax=vmax_rnd_cr)

                if self.handle_breakdown():
                    self.passivate() 
                    curr_phase = self.phase
                    self.update_phase(phase_broken_down)
                    breakdown_start_time = env.now() 
                    yield self.hold(self.time_to_repair)

                    # Adjust travel-time/trajectory to include repair time,
                    # for the animation to reflect the delay.
                    repair_end_time = env.now()
                    repair_duration = repair_end_time - breakdown_start_time
                    adjusted_time_cr = norm_time_cr + repair_duration
                    vmax_rnd_cr = 300 / adjusted_time_cr
                    traj_c01 = sim.TrajectoryCircle(radius=200, x_center=750, y_center=600, angle0=230, angle1=360, v0=0, vmax=vmax_rnd_cr)

                    self.activate()  
                    self.update_phase(curr_phase) 
                    self.next_breakdown_time = env.now() + cfg_samp.get_sampled_value('FTR')

                txt = str(self.name())
                num = txt.split('.')[-1]
                self.dump_truck_cr = sim.AnimateImage("dump_truck_01.png", width=70, x=traj_c01.x, y=traj_c01.y, angle=traj_c01.angle, text=str(num), text_anchor="c", fontsize=20, textcolor="yellow")
                yield self.hold(norm_time_cr)
                self.dump_truck_cr.remove()

                yield self.request(selected_crusher)
                crush_dump = cfg_samp.get_sampled_value('TRCR')
                yield self.hold(crush_dump)
                self.release()

                global total_crush_trips
                total_crush_trips += 1
                # Per-truck count for the production reward, credited to
                # this truck's previous dispatch at its next decision point.
                self.crush_trips_total += 1

                self.update_phase(phase_travel_crusher_shovel) 

                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    location_clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
                    location_cluster = location_clusters[shovel_id]
                    norm_time_rev_cr = cfg_samp.get_sampled_value(f'SZ{location_cluster}C')
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get return travel time from crusher: {e}")
                vmax_rnd_rev_cr = 330/ (norm_time_rev_cr)
                traj_c02 = sim.TrajectoryCircle(radius=230, x_center=750, y_center=600, angle0=330, angle1=230, v0 = 0, vmax  = vmax_rnd_rev_cr)
                self.dump_truck_cr_2 = sim.AnimateImage("dump_truck_02.png", width=70, x=traj_c02.x, y=traj_c02.y,angle=traj_c02.angle,text=str(num), text_anchor = "c", fontsize= 20, textcolor = "red")
                yield self.hold(norm_time_rev_cr)
                self.dump_truck_cr_2.remove()

                truck_last_trip_times[self.truck_id] = norm_time_cr + crush_dump + norm_time_rev_cr

            else:
                self.update_phase(phase_travel_shovel_dump)  

                selected_dump = random.choice(dumps)
                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    location_clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
                    location_cluster = location_clusters[shovel_id]
                    norm_time = cfg_samp.get_sampled_value(f'SZ{location_cluster}D')
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get dump travel time for shovel {selected_shovel.name()}: {e}")
                vmax_rnd = 830 / norm_time
                traj_03 = sim.TrajectoryCircle(radius=230, x_center=300, y_center=470, angle0=360, angle1=150, vmax=vmax_rnd)

                if self.handle_breakdown():
                    self.passivate() 
                    curr_phase = self.phase
                    self.update_phase(phase_broken_down)
                    breakdown_start_time = env.now() 
                    yield self.hold(self.time_to_repair)

                    repair_end_time = env.now()
                    repair_duration = repair_end_time - breakdown_start_time
                    adjusted_time = norm_time + repair_duration
                    vmax_rnd = 830 / adjusted_time
                    traj_03 = sim.TrajectoryCircle(radius=230, x_center=300, y_center=470, angle0=360, angle1=150, vmax=vmax_rnd)

                    self.activate() 
                    self.update_phase(curr_phase) 
                    self.next_breakdown_time = env.now() + cfg_samp.get_sampled_value('FTR')

                txt = str(self.name())
                num = txt.split('.')[-1]
                self.dump_truck_ds = sim.AnimateImage("dump_truck_01.png", width=70, x=traj_03.x, y=traj_03.y, angle=traj_03.angle, text=str(num), text_anchor="c", fontsize=20, textcolor="yellow")   
                yield self.hold(norm_time)
                self.dump_truck_ds.remove()

                yield self.request(selected_dump)
                dump_time = cfg_samp.get_sampled_value('TRDM')
                yield self.hold(dump_time)
                self.release()
                self.update_phase(phase_travel_dump_shovel)

                try:
                    shovel_id = int(selected_shovel.name().split('_')[1])
                    location_clusters = cfg_samp.get_sampled_value('shovel_location_cluster')
                    location_cluster = location_clusters[shovel_id]
                    norm_time_rev = cfg_samp.get_sampled_value(f'SZ{location_cluster}D')
                except (KeyError, IndexError) as e:
                    print(f"ERROR: Could not get return travel time from dump: {e}")
                vmax_rnd_rev = 330/ (norm_time_rev)
                traj_04 = sim.TrajectoryCircle(radius=180, x_center=300, y_center=470, angle0=150, angle1=360, v0 = 0, vmax  = vmax_rnd_rev)
                self.dump_truck_2 = sim.AnimateImage("dump_truck_02.png", width=70, x=traj_04.x, y=traj_04.y,angle=traj_04.angle,text=str(num), text_anchor = "c", fontsize= 20, textcolor = "red")
                yield self.hold(norm_time_rev)
                self.dump_truck_2.remove()

                truck_last_trip_times[self.truck_id] = norm_time + dump_time + norm_time_rev

            self.trip_count += 1
            truck_trip_counts[self.name()] = self.trip_count

            global total_trips
            total_trips = sum(truck_trip_counts.values())


class BreakdownManager(sim.Component):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        self.shovel_breakdowns = []
    
    def _select_shovels_to_fail(self):
        """Select which shovels fail this episode, with unique jittered
        initial breakdown times so failures don't all trigger at once."""
        num_shovels = cfg_samp.get_sampled_value('SH')
        num_shovels_to_fail = min(int(cfg_samp.get_sampled_value('STF')), num_shovels)

        shovel_ids = [int(s.name().split('_')[1]) for s in shovels]
        shovels_to_fail = random.sample(shovel_ids, num_shovels_to_fail)

        breakdown_schedule = {}
        for s in shovels:
            shovel_id = int(s.name().split('_')[1])
            if shovel_id in shovels_to_fail:
                base_time = cfg_samp.get_sampled_value('SIB')
                jitter = random.uniform(0, 50)
                initial_breakdown = base_time + jitter

                breakdown_schedule[s.name()] = {
                    'should_fail': True,
                    'initial_breakdown': initial_breakdown
                }
            else:
                breakdown_schedule[s.name()] = {
                    'should_fail': False,
                    'initial_breakdown': None
                }
        
        print(f"\nSelected {num_shovels_to_fail} shovels to fail with schedule:")
        for name, schedule in breakdown_schedule.items():
            if schedule['should_fail']:
                print(f"{name}: Initial breakdown at {schedule['initial_breakdown']}")
        
        return breakdown_schedule

    def process(self):
        try:
            breakdown_schedule = self._select_shovels_to_fail()

            for idx, shovel in enumerate(shovels):
                schedule = breakdown_schedule[shovel.name()]

                if idx > 0:  # small delay for temporal separation
                    yield self.hold(random.uniform(0.1, 0.5))

                breakdown = IndividualShovelBreakdown(
                    shovel=shovel,
                    shovel_animations=shovel_animations,
                    should_fail=schedule['should_fail'],
                    initial_breakdown_time=schedule['initial_breakdown']
                )
                self.shovel_breakdowns.append(breakdown)

            yield self.passivate()

        except Exception as e:
            print(f"Error in BreakdownManager process: {str(e)}")
            raise


class IndividualShovelBreakdown(sim.Component):
    """Individual component to handle each shovel's breakdowns"""
    def setup(self, shovel, shovel_animations, should_fail, initial_breakdown_time=None):
        self.shovel = shovel
        self.shovel_animations = shovel_animations
        self.has_failed = False
        self.is_broken = False
        self.should_fail = should_fail

        if self.should_fail:
            self.next_breakdown_time = initial_breakdown_time
            print(f"{self.shovel.name()}: Initial breakdown scheduled at {self.next_breakdown_time}")
        else:
            self.next_breakdown_time = float('inf')

    def process(self):
        try:
            while True:
                if not self.has_failed and self.should_fail:
                    time_until_breakdown = max(0, self.next_breakdown_time - self.env.now())
                    if time_until_breakdown > 0:
                        yield self.hold(time_until_breakdown)

                    yield self.request((self.shovel, 1, 1))
                    print(f"\nShovel {self.shovel.name()} breaking down at time {self.env.now()}")

                    base_repair_time = cfg_samp.get_sampled_value('RSH')
                    repair_time = base_repair_time + random.uniform(0, 5)

                    self.has_failed = True
                    self.is_broken = True

                    self.shovel_animations[self.shovel].image = "shovel_broken.png"
                    print(f"Repair time for {self.shovel.name()}: {repair_time:.2f}")

                    yield self.hold(repair_time)

                    self.shovel_animations[self.shovel].image = "shovel_active.png"
                    self.release()
                    self.is_broken = False
                    
                    # Schedule next breakdown using FSH with jitter
                    base_next_time = cfg_samp.get_sampled_value('FSH')
                    jitter = random.uniform(0, 10)
                    self.next_breakdown_time = self.env.now() + base_next_time + jitter
                    print(f"Next breakdown for {self.shovel.name()} scheduled at {self.next_breakdown_time}")
                
                else:
                    time_until_next_shift = shift_dura - (self.env.now() % shift_dura)
                    yield self.hold(time_until_next_shift)

                    self.has_failed = False
                    if self.should_fail:
                        base_time = cfg_samp.get_sampled_value('SIB')
                        jitter = random.uniform(0, 10)
                        self.next_breakdown_time = self.env.now() + base_time + jitter
                        print(f"New shift: {self.shovel.name()} scheduled for breakdown at {self.next_breakdown_time}")
                        
        except Exception as e:
            print(f"Error in IndividualShovelBreakdown process for {self.shovel.name()}: {str(e)}")
            raise


# --- Trace / snapshot helpers -----------------------------------------------
# `ppflag` / `pflag` are kept for backward compatibility with existing call
# sites (both are always invoked with the flag set to 0 on the live decision
# path). Verbose console printing has been retired in favor of the
# DES_VERBOSE gate defined near the top of this module.

def print_trip_counts(ppflag=1):
    """Return {truck_id: {'trip_count': int, 'phase': str}} for every truck."""
    truck_info = {}
    for truck_id, trip_count in truck_trip_counts.items():
        phase = truck_phases.get(truck_id, '000')
        truck_info[truck_id] = {'trip_count': trip_count, 'phase': phase}
    return truck_info


class PrintTripCountsEvent(sim.Component):
    """Periodically refreshes the truck-trip-count snapshot."""
    def process(self):
        while True:
            yield self.hold(5)
            print_trip_counts()


class TrackIdleTime(sim.Component):
    """Accumulates per-shovel idle time between checks."""
    def process(self):
        while True:
            yield self.hold(5)
            current_time = env.now()
            for shovel in shovels:
                if not shovel.claimers():
                    shovel_idle_times[shovel.name()] += current_time - shovel_last_check[shovel.name()]
                shovel_last_check[shovel.name()] = current_time


class PrintShovelIdleTimes(sim.Component):
    """Recomputes the fleet-average shovel idle time."""
    def process(self):
        while True:
            yield self.hold(1)
            global avg_idle_orig_time
            avg_idle_orig_time = sum(shovel_idle_times.values()) / len(shovels)


class PrintClaimersStatusEvent(sim.Component):
    """Periodically refreshes the resource-status snapshot."""
    def process(self):
        while True:
            yield self.hold(10)
            print_resource_status()


def print_resource_status(pflag=1):
    """Snapshot claimer/requester counts and breakdown status for every
    shovel, crusher, and dump, keyed as
    {'Shovels': [(total, status), ...], 'Crushers': [...], 'Dumps': [...]}.
    """
    resource_status = {}

    def is_resource_broken_down(resource):
        return 'breakdownevent' in str(resource.claimers().head())

    resource_status['Shovels'] = []
    for shovel in shovels:
        total = len(list(shovel.claimers())) + len(list(shovel.requesters()))
        status = 1 if not is_resource_broken_down(shovel) else 0
        resource_status['Shovels'].append((total, status))

    resource_status['Crushers'] = []
    for crusher in crushers:
        total = len(list(crusher.claimers())) + len(list(crusher.requesters()))
        status = 1 if not is_resource_broken_down(crusher) else 0
        resource_status['Crushers'].append((total, status))

    resource_status['Dumps'] = []
    for dump in dumps:
        total = len(list(dump.claimers())) + len(list(dump.requesters()))
        status = 1 if not is_resource_broken_down(dump) else 0
        resource_status['Dumps'].append((total, status))

    return resource_status



# Global lock so concurrent KPICalculator instances (e.g. an eval thread and
# a train thread, if that ever overlaps) don't interleave rows mid-write to
# the same shared KPI log file.
_KPI_WRITE_LOCK = threading.Lock()


class KPICalculator(sim.Component):
    """
    Per-tick KPI logger. One row every HIGH_FREQ sim-minutes plus extra
    rows on equipment state changes.

    Updated for multi-episode play/eval runs:
      1. Accepts an external `csv_path` (the same path runDes already
         receives) so every episode in a play/eval run appends to ONE file
         instead of each episode getting its own timestamped file.
      2. Adds Episode / Scenario / Seed columns for provenance --
         downstream aggregation groups on Episode instead of globbing many
         files.
      3. Append-safe across episodes: header is written exactly once (when
         the file does not yet exist or is empty); subsequent
         KPICalculator instances attach to the same file without
         truncating it.
      4. Filename fallback (when no csv_path is passed) preserves the old
         timestamped behavior, so legacy callers are unaffected.

    `episode_idx` is passed in by the caller (GymEnv forwards its own
    self._episode_counter through runDes). We deliberately do NOT maintain
    a second counter here -- duplicating that state would be a source of
    drift between the GymEnv's episode count and what gets logged.
    """

    HEADERS_BASE = [
        'Episode',
        'Scenario',
        'Seed',
        'Timestamp',
        'Shift_Number',
        'Shovel_Queue_Lengths',
        'Crusher_Queue_Lengths',
        'Dump_Queue_Lengths',
        'Changed_Shovel',
        'New_State',
        'Changed_Truck',
        'New_State_Truck',
        'Trips_Per_Hour',
        'Production_Volume_Per_Hour',
        'Total_Production_Volume',
        'Cost_Per_Ton',
        'Total_Fuel_Consumption',
    ]

    @staticmethod
    def _derive_kpi_path(csv_path, scenario_name, seed):
        """
        Build the KPI log filename.

        If csv_path is provided (the normal play/eval case), put the KPI
        log next to it with a `kpi_log_` prefix and the original base
        name, so all artifacts of one run share a stem and one file holds
        every episode.

        If csv_path is None (legacy call path), fall back to the original
        timestamped name so nothing else breaks.
        """
        if csv_path:
            base = os.path.splitext(os.path.basename(csv_path))[0]
            directory = os.path.dirname(os.path.abspath(csv_path)) or "."
            return os.path.join(directory, f"kpi_log_{base}.csv")

        scenario_part = f"scenario_{scenario_name}_" if scenario_name else ""
        seed_part = f"seed_{seed}_" if seed is not None else ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"kpi_log_{scenario_part}{seed_part}{timestamp}.csv"

    def setup(self, scenario_name=None, seed=None, csv_path=None,
              episode_idx=0, kpi_log_path=None):
        self.HIGH_FREQ = 2
        self.SHIFT_DURATION = shift_dura
        self.HOUR = 60  # Assuming time units are in minutes
        self.last_high_freq_update = 0
        self.last_shift_update = 0
        self.last_hour_update = 0

        self.episode_idx = episode_idx
        self.scenario_name = scenario_name if scenario_name is not None else ""
        self.seed = seed if seed is not None else ""

        # Explicit kpi_log_path wins (results_paths.py convention);
        # otherwise derive from csv_path (legacy); otherwise timestamped.
        if kpi_log_path:
            self.csv_filename = kpi_log_path
        else:
            self.csv_filename = self._derive_kpi_path(csv_path, scenario_name, seed)
        self.file_exists = os.path.exists(self.csv_filename) and os.path.getsize(self.csv_filename) > 0
        self.shift_counter = 0

        self.last_hour_trips = 0
        self.last_hour_volume = 0
        self.trips_per_hour = 0
        self.volume_per_hour = 0

        self.shovel_states = {shovel.name(): {'state': 'operational', 'last_state_change': 0} 
                            for shovel in shovels}
        self.truck_states = {truck.name(): {'state': 'operational', 'last_state_change': 0} 
                           for truck in truck}

        self.shovel_monitors = {
            shovel.name(): {'queue': shovel.requesters()} for shovel in shovels
        }
        self.crusher_monitors = {
            crusher.name(): {'queue': crusher.requesters()} for crusher in crushers
        }
        self.dump_monitors = {
            dump.name(): {'queue': dump.requesters()} for dump in dumps
        }

        self.headers = list(self.HEADERS_BASE)

        if RL_sched:
            # match original ordering: Immediate_Reward after the state
            # change cols, Shift_Reward at the end.
            insert_at = self.headers.index('Changed_Shovel')
            self.headers.insert(insert_at, 'Immediate_Reward')
            self.headers.append('Shift_Reward')
        
        if not self.file_exists:
            self.initialize_csv()

    def calculate_hourly_metrics(self):
        """Calculate metrics that are tracked on an hourly basis"""
        global total_trips, load_per_trip, total_crush_trips
        
        current_hour = self.env.now() // self.HOUR
        if current_hour > self.last_hour_update // self.HOUR:
            current_trips = total_trips
            self.trips_per_hour = current_trips - self.last_hour_trips
            self.last_hour_trips = current_trips

            current_volume = total_crush_trips * load_per_trip
            self.volume_per_hour = current_volume - self.last_hour_volume
            self.last_hour_volume = current_volume

            self.last_hour_update = self.env.now()

        return {
            'Trips_Per_Hour': self.trips_per_hour,
            'Production_Volume_Per_Hour': self.volume_per_hour
        }

    def is_shovel_broken(self, shovel):
        """Return True if `shovel` is currently broken down."""
        for claimer in shovel.claimers():
            claimer_str = str(claimer).lower()
            if 'breakdownevent' in claimer_str or 'individualshovelbreakdown' in claimer_str:
                return True
        return False

    def check_equipment_states(self):
        """Check every shovel/truck for a state change since last check,
        writing an immediate CSV row for each transition."""
        current_time = self.env.now()
        changed_equipment = []

        for shovel in shovels:
            current_state = 'breakdown' if self.is_shovel_broken(shovel) else 'operational'
            prev_state = self.shovel_states[shovel.name()]['state']
            
            if current_state != prev_state:
                self.shovel_states[shovel.name()]['state'] = current_state
                self.shovel_states[shovel.name()]['last_state_change'] = current_time
                
                metrics = {
                    'Timestamp': current_time,
                    'Shift_Number': self.shift_counter,
                    'Shovel_Queue_Lengths': None,
                    'Crusher_Queue_Lengths': None,
                    'Dump_Queue_Lengths': None,
                    'Changed_Shovel': shovel.name(),
                    'New_State': current_state,
                    'Changed_Truck': None,
                    'New_State_Truck': None,
                    'Trips_Per_Hour': None,
                    'Production_Volume_Per_Hour': None,
                    'Total_Production_Volume': None,
                    'Cost_Per_Ton': None,
                    'Total_Fuel_Consumption': None
                }
                
                if RL_sched:
                    metrics['Immediate_Reward'] = None
                    metrics['Shift_Reward'] = None
                
                self.update_csv(metrics)
                changed_equipment.append(('shovel', shovel.name(), current_state))
        
        for t in truck:
            current_state = 'breakdown' if t.phase == phase_broken_down else 'operational'
            prev_state = self.truck_states[t.name()]['state']
            
            if current_state != prev_state:
                self.truck_states[t.name()]['state'] = current_state
                self.truck_states[t.name()]['last_state_change'] = current_time
                
                metrics = {
                    'Timestamp': current_time,
                    'Shift_Number': self.shift_counter,
                    'Shovel_Queue_Lengths': None,
                    'Crusher_Queue_Lengths': None,
                    'Dump_Queue_Lengths': None,
                    'Changed_Shovel': None,
                    'New_State': None,
                    'Changed_Truck': t.name(),
                    'New_State_Truck': current_state,
                    'Trips_Per_Hour': None,
                    'Production_Volume_Per_Hour': None,
                    'Total_Production_Volume': None,
                    'Cost_Per_Ton': None,
                    'Total_Fuel_Consumption': None
                }
                
                if RL_sched:
                    metrics['Immediate_Reward'] = None
                    metrics['Shift_Reward'] = None
                
                self.update_csv(metrics)
                changed_equipment.append(('truck', t.name(), current_state))
        
        return changed_equipment

    def calculate_high_freq_metrics(self):
        """Calculate metrics that are updated at high frequency"""
        try:
            metrics = {
                'Shovel_Queue_Lengths': [m['queue'].length() for m in self.shovel_monitors.values()],
                'Crusher_Queue_Lengths': [m['queue'].length() for m in self.crusher_monitors.values()],
                'Dump_Queue_Lengths': [m['queue'].length() for m in self.dump_monitors.values()],
                'Changed_Shovel': None,
                'New_State': None,
                'Changed_Truck': None,
                'New_State_Truck': None
            }
            
            hourly_metrics = self.calculate_hourly_metrics()
            metrics.update(hourly_metrics)
            
            if RL_sched:
                metrics['Immediate_Reward'] = r_imm_d_pt if 'r_imm_d_pt' in globals() else None
                
            return metrics
        except Exception as e:
            print(f"Error calculating high frequency metrics: {e}")
            return {}

    def calculate_shift_metrics(self):
        """Calculate metrics that are tracked on a per-shift basis"""
        try:
            material_load_per_trip = cfg_samp.get_sampled_value('LO')
            
            total_production = total_crush_trips * material_load_per_trip
            total_cost = (
                cfg_samp.get_sampled_value('known_cost') + 
                cfg_samp.get_sampled_value('estimated_cost')
            )
            
            metrics = {
                'Total_Production_Volume': total_production,
                'Cost_Per_Ton': total_cost / total_production if total_production > 0 else 0,
                'Total_Fuel_Consumption': total_trips * cfg_samp.get_sampled_value('FO'),
                'Changed_Shovel': None,
                'New_State': None,
                'Changed_Truck': None,
                'New_State_Truck': None
            }
            
            hourly_metrics = self.calculate_hourly_metrics()
            metrics.update(hourly_metrics)
            
            if RL_sched:
                metrics['Shift_Reward'] = r_epi if 'r_epi' in globals() else None
                
            return metrics
        except Exception as e:
            print(f"Error calculating shift metrics: {e}")
            return {}

    def initialize_csv(self):
        """Initialize the CSV file with headers (no-op if already present/non-empty)."""
        with _KPI_WRITE_LOCK:
            need_header = (
                not os.path.exists(self.csv_filename)
                or os.path.getsize(self.csv_filename) == 0
            )
            if not need_header:
                self.file_exists = True
                return
            try:
                with open(self.csv_filename, 'w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=self.headers)
                    writer.writeheader()
                    file.flush()
                self.file_exists = True
            except IOError as e:
                print(f"Error initializing CSV: {e}")

    def update_csv(self, metrics):
        """Append one row to the shared KPI log. Stamps Episode/Scenario/Seed
        automatically so multi-episode runs can be aggregated by Episode
        instead of relying on one file per episode."""
        # always overwrite identity fields -- never trust the caller for these
        metrics['Episode'] = self.episode_idx
        metrics['Scenario'] = self.scenario_name
        metrics['Seed'] = self.seed
        try:
            with _KPI_WRITE_LOCK, open(self.csv_filename, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.headers, extrasaction='ignore')
                if os.path.getsize(self.csv_filename) == 0:
                    writer.writeheader()
                writer.writerow(metrics)
                file.flush()
        except IOError as e:
            print(f"Error writing to CSV: {e}")
            if not os.path.exists(self.csv_filename):
                self.initialize_csv()
                self.update_csv(metrics)

    def process(self):
        """Main process loop for the KPI Calculator"""
        while True:
            try:
                yield self.hold(1)
                current_time = env.now()
                
                # Check equipment states more frequently
                self.check_equipment_states()
                
                if current_time - self.last_high_freq_update >= self.HIGH_FREQ:
                    metrics = {
                        'Timestamp': current_time,
                        'Shift_Number': self.shift_counter,
                    }
                    
                    high_freq_metrics = self.calculate_high_freq_metrics()
                    if high_freq_metrics:
                        metrics.update(high_freq_metrics)
                        self.last_high_freq_update = current_time
                
                    if current_time - self.last_shift_update >= self.SHIFT_DURATION:
                        shift_metrics = self.calculate_shift_metrics()
                        if shift_metrics:
                            metrics.update(shift_metrics)
                            self.last_shift_update = current_time
                            self.shift_counter += 1
                    
                    self.update_csv(metrics)
                    
            except Exception as e:
                print(f"Error in KPI Calculator process: {e}")
                raise  # Re-raise the exception for debugging





@_with_runDes_lock
def runDes(fsim=True, flag_RL_sched=True, fdef_schdlr_choice = None, episode_seed=None, scenario_overrides=None, csv_path=None, scenario_name=None, play_seed=None, channel=None, episode_idx=0, kpi_log_path=None):

    global file_path
    global env, shovels, truck, dumps, crushers
    global shovel_idle_times, shovel_last_check, shovel_animations
    global RL_sched, def_schdlr_choice, all_trk_shv_dec, epsilon
    global rl_decision_index
    global cfg_samp
    global Num_trucks, Num_shovels, Num_crushers, Num_dumps
    global shift_dura, targ_pvol, load_per_trip, choice
    global total_trips, total_crush_trips, episode_shovel_choices, episode_queue_sum, episode_queue_count
    global broken_shovel_dispatch_count
    global truck_trip_counts, truck_phases, truck_last_trip_times
    global shovel_wait_time_tracker
    global trip_times, shovel_queues
    global r_imm_d_pt, terminated, pvol, sim_exit
    global _channel

    # For RL runs the GymEnv passes a StepChannel; for classical/rule-based
    # runs it stays None and every channel hook in this module is inert.
    _channel = channel

    # csv_path is kept only for backward-compatible call signatures and
    # KPI-output naming; no longer used for RL<->DES communication.
    file_path = csv_path

    # 1. Reset Truck class variables first.
    Truck.trucks_to_fail = None
    Truck.trucks_failed = 0

    previous_episode_count = getattr(cfg_samp, 'episode_count', 0)
    # 2. Always reinitialize ConfigSampler (not just for overrides).
    cfg_samp = ConfigSampler('config_extend_review.txt', 
                             scenario_overrides=scenario_overrides)

    # episode_count is a bolted-on attribute, not part of ConfigSampler's
    # class definition, so the fresh instance above starts without it
    # every episode; restore it here so it actually accumulates (the
    # increment at episode end is unchanged).
    cfg_samp.episode_count = previous_episode_count

    if scenario_overrides and _DES_VERBOSE:
        print(f"Scenario overrides active: {scenario_overrides}")

    # 3. Initialize fresh episodic parameters for this episode.
    cfg_samp.new_episode(episode_seed)

    # Seed the global `random`/`np.random` modules for this episode: they
    # drive the per-event coin flips (epsilon crusher/dump split, which
    # trucks/shovels pre-fail, unconstrained destination choice, breakdown
    # jitter/repair variation). cfg_samp.new_episode() only seeds
    # ConfigSampler's own RNG, not these -- seeding here makes runDes
    # self-sufficient for reproducibility regardless of entry point.
    if episode_seed is not None:
        random.seed(episode_seed)
        np.random.seed(episode_seed % (2**32))

    global current_scenario, current_seed
    current_scenario = scenario_name if scenario_name else "default"
    current_seed = episode_seed if episode_seed is not None else (play_seed if play_seed is not None else 0)

    # 4. Sample all parameters before using them.
    Num_trucks = int(cfg_samp.get_sampled_value('TR'))
    Num_shovels = int(cfg_samp.get_sampled_value('SH'))
    Num_crushers = int(cfg_samp.get_sampled_value('CR')) 
    Num_dumps = int(cfg_samp.get_sampled_value('DS'))
    shift_dura = cfg_samp.get_sampled_value('Sdur')
    targ_pvol = cfg_samp.get_sampled_value('PVol_targ')
    load_per_trip = cfg_samp.get_sampled_value('LO')
    choice = int(cfg_samp.get_sampled_value('scheduler_choice'))
    epsilon = cfg_samp.get_sampled_value('Eps')

    # 5. Reset all counters and state dictionaries.
    total_trips = 0
    total_crush_trips = 0
    broken_shovel_dispatch_count = 0
    episode_shovel_choices = {}
    episode_queue_sum = 0.0
    episode_queue_count = 0
    truck_trip_counts = {}
    truck_phases = {}
    truck_last_trip_times = {}
    sim_exit = False           
    r_imm_d_pt = None          
    terminated = False         
    pvol = 0 
    rl_decision_index = 0

    # 6. Create deques/trackers now that sizes are known.
    all_trk_shv_dec = deque(maxlen=Num_trucks * 2)
    shovel_wait_time_tracker = ShovelWaitTimeTracker(Num_shovels)

    # Rebound (not cleared in-place) so a concurrent eval-thread runDes
    # can't reach into this thread's RewardCalculator.trip_times/
    # shovel_queues and wipe its sliding-window data mid-episode -- each
    # episode gets a private deque pair no other thread holds a handle to.
    trip_times = deque(maxlen=k)
    shovel_queues = deque(maxlen=k)

    # 7. Print episode summary and continue.
    print_episode_summary()

    if flag_RL_sched:
        print("\n" + "="*70)
        print(f"    STARTING EPISODE {cfg_samp.episode_count}")
        print("="*70 + "\n")

    shovel_animations = {}
    env = sim.Environment(trace=False, time_unit='minutes')
    env.animate(False)
    env.width(1280)
    env.height(1024)
    RL_sched = flag_RL_sched  # Update the global flag
    def_schdlr_choice = fdef_schdlr_choice

    # Initialize print event
    print_event = PrintTripCountsEvent()
    print_event.activate()

    # Generate trucks as Component
    truck = [Truck() for _ in range(Num_trucks)]
    shovels = [sim.Resource(f"Shovel_{i}", capacity=1, preemptive=True) for i in range(Num_shovels)] # Create multiple shovels

    #Initialize dump site and crushers
    dumps = [sim.Resource(f'Dump{j}') for j in range(Num_dumps)]
    crushers = [sim.Resource(f'Crushers{j}') for j in range(Num_crushers)] 

    breakdown_manager = BreakdownManager()
    breakdown_manager.activate()

    # Create the KPI calculator component
    kpi_calculator = KPICalculator(
        scenario_name=current_scenario,
        seed=current_seed,
        csv_path=csv_path,
        episode_idx=episode_idx,
        kpi_log_path=kpi_log_path,
    )


    # Animation display setup------------------------------------------------------------------
    time_display = lambda: f"Time: {env.t():.2f}" 
    sim.AnimateImage("mine_site_1280_1024.png",x=5,y=5,width= 1020) #Wallpaper
    env.AnimateText(text=time_display, x=800, y=50, fontsize=20, textcolor = "white") #Display time
    env.background_color(("#eeffcc"))


    # Dictionary to track idle times for each shovel
    shovel_idle_times = {shovel.name(): 0 for shovel in shovels}
    shovel_last_check = {shovel.name(): 0 for shovel in shovels}

    # Initialize idle time tracking and print events
    track_idle_time_event = TrackIdleTime()
    track_idle_time_event.activate()

    print_idle_times_event = PrintShovelIdleTimes()
    print_idle_times_event.activate()


    # Initialize and activate the new event
    print_resource_status_event = PrintClaimersStatusEvent()
    print_resource_status_event.activate()

    # Shovel Section
    sim.AnimateText(text="< SHOVEL >", x=320, y=620, fontsize=20, textcolor="yellow")
    for shovel in shovels:
        shv_txt = str(shovel.base_name())
        sv_id = int(shv_txt.split('_')[-1])
        xs_val = xs_init + sv_id * 50
        shovel_animations[shovel] = sim.AnimateImage("shovel_active.png", x=(xs_init + sv_id * 40) - 120, y=570, width=40, env=env)
        shovel.claimers().animate(x=xs_val, y=660, title=".", direction="e")
        shovel.requesters().animate(x=xs_val, y=580, title=".", direction="s")

    sim.AnimateText(text="Loading", x=340, y=550, fontsize=15, textcolor="white")
    sim.AnimateText(text="Waiting", x=340, y=490, fontsize=15, textcolor="white")

    # Add a header for breakdown status display
    sim.AnimateText(text="< BREAKDOWN STATUS >", x=640, y=900, fontsize=20, textcolor="yellow")

    # Dump Section
    sim.AnimateText(text="< DUMPS>", x=50, y=780, fontsize=20, textcolor="yellow")
    for dump in dumps:
        dp_txt = dump.base_name()  # Assuming .name() method returns the name of the dump
        match = re.search(r'\d+$', dp_txt)  # This regex finds one or more digits at the end of the string
        if match:
            dmp_id = int(match.group())  # Convert the found digits to an integer
            xd_val = xd_init + dmp_id * 50
            dump.claimers().animate(x=xd_val, y=930, title=".", direction="e")
            dump.requesters().animate(x=xd_val, y=870, title=".", direction="s")
        else:
            print("Invalid dump name format:", dp_txt)

    sim.AnimateText(text="Dumping", x=50, y=765, fontsize=15, textcolor="white")
    sim.AnimateText(text="Waiting", x=50, y=713, fontsize=15, textcolor="white")


    # Crusher Section
    xc_init = 1100  # Initial x-coordinate for crushers
    yc_init = 700  # Initial y-coordinate for crushers
    # Add a label for Crushers
    sim.AnimateText(text="< CRUSHERS >", x=800, y=660, fontsize=20, textcolor="yellow")

    for crusher in crushers:
        cp_txt = crusher.base_name()  # Assuming .name() method returns the name of the crusher
        match = re.search(r'\d+$', cp_txt)  # This regex finds one or more digits at the end of the string
        if match:
            crusher_id = int(match.group())  # Convert the found digits to an integer
            xc_val = (xc_init-90) + crusher_id * 50
            yc_val = yc_init  # y-coordinate remains the same for all crushers
            crusher.claimers().animate(x=xc_val, y=yc_val+70, title=".", direction="e")
            crusher.requesters().animate(x=xc_val, y=yc_val, title=".", direction="s")
        else:
            print("Invalid crusher name format:", cp_txt)

    # Add labels for Crusher actions
    sim.AnimateText(text="Crushing", x=xc_init-300, y=yc_init - 60, fontsize=15, textcolor="white")
    sim.AnimateText(text="Waiting", x=xc_init-300, y=yc_init - 120, fontsize=15, textcolor="white")


    print('-------**----')
    print(f"\nStarting simulation. Duration set to: {shift_dura} time units")
    env.run(till=shift_dura)
    print(f"\nSimulation ended at: {env.now()} time units")
    sim_exit = True
    
  
    pvol = total_crush_trips * load_per_trip
    wvol = max(total_trips - total_crush_trips, 0) * load_per_trip

    if RL_sched  == True:
        prod_ratio = min(1.0, pvol/targ_pvol)

        # Informational state for the agent / play-mode logging only
        # (Fleet_Diversity, DivScore in the terminal info); not part of
        # the reward.
        final_diversity_score = diversity_score()


        EPI_PROD_W = 30.0
        b1, b2 = 10.0, 4.0
        th1, th2 = 0.80, 0.65
        phi1, phi2 = 0.65, 0.50  # light final-diversity gate (kept modest)

        if prod_ratio >= th1 and final_diversity_score > phi1:
            perf_bonus = b1
        elif prod_ratio >= th2 and final_diversity_score > phi2:
            perf_bonus = b2
        else:
            perf_bonus = 0.0

        r_epi = EPI_PROD_W * prod_ratio + perf_bonus


        _total_choices = sum(episode_shovel_choices.values())
        if _total_choices > 0 and Num_shovels and Num_shovels > 0:
            _counts = list(episode_shovel_choices.values())
            max_shovel_share = round(max(_counts) / _total_choices, 5)
            _probs = [c / _total_choices for c in _counts]
            _entropy = -sum(p * np.log2(p + 1e-12) for p in _probs)
            _max_entropy = np.log2(Num_shovels)
            shovel_sel_entropy = round(_entropy / _max_entropy, 5) if _max_entropy > 0 else 0.0
            unused_shovels = int(Num_shovels - len(episode_shovel_choices))
        else:
            max_shovel_share = ""
            shovel_sel_entropy = ""
            unused_shovels = ""

        print('\n **** ')
        print("pvol: {}, r_epi: {}, prod_ratio: {}, perf_bonus: {}, total_trips: {}, "
              "max_shovel_share: {}, shovel_sel_entropy: {}, targ_pvol: {}".format(
                  pvol, r_epi, prod_ratio, perf_bonus, total_trips,
                  max_shovel_share, shovel_sel_entropy, targ_pvol))
        if broken_shovel_dispatch_count > 0:
            print(f"WARNING: {broken_shovel_dispatch_count} dispatch(es) to a broken "
                  f"shovel this episode -- action masking may not be active.")

        # Append this episode's metrics to a live-trackable CSV (separate
        # file per csv_path/scenario name so train/test/eval runs don't mix).
        # See episode_metrics_logger.py.
        try:
            metrics_path = (os.path.splitext(str(file_path))[0] + "_episode_metrics.csv"
                             if file_path else "episode_metrics.csv")
            log_episode_metrics(
                csv_path=metrics_path,
                episode=cfg_samp.episode_count if hasattr(cfg_samp, 'episode_count') else None,
                pvol=pvol, targ_pvol=targ_pvol, prod_ratio=prod_ratio, r_epi=r_epi,
                total_trips=total_trips, total_crush_trips=total_crush_trips,
                broken_shovel_dispatch_count=broken_shovel_dispatch_count,
                scenario_name=scenario_name,
                max_shovel_share=max_shovel_share,
                shovel_sel_entropy=shovel_sel_entropy,
                unused_shovels=unused_shovels,
            )
        except Exception as e:
            # Never let logging failures interrupt the simulation.
            print(f"WARNING: could not write episode metrics log: {e}")

        # Get all truck data
        all_truck_data = print_trip_counts(ppflag=0)
    
        # Filter to just one truck (pick any - doesn't matter which)
        if all_truck_data:
            # Use the first available truck
            first_truck_key = list(all_truck_data.keys())[0]
            terminal_truck_data = {first_truck_key: all_truck_data[first_truck_key]}
        else:
            # Fallback (shouldn't happen - there should always be trucks)
            terminal_truck_data = {'Truck.1': {'trip_count': 0, 'phase': '000'}}

        all_trips = list(truck_trip_counts.values())
        fleet_summary = {
            'avg_trips': sum(all_trips) / len(all_trips) if all_trips else 0,
            'recent_decisions': all_trk_shv_dec,
            'diversity_score': final_diversity_score
        }
    
        # Now pass only 1 truck (same structure as during episode)
        observ = create_observation(Num_shovels, Num_trucks,
                                   print_resource_status(pflag=0)['Shovels'],
                                   terminal_truck_data, fleet_summary)  # ← Now passes 1 truck
    
        # Terminal info dict. PVOL/DivScore kept for backward compatibility;
        # prod_ratio and mean_queue let an external eval callback read
        # operational quality straight off info at episode end. mean_queue
        # is the average queue length at the chosen shovel across this
        # episode's RL decisions -- a direct measure of dispatch quality.
        mean_queue_this_ep = (episode_queue_sum / episode_queue_count
                               if episode_queue_count > 0 else 0.0)
        info = {
            'PVOL': pvol,
            'DivScore': final_diversity_score,
            'prod_ratio': prod_ratio,
            'mean_queue': mean_queue_this_ep,
        }
        final_step_update(observ, r_epi, info)
        print("Current observ: "+str(observ))

        # Log a synchronicity summary, then return so the DES worker
        # thread ends cleanly (as a thread, not a process, we must return
        # rather than sys.exit()).
        if _channel is not None:
            _channel.episode_summary()

        # Release the salabim env (memory hygiene): `env` is a module
        # global reassigned at the top of the next runDes() call, but for
        # this daemon thread's lifetime it still pins the entire previous
        # episode's object graph (components, events, trajectories,
        # animation handles). Setting it to None here releases that graph
        # immediately rather than waiting for reassignment, which matters
        # over many back-to-back episodes in a long training run. Safe:
        # anything that still needs these has finished by the time we
        # return.
        env = None
        shovels = None
        truck = None
        dumps = None
        crushers = None
        shovel_animations = None
        return
    elif RL_sched  == False:
        return pvol