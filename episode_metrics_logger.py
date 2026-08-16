"""
Lightweight, append-only episode-metrics logger.
"""
import csv
import os
import threading
from datetime import datetime
from typing import Optional

_write_lock = threading.Lock()

FIELDNAMES = [
    "timestamp",
    "episode",
    "scenario_name",
    "pvol",
    "targ_pvol",
    "prod_ratio",
    "r_epi",
    "total_trips",
    "total_crush_trips",
    "broken_shovel_dispatch_count",
    # Shovel-hugging monitor (per-episode summary of RL shovel choices):
    "max_shovel_share",    # share of dispatches to the single most-used shovel
    "shovel_sel_entropy",  # normalized Shannon entropy (1.0 == even, 0 == hugging)
    "unused_shovels",      # operational shovels never chosen
]

DISPATCH_FIELDNAMES = [
    "timestamp",
    "episode",
    "decision_index",
    "chosen_shovel_id",
    "num_operational",
    "chosen_queue_len",
    "min_queue_len",
    "max_queue_len",
    "spread",
    "degenerate",
    "dispatch_term",
    # Hugging diagnostics:
    "streak_ratio",      # raw consecutive-same-shovel ratio, pre reward-weight
    "streak_term",        # weighted contribution to r_imm_d
    "production_reward",  # this decision's production-progress contribution
]

_dispatch_call_count = 0
_dispatch_lock = threading.Lock()


def log_episode_metrics(
    csv_path: str,
    episode,
    pvol: float,
    targ_pvol: float,
    prod_ratio: float,
    r_epi: float,
    total_trips: int,
    total_crush_trips: int,
    broken_shovel_dispatch_count: int = 0,
    scenario_name: Optional[str] = None,
    max_shovel_share=None,
    shovel_sel_entropy=None,
    unused_shovels=None,
) -> None:
    """
    Append one row of episode-level metrics to `csv_path`. Creates the
    file with a header on first use; safe to call every episode (single
    small append, not a full-file rewrite).
    """
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "episode": episode,
        "scenario_name": scenario_name or "",
        "pvol": pvol,
        "targ_pvol": targ_pvol,
        "prod_ratio": round(prod_ratio, 5) if prod_ratio is not None else "",
        "r_epi": round(r_epi, 5) if r_epi is not None else "",
        "total_trips": total_trips,
        "total_crush_trips": total_crush_trips,
        "broken_shovel_dispatch_count": broken_shovel_dispatch_count,
        "max_shovel_share": max_shovel_share if max_shovel_share is not None else "",
        "shovel_sel_entropy": shovel_sel_entropy if shovel_sel_entropy is not None else "",
        "unused_shovels": unused_shovels if unused_shovels is not None else "",
    }

    with _write_lock:
        file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def log_dispatch_decision(
    csv_path: str,
    episode,
    decision_index: int,
    chosen_queue_len,
    all_queue_lens,
    spread,
    degenerate,
    dispatch_term: float,
    chosen_shovel_id=None,
    num_operational=None,
    streak_ratio=None,
    streak_term=None,
    production_reward=None,
    sample_rate: int = 5,
) -> None:
    """
    Append one row of per-decision dispatch diagnostics, sampled at
    1/sample_rate. chosen_queue_len/all_queue_lens may be None (a
    truck's first RL-mediated decision has no prior dispatch to score)
    -- those rows are skipped rather than logged with placeholders.
    """
    global _dispatch_call_count
    with _dispatch_lock:
        _dispatch_call_count += 1
        if _dispatch_call_count % sample_rate != 0:
            return

    if chosen_queue_len is None or all_queue_lens is None:
        return

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "episode": episode,
        "decision_index": decision_index,
        "chosen_shovel_id": chosen_shovel_id if chosen_shovel_id is not None else "",
        "num_operational": num_operational if num_operational is not None else "",
        "chosen_queue_len": chosen_queue_len,
        "min_queue_len": min(all_queue_lens) if all_queue_lens else "",
        "max_queue_len": max(all_queue_lens) if all_queue_lens else "",
        "spread": round(spread, 5) if spread is not None else "",
        "degenerate": bool(degenerate) if degenerate is not None else "",
        "dispatch_term": round(dispatch_term, 5) if dispatch_term is not None else "",
        "streak_ratio": round(streak_ratio, 5) if streak_ratio is not None else "",
        "streak_term": round(streak_term, 5) if streak_term is not None else "",
        "production_reward": round(production_reward, 5) if production_reward is not None else "",
    }

    with _write_lock:
        file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DISPATCH_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
