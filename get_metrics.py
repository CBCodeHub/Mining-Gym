"""
Aggregate a single multi-episode KPI log  into plotting-ready summaries.

Inputs
------
A single CSV containing all episodes of one play/eval run, with at least
these columns (the patched KPICalculator writes all of them):
    Episode, Timestamp, Shovel_Queue_Lengths,
    Trips_Per_Hour, Total_Production_Volume

Outputs
-------
1. <out_prefix>_hourly.csv    — same shape your combo_plot.py already
                                 consumes (Hour, Mean_*, Std_*).
                                 Variance is now across episodes within
                                 the same file rather than across files.
2. <out_prefix>_per_episode.csv — one row per episode with final/peak
                                  values. Use for box plots, violin
                                  plots, scatter, or composite scoring.
3. <out_prefix>_composite.csv   — single-row summary across the whole
                                  run (overall mean ± std of the
                                  per-episode metrics) — handy for
                                  comparing checkpoints at a glance.

Usage
-----
    python get_metrics.py kpi_log_test_shared.csv --out-prefix rl_ckpt2k

"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Loading & parsing
# ----------------------------------------------------------------------

REQUIRED_COLUMNS = {
    "Episode",
    "Timestamp",
    "Shovel_Queue_Lengths",
    "Trips_Per_Hour",
    "Total_Production_Volume",
}


def _parse_queue_list(value) -> int:
    """Sum a queue-length list stored as a string like '[6, 6, 5]'."""
    if isinstance(value, list):
        return sum(value)
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return sum(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return 0
    return 0


def load_kpi_log(path: str | Path) -> pd.DataFrame:
    """Read the KPI log, validate schema, derive Hour + total queue length."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"KPI log not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"KPI log {path.name} is missing required columns: {sorted(missing)}. "
            f"Was it produced by the patched KPICalculator?"
        )

    n_rows_in = len(df)
    raw_minutes = pd.to_numeric(df["Timestamp"], errors="coerce")
    max_minute = raw_minutes.max()
    if pd.notna(max_minute) and max_minute >= 1440:
        logger.warning(
            "%s has Timestamp values >= 1440 minutes (24h) within a single "
            "episode; Hour buckets wrap at 24 and will alias across days "
            "(e.g. minute 1440 collides with minute 0). Hourly aggregates "
            "may be misleading for this file.",
            path.name,
        )
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"], unit="m", origin="unix", errors="coerce"
    )
    df = df.dropna(subset=["Timestamp"]).copy()
    n_dropped = n_rows_in - len(df)
    if n_dropped:
        logger.warning(
            "%d row(s) in %s had an unparseable Timestamp and were dropped",
            n_dropped, path.name,
        )

    df["Hour"] = df["Timestamp"].dt.hour
    df["Total_Shovel_Queue_Length"] = df["Shovel_Queue_Lengths"].apply(
        _parse_queue_list
    )

    n_ep = df["Episode"].nunique()
    logger.info("Loaded %d rows across %d episodes from %s", len(df), n_ep, path.name)
    return df


# ----------------------------------------------------------------------
# Aggregations
# ----------------------------------------------------------------------

def compute_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hourly aggregate across episodes — drop-in for combo_plot.py.

    For each (Episode, Hour) we take:
        Trips_Per_Hour          -> mean of the tick samples
        Total_Shovel_Queue      -> max of the tick samples (peak congestion)
        Total_Production_Volume -> last tick (cumulative; last is the right end)

    Then we take mean / std across episodes for each Hour.
    """
    per_ep_hour = (
        df.groupby(["Episode", "Hour"])
        .agg(
            trips_per_hour=("Trips_Per_Hour", "mean"),
            max_shovel_queue=("Total_Shovel_Queue_Length", "max"),
            total_production_volume=("Total_Production_Volume", "last"),
        )
        .reset_index()
    )

    summary = (
        per_ep_hour.groupby("Hour")
        .agg(
            Mean_Trips_Per_Hour=("trips_per_hour", "mean"),
            Std_Trips_Per_Hour=("trips_per_hour", "std"),
            Mean_Max_Shovel_Queue_Length=("max_shovel_queue", "mean"),
            Std_Max_Shovel_Queue_Length=("max_shovel_queue", "std"),
            Mean_Total_Production_Volume=("total_production_volume", "mean"),
            Std_Total_Production_Volume=("total_production_volume", "std"),
        )
        .reset_index()
        .fillna(0.0)  # std is NaN when n=1; treat as 0 for plotting
    )
    return summary


def compute_per_episode(df: pd.DataFrame) -> pd.DataFrame:
    """One row per episode — useful for box/violin/scatter plots."""
    scenario_col = "Scenario" if "Scenario" in df.columns else None
    seed_col = "Seed" if "Seed" in df.columns else None

    agg = (
        df.groupby("Episode")
        .agg(
            Final_Production_Volume=("Total_Production_Volume", "last"),
            Mean_Trips_Per_Hour=("Trips_Per_Hour", "mean"),
            Peak_Shovel_Queue=("Total_Shovel_Queue_Length", "max"),
            Mean_Shovel_Queue=("Total_Shovel_Queue_Length", "mean"),
            N_Ticks=("Timestamp", "count"),
        )
        .reset_index()
    )

    # carry provenance columns (constant per episode) through if present
    for col in (scenario_col, seed_col):
        if col is None:
            continue
        first_per_ep = df.groupby("Episode")[col].first().reset_index()
        agg = agg.merge(first_per_ep, on="Episode")

    return agg


def compute_composite(per_episode: pd.DataFrame) -> pd.DataFrame:
    """Single-row summary across episodes — for quick checkpoint comparison."""
    # Exclude identifiers/provenance columns carried through by
    # compute_per_episode (Episode, and Seed when present) — these are
    # labels, not metrics, and averaging them is meaningless.
    non_metric_cols = ["Episode", "Seed"]
    numeric = per_episode.select_dtypes(include="number").drop(
        columns=non_metric_cols, errors="ignore"
    )
    composite = {}
    for col in numeric.columns:
        composite[f"{col}_mean"] = numeric[col].mean()
        composite[f"{col}_std"] = numeric[col].std()
    composite["N_Episodes"] = len(per_episode)
    return pd.DataFrame([composite])


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate a multi-episode KPI log into plotting-ready CSVs."
    )
    parser.add_argument("kpi_log", help="Path to the multi-episode KPI log CSV.")
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Prefix for output files. Defaults to the input stem.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable info-level logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    in_path = Path(args.kpi_log)
    prefix = Path(args.out_prefix) if args.out_prefix else in_path.with_suffix("")

    try:
        df = load_kpi_log(in_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    hourly = compute_hourly(df)
    per_episode = compute_per_episode(df)
    composite = compute_composite(per_episode)

    hourly_path = prefix.with_name(prefix.name + "_hourly.csv")
    per_ep_path = prefix.with_name(prefix.name + "_per_episode.csv")
    composite_path = prefix.with_name(prefix.name + "_composite.csv")

    hourly.to_csv(hourly_path, index=False)
    per_episode.to_csv(per_ep_path, index=False)
    composite.to_csv(composite_path, index=False)

    print(f"Wrote {hourly_path}")
    print(f"Wrote {per_ep_path}")
    print(f"Wrote {composite_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
