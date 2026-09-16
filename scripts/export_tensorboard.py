#!/usr/bin/env python3
"""
Export TensorBoard training logs to CSV for empirical figure generation.

Usage:
    python scripts/export_tensorboard.py \\
        --logdir runs/auto_tgnn_ppo_run1 \\
        --out results/training_curves.csv

    # Multiple runs (TGNN and Static-GNN comparison):
    python scripts/export_tensorboard.py \\
        --logdir runs/auto_tgnn_ppo_run1 \\
        --out results/training_curves_tgnn.csv \\
        --model-label "TGNN-NCO (Auto)"

    python scripts/export_tensorboard.py \\
        --logdir runs/auto_static_ppo_run1 \\
        --out results/training_curves_static.csv \\
        --model-label "Static-GNN (Auto)"
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_TAGS = [
    "charts/mean_reward",
    "charts/feasibility_rate",
    "charts/mean_migration_penalty",
    "losses/policy_loss",
    "losses/value_loss",
    "losses/entropy",
    "charts/mean_ep_len",
]


def main():
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar events to CSV for plot generation"
    )
    parser.add_argument("--logdir", required=True, help="Path to TensorBoard log directory")
    parser.add_argument("--out", default="results/training_curves.csv", help="Output CSV path")
    parser.add_argument(
        "--tags", nargs="*", default=DEFAULT_TAGS,
        help="TensorBoard scalar tag names to export (default: all training metrics)",
    )
    parser.add_argument(
        "--model-label", default=None,
        help="Optional model label added as a column for multi-run comparison",
    )
    args = parser.parse_args()

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("ERROR: tensorboard package not found. Install with: pip install tensorboard")
        sys.exit(1)

    if not os.path.exists(args.logdir):
        print(f"ERROR: Log directory '{args.logdir}' not found.")
        sys.exit(1)

    print(f"Loading TensorBoard events from '{args.logdir}'...")
    ea = EventAccumulator(args.logdir)
    ea.Reload()

    available_tags = ea.Tags().get("scalars", [])
    print(f"  Available scalar tags: {available_tags}")

    rows = []
    for tag in args.tags:
        if tag not in available_tags:
            print(f"  WARNING: tag '{tag}' not found — skipping")
            continue
        for ev in ea.Scalars(tag):
            row = {
                "tag":       tag,
                "step":      ev.step,
                "wall_time": ev.wall_time,
                "value":     ev.value,
            }
            if args.model_label is not None:
                row["model"] = args.model_label
            rows.append(row)

    if not rows:
        print("WARNING: No scalar events found. Check --logdir and --tags arguments.")
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fieldnames = ["tag", "step", "wall_time", "value"]
    if args.model_label is not None:
        fieldnames.append("model")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} scalar events → '{args.out}'")
    print(f"Tags exported: {list({r['tag'] for r in rows})}")


if __name__ == "__main__":
    main()
