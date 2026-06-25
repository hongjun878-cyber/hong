import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


LEADER_ORDER = ["goal_seeker", "explorer", "backtracker", "balanced_strategist"]
LEADER_LABELS = {
    "goal_seeker": "Goal-Seeker",
    "explorer": "Explorer",
    "backtracker": "Backtracker",
    "balanced_strategist": "Balanced",
}


def main():
    parser = argparse.ArgumentParser(description="Plot leader persona result CSVs")
    parser.add_argument("--csv_5x5", required=True)
    parser.add_argument("--csv_7x7", required=True)
    parser.add_argument("--out_dir", default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = {
        "5x5": _read_summary(args.csv_5x5),
        "7x7": _read_summary(args.csv_7x7),
    }

    _plot_metric(
        data,
        metric="success_rate",
        ylabel="Success rate",
        title="Leader Persona Success Rate",
        out_path=os.path.join(args.out_dir, "leader_success_rate.png"),
        ylim=(0, 1),
    )
    _plot_metric(
        data,
        metric="avg_deadlocks",
        ylabel="Average deadlocks",
        title="Average Deadlocks by Leader Persona",
        out_path=os.path.join(args.out_dir, "leader_avg_deadlocks.png"),
    )
    _plot_metric(
        data,
        metric="avg_steps",
        ylabel="Average steps",
        title="Average Steps by Leader Persona",
        out_path=os.path.join(args.out_dir, "leader_avg_steps.png"),
    )

    print(f"Wrote plots to {args.out_dir}")


def _read_summary(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            leader = row["leader_persona"]
            rows[leader] = row
    return rows


def _plot_metric(data, metric, ylabel, title, out_path, ylim=None):
    x = range(len(LEADER_ORDER))
    width = 0.36
    offsets = {"5x5": -width / 2, "7x7": width / 2}
    colors = {"5x5": "#4C78A8", "7x7": "#F58518"}

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for size_label, rows in data.items():
        values = [_to_float(rows.get(leader, {}).get(metric, 0)) for leader in LEADER_ORDER]
        xpos = [i + offsets[size_label] for i in x]
        bars = ax.bar(xpos, values, width=width, label=size_label, color=colors[size_label])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}" if metric == "success_rate" else f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LEADER_LABELS[l] for l in LEADER_ORDER], rotation=15, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
