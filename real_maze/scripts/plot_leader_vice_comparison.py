import argparse
import csv
import os

import matplotlib.pyplot as plt


LEADER_ORDER = ["goal_seeker", "explorer", "backtracker", "balanced_strategist"]
LEADER_LABELS = {
    "goal_seeker": "Goal-Seeker",
    "explorer": "Explorer",
    "backtracker": "Backtracker",
    "balanced_strategist": "Balanced",
}


def main():
    parser = argparse.ArgumentParser(description="Plot leader-only vs leader+vice summary CSVs")
    parser.add_argument("--leader_only_csv", required=True)
    parser.add_argument("--leader_vice_csv", required=True)
    parser.add_argument("--out_dir", default="results/leader_vice_plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = {
        "Leader only": _read_summary(args.leader_only_csv),
        "Leader + vice": _read_summary(args.leader_vice_csv),
    }

    _plot_metric(
        data,
        metric="success_rate",
        ylabel="Success rate",
        title="7x7 Success Rate: Leader Only vs Leader + Vice",
        out_path=os.path.join(args.out_dir, "leader_vs_vice_success_rate.png"),
        ylim=(0, 1),
        value_fmt="{:.2f}",
    )
    _plot_metric(
        data,
        metric="avg_steps",
        ylabel="Average steps",
        title="7x7 Average Steps: Leader Only vs Leader + Vice",
        out_path=os.path.join(args.out_dir, "leader_vs_vice_avg_steps.png"),
        value_fmt="{:.1f}",
    )
    _plot_metric(
        data,
        metric="avg_deadlocks",
        ylabel="Average deadlocks",
        title="7x7 Average Deadlocks: Leader Only vs Leader + Vice",
        out_path=os.path.join(args.out_dir, "leader_vs_vice_avg_deadlocks.png"),
        value_fmt="{:.1f}",
    )

    print(f"Wrote comparison plots to {args.out_dir}")


def _read_summary(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["leader_persona"]] = row
    return rows


def _plot_metric(data, metric, ylabel, title, out_path, ylim=None, value_fmt="{:.1f}"):
    x = range(len(LEADER_ORDER))
    width = 0.36
    offsets = {"Leader only": -width / 2, "Leader + vice": width / 2}
    colors = {"Leader only": "#4C78A8", "Leader + vice": "#54A24B"}

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for label, rows in data.items():
        values = [_to_float(rows.get(leader, {}).get(metric, 0)) for leader in LEADER_ORDER]
        xpos = [i + offsets[label] for i in x]
        bars = ax.bar(xpos, values, width=width, label=label, color=colors[label])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_fmt.format(value),
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
