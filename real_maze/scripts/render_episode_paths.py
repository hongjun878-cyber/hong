import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.maze.io import load_json


def main():
    parser = argparse.ArgumentParser(description="Render episode trajectories on maze images")
    parser.add_argument("--index", required=True, help="Maze index JSON used for the experiment")
    parser.add_argument("--results_dir", required=True, help="Experiment output directory with summary.json")
    parser.add_argument("--out_dir", default=None, help="Output image directory")
    parser.add_argument("--leaders", nargs="*", default=None, help="Optional leader names to render")
    parser.add_argument("--only_failures", action="store_true", help="Render failed episodes only")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of episodes")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "path_images")
    os.makedirs(out_dir, exist_ok=True)

    index = _load_index(args.index)
    with open(os.path.join(args.results_dir, "summary.json")) as f:
        results = json.load(f)

    rendered = 0
    for result in results:
        exp = result.get("experiment", {})
        maze_name = exp.get("maze_name") or result.get("maze_name")
        leader = exp.get("leader_persona_name") or result.get("leader_persona_name", "unknown")
        n_agents = exp.get("n_agents") or result.get("n_agents")
        mode = exp.get("decision_mode") or result.get("decision_mode")

        if args.leaders and leader not in args.leaders:
            continue
        if args.only_failures and result.get("success"):
            continue
        if maze_name not in index:
            print(f"Skipping {maze_name}: not found in index")
            continue

        log_path = _find_log_path(args.results_dir, maze_name, n_agents, mode, leader)
        if log_path is None:
            print(f"Skipping {maze_name}/{leader}: decision log not found")
            continue

        maze = load_json(index[maze_name]["json_path"])
        path = _read_path(log_path, maze["start"])
        out_path = os.path.join(out_dir, f"{maze_name}_{leader}_path.png")
        _render_path_image(maze, path, result, leader, out_path)
        print(f"Wrote {out_path}")

        rendered += 1
        if args.limit is not None and rendered >= args.limit:
            break

    print(f"Rendered {rendered} path images to {out_dir}")


def _load_index(path):
    with open(path) as f:
        entries = json.load(f)
    return {entry["name"]: entry for entry in entries}


def _find_log_path(results_dir, maze_name, n_agents, mode, leader):
    logs_dir = os.path.join(results_dir, "decision_logs")
    candidates = [
        f"{maze_name}_n{n_agents}_leader_majority_{leader}.jsonl",
        f"{maze_name}_n{n_agents}_leader_vice_weighted_majority_leader-{leader}.jsonl",
        f"{maze_name}_n{n_agents}_{mode}_leader-{leader}.jsonl",
        f"{maze_name}_n{n_agents}_{mode}_leader_{leader}.jsonl",
    ]
    for name in candidates:
        path = os.path.join(logs_dir, name)
        if os.path.exists(path):
            return path
    return None


def _read_path(log_path, start):
    path = [list(start)]
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            pos = entry.get("position")
            if pos is not None:
                path.append([int(pos[0]), int(pos[1])])
    return path


def _render_path_image(maze, path, result, leader, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    walls = maze["Walls"]
    sizex, sizey = maze["sizex"], maze["sizey"]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.set_aspect("equal")
    ax.axis("off")

    for y in range(sizey):
        for x in range(sizex):
            if walls[y][x][1] == 1:
                ax.plot([x, x + 1], [y + 1, y + 1], color="black", lw=1.4)
            if walls[y][x][3] == 1:
                ax.plot([x, x + 1], [y, y], color="black", lw=1.4)
            if walls[y][x][0] == 1:
                ax.plot([x, x], [y, y + 1], color="black", lw=1.4)
            if walls[y][x][2] == 1:
                ax.plot([x + 1, x + 1], [y, y + 1], color="black", lw=1.4)

    if path:
        xs = [p[0] + 0.5 for p in path]
        ys = [p[1] + 0.5 for p in path]
        ax.plot(xs, ys, color="#1f77b4", lw=2.2, alpha=0.85, label="trajectory")
        ax.scatter(xs, ys, s=16, color="#1f77b4", alpha=0.45)

    sx, sy = maze["start"]
    gx, gy = maze["goal"]
    fx, fy = path[-1] if path else maze["start"]
    ax.scatter([sx + 0.5], [sy + 0.5], s=110, color="#2ca02c", label="start", zorder=5)
    ax.scatter([gx + 0.5], [gy + 0.5], s=150, color="#d62728", marker="*", label="goal", zorder=5)
    ax.scatter([fx + 0.5], [fy + 0.5], s=120, color="#ff7f0e", marker="X", label="final", zorder=6)

    status = "SUCCESS" if result.get("success") else "FAIL"
    title = (
        f"{maze['sizex']}x{maze['sizey']} seed {maze['seed']} | {leader} | {status}\n"
        f"steps={result.get('steps_taken')} deadlocks={result.get('deadlock_count')} "
        f"parse_failures={result.get('parse_failures', 0)}"
    )
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
