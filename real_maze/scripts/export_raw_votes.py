import argparse
import csv
import glob
import json
import os


ACTION_NAMES = {"0": "left", "1": "up", "2": "right", "3": "down"}
ACTION_KEYS = ["left", "up", "right", "down"]


def main():
    parser = argparse.ArgumentParser(description="Export all raw votes from decision logs to CSV")
    parser.add_argument("--results_dir", required=True, help="Experiment directory containing summary.json and decision_logs/")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--leaders", nargs="*", default=None, help="Optional leader names to include")
    parser.add_argument("--only_failures", action="store_true", help="Export failed episodes only")
    args = parser.parse_args()

    with open(os.path.join(args.results_dir, "summary.json")) as f:
        summary = json.load(f)

    episode_meta = {}
    for result in summary:
        exp = result.get("experiment", {})
        maze_name = exp.get("maze_name") or result.get("maze_name")
        leader = exp.get("leader_persona_name") or result.get("leader_persona_name")
        n_agents = exp.get("n_agents") or result.get("n_agents")
        mode = exp.get("decision_mode") or result.get("decision_mode")
        key = (maze_name, str(n_agents), mode, leader)
        episode_meta[key] = {
            "maze_name": maze_name,
            "leader_persona": leader,
            "n_agents": n_agents,
            "decision_mode": mode,
            "success": result.get("success"),
            "steps_taken": result.get("steps_taken"),
            "deadlock_count": result.get("deadlock_count"),
            "parse_failures_total": result.get("parse_failures", 0),
        }

    rows = []
    for log_path in sorted(glob.glob(os.path.join(args.results_dir, "decision_logs", "*.jsonl"))):
        meta = _match_meta(log_path, episode_meta)
        if meta is None:
            continue
        if args.leaders and meta["leader_persona"] not in args.leaders:
            continue
        if args.only_failures and meta["success"]:
            continue

        with open(log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                step = json.loads(line)
                raw_votes = _vote_counts(step.get("raw_votes", {}))
                weighted_votes = _vote_counts(step.get("weighted_votes", {}))
                chosen = _action_name(step.get("action_chosen"))
                leader_action = _action_name(step.get("leader_action"))
                row = {
                    **meta,
                    "log_file": os.path.basename(log_path),
                    "step": step.get("step"),
                    "position": _format_position(step.get("position")),
                    "available": _format_actions(step.get("available", [])),
                    "is_junction": step.get("is_junction"),
                    "leader_action": leader_action,
                    "chosen_action": chosen,
                    "raw_left": raw_votes["left"],
                    "raw_up": raw_votes["up"],
                    "raw_right": raw_votes["right"],
                    "raw_down": raw_votes["down"],
                    "weighted_left": weighted_votes["left"],
                    "weighted_up": weighted_votes["up"],
                    "weighted_right": weighted_votes["right"],
                    "weighted_down": weighted_votes["down"],
                    "parse_failures_step": step.get("parse_failures", 0),
                    "used_fallback": step.get("used_fallback", False),
                    "deadlock_count_step": step.get("deadlock_count"),
                }
                rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "maze_name",
        "leader_persona",
        "success",
        "step",
        "position",
        "available",
        "is_junction",
        "leader_action",
        "chosen_action",
        "raw_left",
        "raw_up",
        "raw_right",
        "raw_down",
        "weighted_left",
        "weighted_up",
        "weighted_right",
        "weighted_down",
        "parse_failures_step",
        "used_fallback",
        "deadlock_count_step",
        "steps_taken",
        "deadlock_count",
        "parse_failures_total",
        "n_agents",
        "decision_mode",
        "log_file",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} vote rows to {args.out}")


def _match_meta(log_path, episode_meta):
    name = os.path.basename(log_path)
    for key, meta in episode_meta.items():
        maze_name, n_agents, mode, leader = key
        patterns = [
            f"{maze_name}_n{n_agents}_leader_majority_{leader}.jsonl",
            f"{maze_name}_n{n_agents}_{mode}_leader-{leader}.jsonl",
        ]
        if name in patterns:
            return meta
    return None


def _vote_counts(votes):
    counts = {name: 0 for name in ACTION_KEYS}
    for action, count in votes.items():
        name = _action_name(action)
        if name in counts:
            counts[name] = count
    return counts


def _action_name(action):
    if action is None:
        return ""
    return ACTION_NAMES.get(str(action), str(action))


def _format_actions(actions):
    return " ".join(_action_name(a) for a in actions)


def _format_position(pos):
    if not pos:
        return ""
    return f"({pos[0]},{pos[1]})"


if __name__ == "__main__":
    main()
