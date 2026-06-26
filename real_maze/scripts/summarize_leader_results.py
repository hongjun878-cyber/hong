import argparse
import csv
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Summarize leader-persona experiment results")
    parser.add_argument("--summary", required=True, help="Path to summary.json from run_experiment.py")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    with open(args.summary) as f:
        rows = json.load(f)

    groups = defaultdict(list)
    for row in rows:
        exp = row.get("experiment", {})
        key = (
            exp.get("decision_mode", row.get("decision_mode")),
            exp.get("n_agents", row.get("n_agents")),
            exp.get("leader_persona_name", row.get("leader_persona_name", "unknown")),
        )
        groups[key].append(row)

    out_rows = []
    for (mode, n_agents, leader), items in sorted(groups.items()):
        count = len(items)
        success_count = sum(1 for item in items if item.get("success"))
        out_rows.append({
            "decision_mode": mode,
            "n_agents": n_agents,
            "leader_persona": leader,
            "episodes": count,
            "successes": success_count,
            "success_rate": round(success_count / count, 4) if count else 0.0,
            "avg_steps": _avg(items, "steps_taken"),
            "avg_deadlocks": _avg(items, "deadlock_count"),
            "avg_llm_calls": _avg(items, "total_llm_calls"),
            "avg_decision_time_s": _avg(items, "total_decision_time_s"),
        })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            writer.writeheader()
            writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {args.out}")


def _avg(items, key):
    values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
    return round(sum(values) / len(values), 4) if values else ""


if __name__ == "__main__":
    main()
