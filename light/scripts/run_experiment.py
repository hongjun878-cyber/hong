"""
Main experiment entry point.

Examples
--------
# Quick smoke-test (1 maze, 1 agent, majority only)
python run_experiment.py --dry_run

# Full experiment
python run_experiment.py \
    --index data/mazes/index.json \
    --n_agents 1 2 3 5 \
    --modes majority deliberation \
    --view_ratio 0.3 \
    --max_steps 300 \
    --out_dir results/exp01

# Single episode (for debugging)
python run_experiment.py \
    --single data/mazes/10x10/maze_10x10_seed0.json \
    --n_agents 3 \
    --modes majority \
    --verbose
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.runner import run_episode, run_experiment


def main():
    parser = argparse.ArgumentParser(description="Multi-agent maze experiment")
    parser.add_argument("--index",       type=str, default="data/mazes/index.json",
                        help="Dataset index JSON (from generate_dataset.py)")
    parser.add_argument("--single",      type=str, default=None,
                        help="Run a single maze file instead of full dataset")
    parser.add_argument("--n_agents",    type=int, nargs="+", default=[1, 2, 3, 5],
                        help="Agent counts to test")
    parser.add_argument("--modes",       type=str, nargs="+",
                        default=["majority", "deliberation"],
                        choices=["majority", "deliberation"])
    parser.add_argument("--view_ratio",  type=float, default=0.3)
    parser.add_argument("--max_steps",   type=int,   default=300)
    parser.add_argument("--max_rounds",  type=int,   default=4,
                        help="Max deliberation rounds per step")
    parser.add_argument("--out_dir",     type=str,   default="results/exp01")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--enable_override", action="store_true",
                        help="Re-enable deadlock circuit-breaker (off by default)")
    parser.add_argument("--leader_personas", type=str, nargs="+", default=None,
                        help="Leader personas: all, goal_seeker, explorer, backtracker, balanced_strategist, or 0-3")
    parser.add_argument("--dry_run",     action="store_true",
                        help="One maze, one agent, majority only - quick sanity check")
    args = parser.parse_args()

    # Load model once before any episode
    import src.agent.llm_backend as llm_backend
    llm_backend.load_model()

    disable_override = not args.enable_override
    leader_personas = _parse_leader_personas(args.leader_personas)

    if args.dry_run:
        print("=== DRY RUN ===")
        result = run_episode(
            maze_path="data/mazes/5x5/maze_5x5_seed0.json",
            n_agents=1,
            decision_mode="majority",
            view_ratio=args.view_ratio,
            max_steps=args.max_steps,
            disable_override=disable_override,
            verbose=True,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
        return

    if args.single:
        for n in args.n_agents:
            for mode in args.modes:
                for leader_persona in ([None] if leader_personas is None else leader_personas):
                    print(
                        f"\n=== single={args.single}  n={n}  mode={mode}  "
                        f"leader={_leader_label(leader_persona)} ==="
                    )
                    result = run_episode(
                        maze_path=args.single,
                        n_agents=n,
                        decision_mode=mode,
                        view_ratio=args.view_ratio,
                        max_steps=args.max_steps,
                        max_deliberation_rounds=args.max_rounds,
                        disable_override=disable_override,
                        leader_persona=leader_persona,
                        verbose=args.verbose,
                    )
                    print(json.dumps(result, indent=2))
        return

    run_experiment(
        index_path=args.index,
        n_agents_list=args.n_agents,
        decision_modes=args.modes,
        view_ratio=args.view_ratio,
        max_steps=args.max_steps,
        max_deliberation_rounds=args.max_rounds,
        disable_override=disable_override,
        leader_personas=leader_personas,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )


def _parse_leader_personas(values):
    if values is None:
        return None

    name_to_id = {
        "goal": 0,
        "goal_seeker": 0,
        "goal-seeker": 0,
        "explorer": 1,
        "backtracker": 2,
        "balanced": 3,
        "balanced_strategist": 3,
        "balanced-strategist": 3,
    }

    parsed = []
    for value in values:
        key = value.strip().lower()
        if key == "all":
            return [0, 1, 2, 3]
        if key in name_to_id:
            parsed.append(name_to_id[key])
        else:
            parsed.append(int(key))
    return parsed


def _leader_label(leader_persona):
    labels = ["goal_seeker", "explorer", "backtracker", "balanced_strategist"]
    if leader_persona is None:
        return "rotating"
    return labels[int(leader_persona) % len(labels)]


if __name__ == "__main__":
    main()
