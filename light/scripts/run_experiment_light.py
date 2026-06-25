"""
Lightweight leader-persona experiment runner.

This file does not replace run_experiment.py. It uses the same experiment
pipeline, but defaults to a smaller model and shorter deliberation settings.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.runner import run_episode, run_experiment


DEFAULT_LIGHT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Lightweight multi-agent maze experiment")
    parser.add_argument("--index", type=str, default="data/mazes/index_5x5.json")
    parser.add_argument("--single", type=str, default=None)
    parser.add_argument("--n_agents", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["deliberation"],
        choices=["majority", "deliberation"],
    )
    parser.add_argument("--leader_personas", type=str, nargs="+", default=["all"])
    parser.add_argument("--model_id", type=str, default=DEFAULT_LIGHT_MODEL)
    parser.add_argument("--view_ratio", type=float, default=0.3)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--max_rounds", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="results/leader_light")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enable_override", action="store_true")
    args = parser.parse_args()

    import src.agent.llm_backend as llm_backend

    llm_backend.load_model(model_id=args.model_id)

    disable_override = not args.enable_override
    leader_personas = _parse_leader_personas(args.leader_personas)

    if args.single:
        for n in args.n_agents:
            for mode in args.modes:
                for leader_persona in leader_personas:
                    print(
                        f"\n=== single={args.single}  n={n}  mode={mode}  "
                        f"leader={_leader_label(leader_persona)}  model={args.model_id} ==="
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
    for value in values or ["all"]:
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
    return labels[int(leader_persona) % len(labels)]


if __name__ == "__main__":
    main()
