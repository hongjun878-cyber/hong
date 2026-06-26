"""
Leader + vice-leader weighted majority experiment runner.

Designed for the 7x7 condition:
- leader vote weight = 3
- vice-leader vote weight = 2
- other agent votes weight = 1
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent.env import MazeEnv, get_persona, get_persona_name, obs_to_prompt


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ACTION_NAMES = {0: "left", 1: "up", 2: "right", 3: "down"}

# Persona IDs:
# 0 = goal_seeker, 1 = explorer, 2 = backtracker, 3 = balanced_strategist
VICE_PERSONA_BY_LEADER = {
    0: 1,  # goal_seeker leader -> explorer vice
    1: 3,  # explorer leader -> balanced_strategist vice
    2: 1,  # backtracker leader -> explorer vice
    3: 1,  # balanced_strategist leader -> explorer vice
}


def main():
    parser = argparse.ArgumentParser(description="Leader + vice weighted majority maze experiment")
    parser.add_argument("--index", type=str, default="data/mazes/index_7x7.json")
    parser.add_argument("--single", type=str, default=None)
    parser.add_argument("--n_agents", type=int, nargs="+", default=[4])
    parser.add_argument("--leader_personas", type=str, nargs="+", default=["all"])
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--leader_weight", type=float, default=3.0)
    parser.add_argument("--vice_weight", type=float, default=2.0)
    parser.add_argument("--view_ratio", type=float, default=0.3)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--out_dir", type=str, default="results/leader_vice_majority_7x7_1_5b")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import src.agent.llm_backend as llm_backend

    llm_backend.load_model(model_id=args.model_id)
    leader_personas = _parse_leader_personas(args.leader_personas)

    if args.single:
        for n in args.n_agents:
            for leader_persona in leader_personas:
                result = run_episode(
                    maze_path=args.single,
                    n_agents=n,
                    leader_persona=leader_persona,
                    leader_weight=args.leader_weight,
                    vice_weight=args.vice_weight,
                    view_ratio=args.view_ratio,
                    max_steps=args.max_steps,
                    verbose=args.verbose,
                )
                print(json.dumps(result, indent=2))
        return

    run_experiment(
        index_path=args.index,
        n_agents_list=args.n_agents,
        leader_personas=leader_personas,
        leader_weight=args.leader_weight,
        vice_weight=args.vice_weight,
        view_ratio=args.view_ratio,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )


def run_experiment(
    index_path,
    n_agents_list,
    leader_personas,
    leader_weight=3.0,
    vice_weight=2.0,
    view_ratio=0.3,
    max_steps=80,
    out_dir="results/leader_vice_majority_7x7_1_5b",
    verbose=False,
):
    with open(index_path) as f:
        maze_index = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "decision_logs")
    os.makedirs(logs_dir, exist_ok=True)

    all_results = []
    total = len(maze_index) * len(n_agents_list) * len(leader_personas)
    done = 0

    for entry in maze_index:
        for n_agents in n_agents_list:
            for leader_persona in leader_personas:
                done += 1
                leader_name = get_persona_name(leader_persona)
                vice_persona = _vice_for_leader(leader_persona)
                vice_name = get_persona_name(vice_persona)
                print(
                    f"[{done}/{total}] {entry['name']}  n={n_agents}  "
                    f"leader={leader_name}  vice={vice_name}  mode=leader_vice_weighted_majority",
                    flush=True,
                )
                fname = f"{entry['name']}_n{n_agents}_leader_vice_weighted_majority_leader-{leader_name}"
                t0 = time.time()
                result = run_episode(
                    maze_path=entry["json_path"],
                    n_agents=n_agents,
                    leader_persona=leader_persona,
                    leader_weight=leader_weight,
                    vice_weight=vice_weight,
                    view_ratio=view_ratio,
                    max_steps=max_steps,
                    verbose=verbose,
                    decision_log_path=os.path.join(logs_dir, f"{fname}.jsonl"),
                )
                result["experiment"] = {
                    "maze_name": entry["name"],
                    "difficulty": entry.get("difficulty", "unknown"),
                    "path_length": entry.get("path_length"),
                    "junctions": entry.get("junctions"),
                    "n_agents": n_agents,
                    "decision_mode": "leader_vice_weighted_majority",
                    "leader_persona": leader_persona,
                    "leader_persona_name": leader_name,
                    "vice_persona": vice_persona,
                    "vice_persona_name": vice_name,
                    "leader_weight": leader_weight,
                    "vice_weight": vice_weight,
                    "wall_time_s": round(time.time() - t0, 2),
                }

                with open(os.path.join(out_dir, f"{fname}.json"), "w") as f:
                    json.dump(result, f, indent=2)

                all_results.append(result)
                print(
                    f"    success={result['success']}  steps={result['steps_taken']}  "
                    f"deadlocks={result['deadlock_count']}  calls={result['total_llm_calls']}  "
                    f"parse_failures={result['parse_failures']}",
                    flush=True,
                )

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nExperiment complete. {len(all_results)} episodes -> {summary_path}", flush=True)
    return all_results


def run_episode(
    maze_path,
    n_agents,
    leader_persona,
    leader_weight=3.0,
    vice_weight=2.0,
    view_ratio=0.3,
    max_steps=80,
    verbose=False,
    decision_log_path=None,
):
    env = MazeEnv(maze_path, view_ratio=view_ratio, max_steps=max_steps)
    obs = env.reset()
    vice_persona = _vice_for_leader(leader_persona)
    agent_ids = _build_agent_ids(n_agents, leader_persona, vice_persona)
    leader_id = agent_ids[0]
    vice_id = agent_ids[1] if len(agent_ids) > 1 else None
    leader_name = get_persona_name(leader_id)
    vice_name = get_persona_name(vice_id) if vice_id is not None else None
    shared_map = {}
    decision_log = []
    total_llm_calls = 0
    total_parse_failures = 0
    total_fallbacks = 0

    while True:
        current = tuple(obs["position"])
        shared_map[current] = shared_map.get(current, 0) + 1

        decision = leader_vice_weighted_majority(
            obs=obs,
            agent_ids=agent_ids,
            leader_id=leader_id,
            vice_id=vice_id,
            leader_weight=leader_weight,
            vice_weight=vice_weight,
            shared_map=shared_map,
            verbose=verbose,
        )
        total_llm_calls += decision["llm_calls"]
        total_parse_failures += decision["parse_failures"]
        total_fallbacks += 1 if decision["used_fallback"] else 0

        obs, reward, done, info = env.step(
            decision["action"],
            decision_time_s=decision["elapsed_s"],
            votes=decision["raw_votes"],
            discussion_rounds=None,
        )

        log_entry = {
            "step": info["step"],
            "position": info["position"],
            "available": decision["available"],
            "is_junction": len(decision["available"]) >= 3,
            "agent_order": list(agent_ids),
            "leader_persona": leader_name,
            "vice_persona": vice_name,
            "leader_weight": leader_weight,
            "vice_weight": vice_weight,
            "leader_action": decision["leader_action"],
            "vice_action": decision["vice_action"],
            "action_chosen": decision["action"],
            "raw_votes": decision["raw_votes"],
            "weighted_votes": decision["weighted_votes"],
            "parse_failures": decision["parse_failures"],
            "used_fallback": decision["used_fallback"],
            "decision_time_s": decision["elapsed_s"],
            "llm_calls": decision["llm_calls"],
            "deadlock_count": info["deadlock_count"],
            "responses": decision["responses"],
        }
        decision_log.append(log_entry)

        if verbose:
            print(
                f"step={info['step']} pos={info['position']} "
                f"action={ACTION_NAMES.get(decision['action'])} "
                f"deadlocks={info['deadlock_count']}",
                flush=True,
            )

        if done:
            break

    result = asdict(env.get_result(n_agents=n_agents, decision_mode="leader_vice_weighted_majority"))
    result.pop("history", None)
    result["leader_persona"] = leader_persona
    result["leader_persona_name"] = leader_name
    result["vice_persona"] = vice_persona
    result["vice_persona_name"] = vice_name
    result["leader_weight"] = leader_weight
    result["vice_weight"] = vice_weight
    result["agent_ids"] = agent_ids
    result["total_llm_calls"] = total_llm_calls
    result["parse_failures"] = total_parse_failures
    result["fallback_decisions"] = total_fallbacks
    result["leader_vice_majority_stats"] = _compute_leader_vice_stats(decision_log)

    if decision_log_path:
        os.makedirs(os.path.dirname(decision_log_path) or ".", exist_ok=True)
        with open(decision_log_path, "w") as f:
            for entry in decision_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return result


def leader_vice_weighted_majority(
    obs,
    agent_ids,
    leader_id,
    vice_id,
    leader_weight,
    vice_weight,
    shared_map,
    verbose=False,
):
    from src.agent.llm_backend import generate, parse_action

    t0 = time.time()
    available = obs["available_moves"]
    weighted_votes = defaultdict(float)
    raw_votes = Counter()
    responses = []
    leader_action = None
    vice_action = None
    parse_failures = 0

    for aid in agent_ids:
        prompt = obs_to_prompt(obs, agent_id=aid, shared_map=shared_map)
        resp = generate(prompt, system=get_persona(aid), max_new_tokens=16)
        action = parse_action(resp, available)
        if action is None:
            parse_failures += 1
        else:
            if aid == leader_id:
                weight = leader_weight
                leader_action = action
            elif aid == vice_id:
                weight = vice_weight
                vice_action = action
            else:
                weight = 1.0
            weighted_votes[action] += weight
            raw_votes[action] += 1
        responses.append({
            "agent_id": aid,
            "persona": get_persona_name(aid),
            "role": _role_name(aid, leader_id, vice_id),
            "raw_text": resp,
            "action": action,
        })
        if verbose:
            print(
                f"    agent={aid} role={_role_name(aid, leader_id, vice_id)} "
                f"persona={get_persona_name(aid)} action={ACTION_NAMES.get(action, action)} "
                f"text={resp.strip()!r}",
                flush=True,
            )

    if weighted_votes:
        action = max(
            weighted_votes,
            key=lambda a: (
                weighted_votes[a],
                1 if leader_action == a else 0,
                1 if vice_action == a else 0,
                raw_votes[a],
                -a,
            ),
        )
        used_fallback = False
    else:
        action = available[0]
        used_fallback = True

    return {
        "action": action,
        "available": list(available),
        "leader_action": leader_action,
        "vice_action": vice_action,
        "weighted_votes": {str(k): v for k, v in sorted(weighted_votes.items())},
        "raw_votes": {str(k): v for k, v in sorted(raw_votes.items())},
        "responses": responses,
        "parse_failures": parse_failures,
        "used_fallback": used_fallback,
        "llm_calls": len(agent_ids),
        "elapsed_s": round(time.time() - t0, 4),
    }


def _compute_leader_vice_stats(decision_log):
    if not decision_log:
        return {}
    leader_followed = [
        entry for entry in decision_log
        if entry["leader_action"] is not None and entry["leader_action"] == entry["action_chosen"]
    ]
    vice_followed = [
        entry for entry in decision_log
        if entry["vice_action"] is not None and entry["vice_action"] == entry["action_chosen"]
    ]
    split_steps = [
        entry for entry in decision_log
        if len(entry["raw_votes"]) > 1
    ]
    return {
        "decision_steps": len(decision_log),
        "leader_followed_ratio": round(len(leader_followed) / len(decision_log), 3),
        "vice_followed_ratio": round(len(vice_followed) / len(decision_log), 3),
        "split_ratio": round(len(split_steps) / len(decision_log), 3),
        "avg_parse_failures": round(
            sum(entry["parse_failures"] for entry in decision_log) / len(decision_log),
            3,
        ),
        "fallback_decision_ratio": round(
            sum(1 for entry in decision_log if entry["used_fallback"]) / len(decision_log),
            3,
        ),
    }


def _vice_for_leader(leader_persona):
    return VICE_PERSONA_BY_LEADER[int(leader_persona) % 4]


def _build_agent_ids(n_agents, leader_persona, vice_persona):
    leader_id = int(leader_persona) % 4
    vice_id = int(vice_persona) % 4
    ids = [leader_id]
    if n_agents > 1 and vice_id != leader_id:
        ids.append(vice_id)
    candidate = 0
    while len(ids) < n_agents:
        persona = candidate % 4
        if persona not in ids:
            ids.append(persona)
        candidate += 1
    return ids


def _role_name(agent_id, leader_id, vice_id):
    if agent_id == leader_id:
        return "leader"
    if agent_id == vice_id:
        return "vice"
    return "member"


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


if __name__ == "__main__":
    main()
