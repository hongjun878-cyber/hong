"""
Episode runner — wires MazeEnv + decision module into a full episode.
"""

import json
import os
import random
import time
from dataclasses import asdict

from src.agent.env import AGENT_PERSONAS, MazeEnv, get_persona_name, obs_to_prompt
from src.agent.decision import majority_vote, deliberation


def run_episode(
    maze_path: str,
    n_agents: int,
    decision_mode: str,
    view_ratio: float = 0.3,
    max_steps: int = 300,
    max_deliberation_rounds: int = 4,
    disable_override: bool = True,
    leader_persona: int = None,
    verbose: bool = False,
    decision_log_path: str = None,
) -> dict:
    """
    Run one full episode and return a result dict.

    Parameters
    ----------
    disable_override   : if True, circuit-breaker is disabled (override off).
                         Set False only for baseline comparisons.
    decision_log_path  : if given, per-step agent decisions are written here
                         as a JSONL file (one JSON object per step).
    """
    # override threshold: effectively infinite when disabled
    deadlock_override_threshold = 9999 if disable_override else 8

    env = MazeEnv(maze_path, view_ratio=view_ratio, max_steps=max_steps)
    obs = env.reset()

    agent_ids          = _build_agent_ids(n_agents, leader_persona=leader_persona)
    leader_name        = get_persona_name(agent_ids[0]) if leader_persona is not None else None
    override_count     = 0
    visit_count        = {}   # for override fallback (unused when disabled)
    total_llm_calls    = 0    # 비판 4: 총 LLM 호출 수 추적
    shared_map         = {}   # 비판 3: 에피소드 전체 공유 탐색 지도 {(x,y): visit_count}
    decision_log       = []
    _rng               = random.Random()  # 첫 제안자 랜덤 rotate용

    step_lines = []

    def decision_log_fn(msg: str):
        step_lines.append(msg)
        if verbose:
            print(msg, flush=True)

    while True:
        available = obs["available_moves"]
        deadlocks = obs.get("deadlock_count", 0)

        cur = tuple(obs["position"])
        visit_count[cur] = visit_count.get(cur, 0) + 1
        shared_map[cur]  = shared_map.get(cur, 0) + 1   # 비판 3: 공유 지도 갱신

        # 첫 제안자를 매 스텝 랜덤하게 결정 — 규칙적 교번 패턴 방지
        if leader_persona is None:
            offset = _rng.randint(0, n_agents - 1)
            step_agent_ids = agent_ids[offset:] + agent_ids[:offset]
        else:
            step_agent_ids = list(agent_ids)

        step_lines.clear()
        votes        = None
        disc_rounds  = None
        was_override = False
        llm_calls    = 0

        # circuit-breaker (disable_override=True のとき実質無効)
        if deadlocks >= deadlock_override_threshold:
            def visit_score(a):
                t = tuple(_action_target(obs["position"], a))
                return visit_count.get(t, 0)

            action = min(available, key=visit_score)
            scores = {a: visit_score(a) for a in available}
            dt     = 0.0
            was_override = True
            override_count += 1
            decision_log_fn(
                f"  [OVERRIDE] deadlock={deadlocks} "
                f"→ force action={action}  visit_scores={scores}"
            )

        elif decision_mode == "majority":
            action, votes, raw, dt, llm_calls = majority_vote(
                obs, n_agents, available,
                agent_ids=list(step_agent_ids),
                shared_map=shared_map,
                log_fn=decision_log_fn,
            )
            disc_rounds = None

        else:
            # rotated_agent_ids → 스텝마다 다른 에이전트가 먼저 제안
            action, consensus, disc_rounds, dlog, dt, llm_calls = deliberation(
                obs, n_agents, available,
                max_rounds=max_deliberation_rounds,
                agent_ids=list(step_agent_ids),
                shared_map=shared_map,
                log_fn=decision_log_fn,
            )
            votes = None

        total_llm_calls += llm_calls   # 비판 4: 누적

        obs, reward, done, info = env.step(
            action,
            decision_time_s=dt,
            votes=votes,
            discussion_rounds=disc_rounds,
        )

        if verbose:
            print(
                f"step {info['step']:3d}  pos={info['position']}  "
                f"action={action}  valid={info['valid_move']}  "
                f"dt={dt:.2f}s  llm_calls={llm_calls}  deadlocks={info['deadlock_count']}",
                flush=True,
            )

        is_junction = len(available) >= 3   # 갈랫길: 선택지 3개 이상

        decision_log.append({
            "step":            info["step"],
            "position":        info["position"],
            "available":       available,
            "is_junction":     is_junction,
            "agent_order":     list(step_agent_ids),
            "leader_persona":  leader_name,
            "action_chosen":   action,
            "was_override":    was_override,
            "decision_time_s": dt,
            "llm_calls":       llm_calls,
            "deadlock_count":  info["deadlock_count"],
            "votes":           votes,
            "disc_rounds":     disc_rounds,
            "agent_log":       list(step_lines),
        })

        if done:
            break

    result = env.get_result(n_agents=n_agents, decision_mode=decision_mode)
    result_dict = _serialise(result)
    result_dict["override_count"]   = override_count
    result_dict["total_llm_calls"]  = total_llm_calls
    result_dict["disable_override"] = disable_override
    result_dict["leader_persona"] = leader_persona
    result_dict["leader_persona_name"] = leader_name
    result_dict["agent_ids"] = agent_ids
    result_dict["persona_diversity"]  = _compute_persona_diversity(decision_log)
    result_dict["junction_stats"]     = _compute_junction_stats(decision_log)

    if verbose:
        print(
            f"\n{'SUCCESS' if result_dict['success'] else 'FAILURE'}  "
            f"steps={result_dict['steps_taken']}  "
            f"llm_calls={total_llm_calls}  "
            f"decision_time={result_dict['total_decision_time_s']:.1f}s  "
            f"deadlocks={result_dict['deadlock_count']}",
            flush=True,
        )

    if decision_log_path:
        os.makedirs(os.path.dirname(decision_log_path) or ".", exist_ok=True)
        with open(decision_log_path, "w") as f:
            for entry in decision_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return result_dict


def run_experiment(
    index_path: str,
    n_agents_list: list,
    decision_modes: list,
    view_ratio: float = 0.3,
    max_steps: int = 300,
    max_deliberation_rounds: int = 4,
    disable_override: bool = True,
    leader_personas: list = None,
    out_dir: str = "results",
    verbose: bool = False,
) -> list:
    if leader_personas is not None:
        return run_leader_experiment(
            index_path=index_path,
            n_agents_list=n_agents_list,
            decision_modes=decision_modes,
            leader_personas=leader_personas,
            view_ratio=view_ratio,
            max_steps=max_steps,
            max_deliberation_rounds=max_deliberation_rounds,
            disable_override=disable_override,
            out_dir=out_dir,
            verbose=verbose,
        )

    with open(index_path) as f:
        maze_index = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "decision_logs")
    os.makedirs(logs_dir, exist_ok=True)

    all_results = []
    leader_values = [None] if leader_personas is None else list(leader_personas)
    total = len(maze_index) * len(n_agents_list) * len(decision_modes) * len(leader_values)
    done  = 0

    for entry in maze_index:
        for n in n_agents_list:
            for mode in decision_modes:
                done += 1
                _log(f"[{done}/{total}] {entry['name']}  n={n}  mode={mode}")

                t0    = time.time()
                fname = f"{entry['name']}_n{n}_{mode}"

                result = run_episode(
                    maze_path=entry["json_path"],
                    n_agents=n,
                    decision_mode=mode,
                    view_ratio=view_ratio,
                    max_steps=max_steps,
                    max_deliberation_rounds=max_deliberation_rounds,
                    disable_override=disable_override,
                    verbose=verbose,
                    decision_log_path=os.path.join(logs_dir, f"{fname}.jsonl"),
                )
                wall = round(time.time() - t0, 2)

                result["experiment"] = {
                    "maze_name":     entry["name"],
                    "difficulty":    entry.get("difficulty", "unknown"),
                    "path_length":   entry["path_length"],
                    "junctions":     entry["junctions"],
                    "n_agents":      n,
                    "decision_mode": mode,
                    "wall_time_s":   wall,
                }

                with open(os.path.join(out_dir, f"{fname}.json"), "w") as f:
                    json.dump(result, f, indent=2)

                all_results.append(result)
                _log(
                    f"    → success={result['success']}  "
                    f"steps={result['steps_taken']}  "
                    f"llm_calls={result['total_llm_calls']}  "
                    f"deadlocks={result['deadlock_count']}  "
                    f"decision_time={result['total_decision_time_s']:.1f}s"
                )

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    _log(f"\nExperiment complete. {len(all_results)} episodes → {summary_path}")

    return all_results


def run_leader_experiment(
    index_path: str,
    n_agents_list: list,
    decision_modes: list,
    leader_personas: list,
    view_ratio: float = 0.3,
    max_steps: int = 300,
    max_deliberation_rounds: int = 4,
    disable_override: bool = True,
    out_dir: str = "results",
    verbose: bool = False,
) -> list:
    with open(index_path) as f:
        maze_index = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    logs_dir = os.path.join(out_dir, "decision_logs")
    os.makedirs(logs_dir, exist_ok=True)

    all_results = []
    total = len(maze_index) * len(n_agents_list) * len(decision_modes) * len(leader_personas)
    done = 0

    for entry in maze_index:
        for n in n_agents_list:
            for mode in decision_modes:
                for leader_persona in leader_personas:
                    done += 1
                    leader_label = _leader_label(leader_persona)
                    _log(
                        f"[{done}/{total}] {entry['name']}  n={n}  mode={mode}  "
                        f"leader={leader_label}"
                    )

                    t0 = time.time()
                    fname = f"{entry['name']}_n{n}_{mode}_leader-{leader_label}"

                    result = run_episode(
                        maze_path=entry["json_path"],
                        n_agents=n,
                        decision_mode=mode,
                        view_ratio=view_ratio,
                        max_steps=max_steps,
                        max_deliberation_rounds=max_deliberation_rounds,
                        disable_override=disable_override,
                        leader_persona=leader_persona,
                        verbose=verbose,
                        decision_log_path=os.path.join(logs_dir, f"{fname}.jsonl"),
                    )
                    wall = round(time.time() - t0, 2)

                    result["experiment"] = {
                        "maze_name": entry["name"],
                        "difficulty": entry.get("difficulty", "unknown"),
                        "path_length": entry["path_length"],
                        "junctions": entry["junctions"],
                        "n_agents": n,
                        "decision_mode": mode,
                        "leader_persona": leader_persona,
                        "leader_persona_name": result["leader_persona_name"],
                        "wall_time_s": wall,
                    }

                    with open(os.path.join(out_dir, f"{fname}.json"), "w") as f:
                        json.dump(result, f, indent=2)

                    all_results.append(result)
                    _log(
                        f"    success={result['success']}  "
                        f"steps={result['steps_taken']}  "
                        f"llm_calls={result['total_llm_calls']}  "
                        f"deadlocks={result['deadlock_count']}  "
                        f"decision_time={result['total_decision_time_s']:.1f}s"
                    )

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    _log(f"\nLeader experiment complete. {len(all_results)} episodes -> {summary_path}")

    return all_results


def _compute_persona_diversity(decision_log: list) -> dict:
    """
    페르소나 효과 검증 지표.

    majority 모드의 votes 기록을 분석해 에이전트 간 의견 불일치율을 계산한다.
    - unanimous_ratio  : 모든 에이전트가 같은 행동을 선택한 스텝 비율 (낮을수록 다양성 높음)
    - split_ratio      : 2가지 이상 행동이 나온 스텝 비율 (높을수록 다양성 높음)
    - avg_unique_votes : 스텝당 평균 고유 행동 수
    """
    vote_steps = [e for e in decision_log if e.get("votes") and not e["was_override"]]
    if not vote_steps:
        return {"note": "no majority-vote steps (deliberation mode or all overrides)"}

    unanimous  = sum(1 for e in vote_steps if len(e["votes"]) == 1)
    split      = sum(1 for e in vote_steps if len(e["votes"]) > 1)
    avg_unique = sum(len(e["votes"]) for e in vote_steps) / len(vote_steps)

    return {
        "vote_steps":       len(vote_steps),
        "unanimous_steps":  unanimous,
        "split_steps":      split,
        "unanimous_ratio":  round(unanimous / len(vote_steps), 3),
        "split_ratio":      round(split / len(vote_steps), 3),
        "avg_unique_votes": round(avg_unique, 3),
    }


def _compute_junction_stats(decision_log: list) -> dict:
    """
    가설 핵심 지표: junction 스텝(선택지 3개 이상)에서의 의사결정 패턴.

    junction에서 에이전트들이 얼마나 많이 분열하고
    합의에 얼마나 많은 라운드가 필요했는지를 추적한다.
    이것이 '사공이 많으면 배가 산으로 간다'를 직접 측정하는 지표다.
    """
    junction_steps     = [e for e in decision_log if e.get("is_junction") and not e["was_override"]]
    non_junction_steps = [e for e in decision_log if not e.get("is_junction") and not e["was_override"]]

    def _split_rate(steps):
        vote_steps = [s for s in steps if s.get("votes")]
        if not vote_steps:
            return None
        return round(sum(1 for s in vote_steps if len(s["votes"]) > 1) / len(vote_steps), 3)

    def _avg_rounds(steps):
        disc_steps = [s for s in steps if s.get("disc_rounds") is not None]
        if not disc_steps:
            return None
        return round(sum(s["disc_rounds"] for s in disc_steps) / len(disc_steps), 2)

    def _deadlock_rate(steps):
        if not steps:
            return None
        return round(sum(1 for s in steps if s["deadlock_count"] > 0) / len(steps), 3)

    return {
        "junction_steps":            len(junction_steps),
        "non_junction_steps":        len(non_junction_steps),
        "junction_split_ratio":      _split_rate(junction_steps),
        "non_junction_split_ratio":  _split_rate(non_junction_steps),
        "junction_avg_disc_rounds":  _avg_rounds(junction_steps),
        "non_junction_avg_disc_rounds": _avg_rounds(non_junction_steps),
        "junction_deadlock_rate":    _deadlock_rate(junction_steps),
        "non_junction_deadlock_rate":_deadlock_rate(non_junction_steps),
    }


def _action_target(pos: list, action: int) -> list:
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    dx, dy = deltas[action]
    return [pos[0] + dx, pos[1] + dy]


def _build_agent_ids(n_agents: int, leader_persona: int = None) -> list:
    if n_agents <= 0:
        raise ValueError("n_agents must be positive")
    if leader_persona is None:
        return list(range(n_agents))

    n_personas = len(AGENT_PERSONAS)
    leader_id = int(leader_persona) % n_personas
    ids = [leader_id]
    candidate = 0
    while len(ids) < n_agents:
        if candidate % n_personas != leader_id:
            ids.append(candidate)
        candidate += 1
    return ids


def _leader_label(leader_persona: int = None) -> str:
    if leader_persona is None:
        return "rotating"
    return get_persona_name(int(leader_persona))


def _serialise(result) -> dict:
    d = asdict(result)
    d.pop("history", None)
    return d


def _log(msg: str) -> None:
    print(msg, flush=True)
