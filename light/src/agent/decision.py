"""
Multi-agent decision modules.

Two modes
---------
majority_vote(responses)
    Each agent replies independently. Most-voted valid action wins.
    Tie → lexicographic order among tied actions (deterministic).

deliberation(prompt_fn, n_agents, available, max_rounds)
    Agents build a shared discussion thread until they reach consensus
    or max_rounds is exhausted. Falls back to majority if no consensus.
"""

import time
from collections import Counter
from src.agent.env import obs_to_prompt, get_persona

ACTION_NAMES = {0: "left", 1: "up", 2: "right", 3: "down"}


# -----------------------------------------------------------------------
# Majority voting
# -----------------------------------------------------------------------

def majority_vote(
    obs: dict,
    n_agents: int,
    available: list,
    agent_ids: list = None,
    shared_map: dict = None,
    log_fn=None,
) -> tuple:
    """
    Ask each agent independently with its own persona, tally votes.

    Parameters
    ----------
    obs        : raw observation dict (not yet rendered to text)
    shared_map : cumulative visit counts for shared-memory prompt enrichment

    Returns
    -------
    action          : int   winning action
    votes           : dict  {action_int: count}
    raw_responses   : list  raw text from each agent
    elapsed_s       : float wall-clock decision time
    llm_calls       : int   number of LLM calls made (비판 4)
    """
    from src.agent.llm_backend import generate, parse_action

    if agent_ids is None:
        agent_ids = list(range(n_agents))

    def _log(msg):
        if log_fn:
            log_fn(msg)

    t0 = time.time()
    raw_responses = []
    actions_cast  = []
    llm_calls     = 0

    for aid in agent_ids:
        # 비판 2: 에이전트별 페르소나 포함 프롬프트, 비판 3: 공유 지도 포함
        prompt = obs_to_prompt(obs, agent_id=aid, shared_map=shared_map)
        system = get_persona(aid)
        resp   = generate(prompt, system=system, max_new_tokens=16)
        llm_calls += 1
        action = parse_action(resp, available)
        raw_responses.append(resp)
        if action is not None:
            actions_cast.append(action)
        _log(f"    [VOTE] Agent {aid}: \"{resp.strip()}\" → {ACTION_NAMES.get(action, action)}")

    tally  = Counter(actions_cast)
    winner = max(tally, key=lambda a: (tally[a], -a)) if tally else available[0]

    tally_str = ", ".join(f"{ACTION_NAMES.get(a,a)}×{c}" for a, c in sorted(tally.items()))
    _log(f"    [TALLY] {tally_str}  →  winner={ACTION_NAMES.get(winner, winner)}")

    return winner, dict(tally), raw_responses, round(time.time() - t0, 4), llm_calls


# -----------------------------------------------------------------------
# Deliberation
# -----------------------------------------------------------------------

_DELIBERATION_SYSTEM = (
    "You are one of several agents navigating a maze together. "
    "Read the shared discussion carefully before replying. "
    "If you agree with the last proposed move, say 'AGREE: <move>'. "
    "If you disagree, say 'PROPOSE: <move>' followed by a brief reason (one sentence). "
    "Be concise."
)

_INITIAL_SYSTEM = (
    "You are the first agent to speak. Propose a move for the group. "
    "Format: 'PROPOSE: <move>  Reason: <one sentence>'"
)


def deliberation(
    obs: dict,
    n_agents: int,
    available: list,
    max_rounds: int = 4,
    agent_ids: list = None,
    shared_map: dict = None,
    log_fn=None,
) -> tuple:
    """
    Agents share a discussion thread until unanimous consensus or max_rounds.

    Parameters
    ----------
    obs        : raw observation dict
    shared_map : cumulative visit counts for shared-memory prompt enrichment

    Returns
    -------
    action             : int   agreed / majority action
    consensus_reached  : bool
    discussion_rounds  : int   actual rounds used
    discussion_log     : list  of {"agent": id, "text": str} dicts
    elapsed_s          : float
    llm_calls          : int   total LLM calls made this step (비판 4)
    """
    from src.agent.llm_backend import generate, parse_action

    if agent_ids is None:
        agent_ids = list(range(n_agents))

    def _log(msg):
        if log_fn:
            log_fn(msg)

    t0        = time.time()
    log       = []
    llm_calls = 0

    # Round 0: first agent proposes (with its persona + shared map)
    obs_prompt   = obs_to_prompt(obs, agent_id=agent_ids[0], shared_map=shared_map)
    first_prompt = f"{obs_prompt}\n\nYou go first. Propose a move."
    first_resp   = generate(first_prompt, system=_INITIAL_SYSTEM, max_new_tokens=64)
    llm_calls   += 1
    log.append({"agent": agent_ids[0], "text": first_resp})
    _log(f"    [DELIBERATION Round 0] Agent {agent_ids[0]} proposes: \"{first_resp.strip()}\"")

    for round_idx in range(1, max_rounds + 1):
        _log(f"    [DELIBERATION Round {round_idx}]")
        discussion_ctx = "\n".join(
            f"Agent {e['agent']}: {e['text']}" for e in log
        )

        new_log = []
        for aid in agent_ids[1:]:
            # 비판 2: 각 에이전트 페르소나 + 비판 3: 공유 지도 포함
            agent_obs_prompt = obs_to_prompt(obs, agent_id=aid, shared_map=shared_map)
            full_prompt = (
                f"{agent_obs_prompt}\n\n"
                f"--- Discussion so far ---\n{discussion_ctx}\n"
                f"--- End of discussion ---\n\n"
                f"Now give your response (AGREE or PROPOSE)."
            )
            resp = generate(full_prompt, system=_DELIBERATION_SYSTEM, max_new_tokens=80)
            llm_calls += 1
            new_log.append({"agent": aid, "text": resp})
            tag = "AGREES" if _extract_agreement(resp) else "COUNTERS"
            _log(f"      Agent {aid} {tag}: \"{resp.strip()}\"")

        log.extend(new_log)

        recent = [e["text"] for e in new_log]
        if all(_extract_agreement(r) for r in recent):
            agreed_action = parse_action(recent[0], available)
            if agreed_action is None:
                _log("    [NO CONSENSUS] agreed text did not contain a valid move")
                continue
            _log(f"    [CONSENSUS] Round {round_idx} → {ACTION_NAMES.get(agreed_action, agreed_action)}")
            return (
                agreed_action,
                True,
                round_idx,
                log,
                round(time.time() - t0, 4),
                llm_calls,
            )

        _log(f"    [NO CONSENSUS] continuing...")
        agent_ids = agent_ids[1:] + [agent_ids[0]]

    # Fallback: majority vote over all proposed actions in log
    all_actions = [parse_action(e["text"], available) for e in log]
    tally   = Counter(a for a in all_actions if a is not None)
    winner  = max(tally, key=lambda a: (tally[a], -a)) if tally else available[0]
    _log(f"    [FALLBACK MAJORITY] {max_rounds} rounds exhausted → {ACTION_NAMES.get(winner, winner)}")

    return winner, False, max_rounds, log, round(time.time() - t0, 4), llm_calls


def _extract_agreement(text: str) -> bool:
    return text.strip().upper().startswith("AGREE")
