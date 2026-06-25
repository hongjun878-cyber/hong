from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# Wall order used everywhere in this project:
# 0 = left, 1 = down, 2 = right, 3 = up
DIRS = {
    "L": (-1, 0, 0, 2),
    "D": (0, 1, 1, 3),
    "R": (1, 0, 2, 0),
    "U": (0, -1, 3, 1),
}


@dataclass
class Maze:
    sizex: int
    sizey: int
    seed: int
    walls: List[List[List[int]]]
    start: List[int]
    goal: List[int]
    solution_path: List[List[int]]
    difficulty_name: str = "custom"


@dataclass
class Candidate:
    moves: str = ""
    path: List[List[int]] = field(default_factory=list)
    confidence: Optional[float] = None


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_sec: float


@dataclass
class AgentAnswer:
    agent_id: str
    round_name: str
    raw_text: str
    candidate: Candidate
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_sec: float


@dataclass
class ConsensusRun:
    method: str
    agent_count: int
    final_candidate: Candidate
    answers: List[AgentAnswer]
    judge_answer: Optional[AgentAnswer] = None
    notes: str = ""

    @property
    def all_answers(self) -> List[AgentAnswer]:
        items = list(self.answers)
        if self.judge_answer is not None:
            items.append(self.judge_answer)
        return items

    @property
    def cost(self) -> Dict[str, float]:
        answers = self.all_answers
        return {
            "calls": len(answers),
            "prompt_tokens": sum(a.prompt_tokens for a in answers),
            "completion_tokens": sum(a.completion_tokens for a in answers),
            "total_tokens": sum(a.total_tokens for a in answers),
            "generation_time_sec": sum(a.elapsed_sec for a in answers),
        }


class MockLLM:
    """Tiny fake model for checking that the experiment code runs."""

    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        start = time.perf_counter()
        text = json.dumps({"moves": "", "path": [[0, 0]], "confidence": 0.1})
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        prompt_tokens = max(1, len(prompt_text.split()))
        completion_tokens = max(1, len(text.split()))
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_sec=time.perf_counter() - start,
        )


class LocalHFLLM:
    """Single shared Hugging Face model used by all simulated agents."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        load_in_4bit: bool = False,
        device_map: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: Dict[str, Any] = {"device_map": self.device_map}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            kwargs["torch_dtype"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **kwargs,
        )
        self.model.eval()

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Call llm.load() before generation.")

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        parts = []
        for message in messages:
            parts.append(f"{message['role'].upper()}: {message['content']}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def generate(self, messages: List[Dict[str, str]]) -> GenerationResult:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call llm.load() before generation.")

        import torch

        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        do_sample = self.temperature > 0
        start = time.perf_counter()
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )
        elapsed = time.perf_counter() - start

        completion_ids = output_ids[0][prompt_tokens:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        completion_tokens = int(completion_ids.shape[-1])
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            elapsed_sec=elapsed,
        )


def generate_maze(sizex: int, sizey: int, seed: Optional[int] = None) -> List[List[List[int]]]:
    if seed is not None:
        random.seed(seed)

    walls = [[[1, 1, 1, 1] for _ in range(sizex)] for _ in range(sizey)]
    visited = [[0 for _ in range(sizex)] for _ in range(sizey)]
    x, y = 0, 0
    visited[y][x] = 1
    stack = [[x, y]]

    while stack:
        x, y = stack[-1]
        options = []

        for direction, (dx, dy, wall_idx, _) in DIRS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < sizex and 0 <= ny < sizey and visited[ny][nx] == 0:
                options.append(direction)

        if not options:
            stack.pop()
            continue

        direction = random.choice(options)
        dx, dy, wall_idx, opposite_idx = DIRS[direction]
        nx, ny = x + dx, y + dy
        walls[y][x][wall_idx] = 0
        walls[ny][nx][opposite_idx] = 0
        visited[ny][nx] = 1
        stack.append([nx, ny])

    return walls


def in_bounds(maze: Maze, x: int, y: int) -> bool:
    return 0 <= x < maze.sizex and 0 <= y < maze.sizey


def open_neighbors_from_walls(
    walls: List[List[List[int]]],
    sizex: int,
    sizey: int,
    x: int,
    y: int,
) -> List[Tuple[str, int, int]]:
    neighbors = []
    for direction, (dx, dy, wall_idx, _) in DIRS.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < sizex and 0 <= ny < sizey and walls[y][x][wall_idx] == 0:
            neighbors.append((direction, nx, ny))
    return neighbors


def find_farthest(
    walls: List[List[List[int]]],
    sizex: int,
    sizey: int,
    start: Sequence[int],
) -> List[int]:
    sx, sy = start
    q = deque([(sx, sy)])
    dist = [[-1 for _ in range(sizex)] for _ in range(sizey)]
    dist[sy][sx] = 0

    while q:
        x, y = q.popleft()
        for _, nx, ny in open_neighbors_from_walls(walls, sizex, sizey, x, y):
            if dist[ny][nx] == -1:
                dist[ny][nx] = dist[y][x] + 1
                q.append((nx, ny))

    best = [sx, sy]
    best_d = -1
    for y in range(sizey):
        for x in range(sizex):
            if dist[y][x] > best_d:
                best_d = dist[y][x]
                best = [x, y]
    return best


def carve_start_goal(
    walls: List[List[List[int]]],
    start: Sequence[int],
    goal: Sequence[int],
) -> List[List[List[int]]]:
    sx, sy = start
    gx, gy = goal
    walls[sy][sx][0] = 0
    walls[gy][gx][2] = 0
    return walls


def shortest_path(
    walls: List[List[List[int]]],
    sizex: int,
    sizey: int,
    start: Sequence[int],
    goal: Sequence[int],
) -> List[List[int]]:
    sx, sy = start
    gx, gy = goal
    q = deque([(sx, sy)])
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {(sx, sy): None}

    while q:
        x, y = q.popleft()
        if [x, y] == [gx, gy]:
            break
        for _, nx, ny in open_neighbors_from_walls(walls, sizex, sizey, x, y):
            if (nx, ny) not in prev:
                prev[(nx, ny)] = (x, y)
                q.append((nx, ny))

    if (gx, gy) not in prev:
        return []

    path = []
    cur: Optional[Tuple[int, int]] = (gx, gy)
    while cur is not None:
        path.append([cur[0], cur[1]])
        cur = prev[cur]
    path.reverse()
    return path


def create_maze_case(
    sizex: int,
    sizey: int,
    seed: int,
    difficulty_name: str = "custom",
    start: Sequence[int] = (0, 0),
    goal_mode: str = "farthest",
) -> Maze:
    walls = generate_maze(sizex, sizey, seed=seed)
    if goal_mode == "farthest":
        goal = find_farthest(walls, sizex, sizey, start)
    elif goal_mode == "random":
        goal = [random.randrange(sizex), random.randrange(sizey)]
        while goal == list(start):
            goal = [random.randrange(sizex), random.randrange(sizey)]
    else:
        raise ValueError(f"Unknown goal_mode: {goal_mode}")

    walls = carve_start_goal(walls, start, goal)
    solution = shortest_path(walls, sizex, sizey, start, goal)
    return Maze(
        sizex=sizex,
        sizey=sizey,
        seed=seed,
        walls=walls,
        start=list(start),
        goal=goal,
        solution_path=solution,
        difficulty_name=difficulty_name,
    )


def path_to_moves(path: Sequence[Sequence[int]]) -> str:
    moves = []
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        dx, dy = x2 - x1, y2 - y1
        if (dx, dy) == (-1, 0):
            moves.append("L")
        elif (dx, dy) == (1, 0):
            moves.append("R")
        elif (dx, dy) == (0, -1):
            moves.append("U")
        elif (dx, dy) == (0, 1):
            moves.append("D")
        else:
            return ""
    return "".join(moves)


def path_from_moves(start: Sequence[int], moves: str) -> List[List[int]]:
    x, y = int(start[0]), int(start[1])
    path = [[x, y]]
    for move in moves:
        if move not in DIRS:
            continue
        dx, dy, _, _ = DIRS[move]
        x, y = x + dx, y + dy
        path.append([x, y])
    return path


def legal_step(maze: Maze, x1: int, y1: int, x2: int, y2: int) -> bool:
    if not in_bounds(maze, x1, y1) or not in_bounds(maze, x2, y2):
        return False
    dx, dy = x2 - x1, y2 - y1
    for _, (ddx, ddy, wall_idx, _) in DIRS.items():
        if (dx, dy) == (ddx, ddy):
            return maze.walls[y1][x1][wall_idx] == 0
    return False


def evaluate_path(maze: Maze, path: Sequence[Sequence[int]]) -> Dict[str, Any]:
    shortest_len = max(0, len(maze.solution_path) - 1)
    if not path:
        return {
            "valid_steps": False,
            "success": False,
            "optimal": False,
            "path_length": None,
            "shortest_length": shortest_len,
            "length_ratio": None,
            "error": "empty path",
        }

    clean_path = [[int(p[0]), int(p[1])] for p in path if len(p) == 2]
    if clean_path[0] != maze.start:
        return {
            "valid_steps": False,
            "success": False,
            "optimal": False,
            "path_length": len(clean_path) - 1,
            "shortest_length": shortest_len,
            "length_ratio": None,
            "error": "path does not start at start",
        }

    for i in range(1, len(clean_path)):
        x1, y1 = clean_path[i - 1]
        x2, y2 = clean_path[i]
        if not legal_step(maze, x1, y1, x2, y2):
            return {
                "valid_steps": False,
                "success": False,
                "optimal": False,
                "path_length": len(clean_path) - 1,
                "shortest_length": shortest_len,
                "length_ratio": None,
                "error": f"illegal step from {[x1, y1]} to {[x2, y2]}",
            }

    path_length = len(clean_path) - 1
    success = clean_path[-1] == maze.goal
    optimal = success and path_length == shortest_len
    length_ratio = (path_length / shortest_len) if success and shortest_len > 0 else None
    return {
        "valid_steps": True,
        "success": success,
        "optimal": optimal,
        "path_length": path_length,
        "shortest_length": shortest_len,
        "length_ratio": length_ratio,
        "error": "" if success else "path does not end at goal",
    }


def evaluate_candidate(maze: Maze, candidate: Candidate) -> Dict[str, Any]:
    path = candidate.path
    if candidate.moves:
        path = path_from_moves(maze.start, candidate.moves)
    result = evaluate_path(maze, path)
    result["moves"] = candidate.moves or path_to_moves(path)
    result["path"] = path
    return result


def difficulty_metrics(maze: Maze) -> Dict[str, int]:
    dead_ends = 0
    branch_cells = 0
    for y in range(maze.sizey):
        for x in range(maze.sizex):
            degree = len(open_neighbors_from_walls(maze.walls, maze.sizex, maze.sizey, x, y))
            if degree == 1:
                dead_ends += 1
            if degree >= 3:
                branch_cells += 1

    turns = 0
    moves = path_to_moves(maze.solution_path)
    for i in range(1, len(moves)):
        if moves[i] != moves[i - 1]:
            turns += 1

    return {
        "cells": maze.sizex * maze.sizey,
        "shortest_length": max(0, len(maze.solution_path) - 1),
        "dead_ends": dead_ends,
        "branch_cells": branch_cells,
        "solution_turns": turns,
    }


def maze_to_adjacency_text(maze: Maze) -> str:
    lines = [
        "Coordinate system: x increases to the right, y increases downward.",
        "Allowed moves: L=left, R=right, U=up, D=down.",
        f"Size: {maze.sizex} columns x {maze.sizey} rows.",
        f"Start: ({maze.start[0]},{maze.start[1]})",
        f"Goal: ({maze.goal[0]},{maze.goal[1]})",
        "Open moves from each cell:",
    ]
    for y in range(maze.sizey):
        row_parts = []
        for x in range(maze.sizex):
            moves = [direction for direction, _, _ in open_neighbors_from_walls(maze.walls, maze.sizex, maze.sizey, x, y)]
            row_parts.append(f"({x},{y}):{''.join(moves) if moves else '-'}")
        lines.append(" | ".join(row_parts))
    return "\n".join(lines)


def agent_strategy(agent_id: str) -> str:
    strategies = [
        "Check moves systematically and avoid revisiting cells.",
        "Focus on reaching the goal with a short route.",
        "Verify every move against the open-move list before answering.",
        "Look for corridors and junctions before committing to a route.",
        "Prefer a complete legal route over a guessed short route.",
    ]
    match = re.search(r"(\d+)", agent_id)
    idx = int(match.group(1)) - 1 if match else 0
    return strategies[idx % len(strategies)]


def solver_system_prompt(agent_id: str) -> str:
    return (
        f"You are {agent_id}, a maze path-finding agent. "
        f"Your checking style: {agent_strategy(agent_id)} "
        "Find a legal path from Start to Goal using only the listed open moves. "
        "Return only valid JSON. Do not use markdown. "
        "The required JSON schema is: "
        '{"moves":"RDLU...", "path":[[x0,y0],[x1,y1]], "confidence":0.0}. '
        "The moves string is mandatory. The path can be omitted if it is too long."
    )


def build_solver_messages(
    maze: Maze,
    agent_id: str,
    extra_context: str = "",
) -> List[Dict[str, str]]:
    user_parts = [
        maze_to_adjacency_text(maze),
        "Task: output a path from Start to Goal.",
        "Use only U, D, L, R in the moves string.",
        "Do not include any explanation outside the JSON object.",
    ]
    if extra_context:
        user_parts.append("Other agents' proposals:")
        user_parts.append(extra_context)
        user_parts.append("You may revise your answer after considering them.")

    return [
        {"role": "system", "content": solver_system_prompt(agent_id)},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_judge_messages(
    maze: Maze,
    answers: Sequence[AgentAnswer],
) -> List[Dict[str, str]]:
    candidate_lines = []
    for idx, answer in enumerate(answers, start=1):
        candidate_lines.append(
            f"Candidate {idx} from {answer.agent_id}: "
            f"moves={answer.candidate.moves!r}, "
            f"raw_answer={answer.raw_text.replace(chr(10), ' ')[:360]!r}"
        )

    system = (
        "You are a consensus judge. Select the best candidate path. "
        "Prefer a path that uses legal moves and reaches the goal. "
        "Return only JSON with this schema: "
        '{"selected":1, "moves":"RDLU...", "confidence":0.0}. '
        "Do not invent a new candidate unless all candidates are unusable."
    )
    user = "\n\n".join(
        [
            maze_to_adjacency_text(maze),
            "Candidate paths:",
            "\n".join(candidate_lines),
        ]
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def normalize_path(raw_path: Any) -> List[List[int]]:
    if not isinstance(raw_path, list):
        return []
    path = []
    for item in raw_path:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                path.append([int(item[0]), int(item[1])])
            except (TypeError, ValueError):
                continue
    return path


def parse_candidate(text: str) -> Candidate:
    data = extract_json_object(text)
    if data is not None:
        moves = str(data.get("moves", "")).upper()
        moves = "".join(ch for ch in moves if ch in DIRS)
        path = normalize_path(data.get("path", []))
        confidence = data.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return Candidate(moves=moves, path=path, confidence=confidence)

    moves_match = re.search(r"moves?\s*[:=]\s*[\"']?([UDLRudlr]+)", text)
    moves = ""
    if moves_match:
        moves = "".join(ch for ch in moves_match.group(1).upper() if ch in DIRS)

    pairs = re.findall(r"[\[\(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\]\)]", text)
    path = [[int(x), int(y)] for x, y in pairs]
    return Candidate(moves=moves, path=path)


def call_agent(
    llm: Any,
    maze: Maze,
    agent_id: str,
    round_name: str,
    extra_context: str = "",
) -> AgentAnswer:
    messages = build_solver_messages(maze, agent_id=agent_id, extra_context=extra_context)
    result = llm.generate(messages)
    return AgentAnswer(
        agent_id=agent_id,
        round_name=round_name,
        raw_text=result.text,
        candidate=parse_candidate(result.text),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        elapsed_sec=result.elapsed_sec,
    )


def canonical_candidate(maze: Maze, candidate: Candidate) -> Tuple[Tuple[int, int], ...]:
    ev = evaluate_candidate(maze, candidate)
    path = ev.get("path") or []
    return tuple((int(x), int(y)) for x, y in path)


def select_by_majority(maze: Maze, answers: Sequence[AgentAnswer]) -> Candidate:
    if not answers:
        return Candidate()

    groups = Counter(canonical_candidate(maze, answer.candidate) for answer in answers)
    best_key = None
    best_score = None
    for key, count in groups.items():
        path = [[x, y] for x, y in key]
        ev = evaluate_path(maze, path)
        score = (
            count,
            1 if ev["success"] else 0,
            1 if ev["valid_steps"] else 0,
            -(ev["path_length"] if ev["path_length"] is not None else math.inf),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_key = key

    final_path = [[x, y] for x, y in best_key] if best_key is not None else []
    return Candidate(moves=path_to_moves(final_path), path=final_path)


def answers_context(maze: Maze, answers: Sequence[AgentAnswer], max_raw_chars: int = 240) -> str:
    lines = []
    for answer in answers:
        raw = answer.raw_text.replace("\n", " ")[:max_raw_chars]
        lines.append(
            f"{answer.agent_id}: moves={answer.candidate.moves!r}, "
            f"raw={raw!r}"
        )
    return "\n".join(lines)


def run_single_agent(maze: Maze, llm: Any) -> ConsensusRun:
    answer = call_agent(llm, maze, agent_id="Agent-1", round_name="single")
    return ConsensusRun(
        method="single",
        agent_count=1,
        final_candidate=answer.candidate,
        answers=[answer],
    )


def run_majority_vote(maze: Maze, llm: Any, n_agents: int) -> ConsensusRun:
    answers = [
        call_agent(llm, maze, agent_id=f"Agent-{i}", round_name="independent")
        for i in range(1, n_agents + 1)
    ]
    final_candidate = select_by_majority(maze, answers)
    return ConsensusRun(
        method="majority_vote",
        agent_count=n_agents,
        final_candidate=final_candidate,
        answers=answers,
    )


def run_debate_vote(maze: Maze, llm: Any, n_agents: int, debate_rounds: int = 1) -> ConsensusRun:
    all_answers: List[AgentAnswer] = [
        call_agent(llm, maze, agent_id=f"Agent-{i}", round_name="initial")
        for i in range(1, n_agents + 1)
    ]
    latest = all_answers

    for round_idx in range(1, debate_rounds + 1):
        context = answers_context(maze, latest)
        latest = [
            call_agent(
                llm,
                maze,
                agent_id=f"Agent-{i}",
                round_name=f"debate_round_{round_idx}",
                extra_context=context,
            )
            for i in range(1, n_agents + 1)
        ]
        all_answers.extend(latest)

    final_candidate = select_by_majority(maze, latest)
    return ConsensusRun(
        method="debate_vote",
        agent_count=n_agents,
        final_candidate=final_candidate,
        answers=all_answers,
        notes=f"debate_rounds={debate_rounds}",
    )


def run_judge_consensus(maze: Maze, llm: Any, n_agents: int) -> ConsensusRun:
    answers = [
        call_agent(llm, maze, agent_id=f"Agent-{i}", round_name="proposal")
        for i in range(1, n_agents + 1)
    ]
    messages = build_judge_messages(maze, answers)
    judge_result = llm.generate(messages)
    judge_candidate = parse_candidate(judge_result.text)
    judge_answer = AgentAnswer(
        agent_id="Judge",
        round_name="judge",
        raw_text=judge_result.text,
        candidate=judge_candidate,
        prompt_tokens=judge_result.prompt_tokens,
        completion_tokens=judge_result.completion_tokens,
        total_tokens=judge_result.total_tokens,
        elapsed_sec=judge_result.elapsed_sec,
    )

    data = extract_json_object(judge_result.text) or {}
    selected = data.get("selected")
    final_candidate = judge_candidate
    if isinstance(selected, int) and 1 <= selected <= len(answers):
        final_candidate = answers[selected - 1].candidate
    elif not final_candidate.moves and not final_candidate.path:
        final_candidate = select_by_majority(maze, answers)

    return ConsensusRun(
        method="judge_consensus",
        agent_count=n_agents,
        final_candidate=final_candidate,
        answers=answers,
        judge_answer=judge_answer,
    )


def run_consensus_method(
    maze: Maze,
    llm: Any,
    method: str,
    agent_count: int,
    debate_rounds: int = 1,
) -> ConsensusRun:
    if method == "single":
        return run_single_agent(maze, llm)
    if method == "majority_vote":
        return run_majority_vote(maze, llm, agent_count)
    if method == "debate_vote":
        return run_debate_vote(maze, llm, agent_count, debate_rounds=debate_rounds)
    if method == "judge_consensus":
        return run_judge_consensus(maze, llm, agent_count)
    raise ValueError(f"Unknown consensus method: {method}")


def save_maze_json(maze: Maze, path: str | Path) -> None:
    data = asdict(maze)
    data["wall_order"] = ["left", "down", "right", "up"]
    data["solution_moves"] = path_to_moves(maze.solution_path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_maze_image(maze: Maze, path: str | Path, show_solution: bool = False) -> None:
    import matplotlib.pyplot as plt

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))

    for y in range(maze.sizey):
        for x in range(maze.sizex):
            if maze.walls[y][x][3] == 1:
                plt.plot([x, x + 1], [y, y], color="black", linewidth=1)
            if maze.walls[y][x][1] == 1:
                plt.plot([x, x + 1], [y + 1, y + 1], color="black", linewidth=1)
            if maze.walls[y][x][0] == 1:
                plt.plot([x, x], [y, y + 1], color="black", linewidth=1)
            if maze.walls[y][x][2] == 1:
                plt.plot([x + 1, x + 1], [y, y + 1], color="black", linewidth=1)

    if show_solution and maze.solution_path:
        xs = [p[0] + 0.5 for p in maze.solution_path]
        ys = [p[1] + 0.5 for p in maze.solution_path]
        plt.plot(xs, ys, color="#2f80ed", linewidth=2)

    plt.scatter([maze.start[0] + 0.5], [maze.start[1] + 0.5], color="green", s=60)
    plt.scatter([maze.goal[0] + 0.5], [maze.goal[1] + 0.5], color="red", s=60)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=180)
    plt.close()


def run_to_row(
    maze: Maze,
    consensus: ConsensusRun,
    experiment_id: str,
    error: str = "",
) -> Dict[str, Any]:
    evaluation = evaluate_candidate(maze, consensus.final_candidate)
    metrics = difficulty_metrics(maze)
    cost = consensus.cost
    return {
        "experiment_id": experiment_id,
        "difficulty": maze.difficulty_name,
        "sizex": maze.sizex,
        "sizey": maze.sizey,
        "seed": maze.seed,
        "method": consensus.method,
        "agent_count": consensus.agent_count,
        "start": json.dumps(maze.start),
        "goal": json.dumps(maze.goal),
        "success": int(bool(evaluation["success"])),
        "valid_steps": int(bool(evaluation["valid_steps"])),
        "optimal": int(bool(evaluation["optimal"])),
        "path_length": evaluation["path_length"],
        "shortest_length": evaluation["shortest_length"],
        "length_ratio": evaluation["length_ratio"],
        "error": error or evaluation["error"],
        "cells": metrics["cells"],
        "dead_ends": metrics["dead_ends"],
        "branch_cells": metrics["branch_cells"],
        "solution_turns": metrics["solution_turns"],
        "calls": int(cost["calls"]),
        "prompt_tokens": int(cost["prompt_tokens"]),
        "completion_tokens": int(cost["completion_tokens"]),
        "total_tokens": int(cost["total_tokens"]),
        "generation_time_sec": round(float(cost["generation_time_sec"]), 4),
        "final_moves": evaluation["moves"],
        "notes": consensus.notes,
    }


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_rows_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_experiment_grid(
    llm: Any,
    difficulties: Dict[str, Dict[str, int]],
    seeds: Sequence[int],
    agent_counts: Sequence[int] = (1, 3, 5, 10),
    methods: Sequence[str] = ("majority_vote", "debate_vote", "judge_consensus"),
    output_dir: str | Path = "results",
    debate_rounds: int = 1,
    save_mazes: bool = True,
    include_single_baseline: bool = True,
    overwrite_csv: bool = True,
) -> str:
    output_dir = Path(output_dir)
    rows: List[Dict[str, Any]] = []
    csv_path = output_dir / "experiment_results.csv"
    if overwrite_csv and csv_path.exists():
        csv_path.unlink()

    for difficulty_name, config in difficulties.items():
        for seed in seeds:
            maze = create_maze_case(
                sizex=int(config["sizex"]),
                sizey=int(config["sizey"]),
                seed=int(seed),
                difficulty_name=difficulty_name,
            )
            maze_id = f"{difficulty_name}_{maze.sizex}x{maze.sizey}_seed{seed}"

            if save_mazes:
                save_maze_json(maze, output_dir / "mazes" / f"{maze_id}.json")
                save_maze_image(maze, output_dir / "mazes" / f"{maze_id}.png", show_solution=False)

            run_plan: List[Tuple[str, int]] = []
            if include_single_baseline and 1 in agent_counts:
                run_plan.append(("single", 1))
            for agent_count in agent_counts:
                if agent_count <= 1:
                    continue
                for method in methods:
                    run_plan.append((method, int(agent_count)))

            for method, agent_count in run_plan:
                experiment_id = f"{maze_id}_{method}_n{agent_count}"
                print(f"Running {experiment_id}")
                try:
                    consensus = run_consensus_method(
                        maze,
                        llm,
                        method=method,
                        agent_count=agent_count,
                        debate_rounds=debate_rounds,
                    )
                    row = run_to_row(maze, consensus, experiment_id=experiment_id)
                    transcript = {
                        "experiment_id": experiment_id,
                        "maze": asdict(maze),
                        "consensus": {
                            "method": consensus.method,
                            "agent_count": consensus.agent_count,
                            "final_candidate": asdict(consensus.final_candidate),
                            "cost": consensus.cost,
                            "answers": [asdict(a) for a in consensus.answers],
                            "judge_answer": asdict(consensus.judge_answer) if consensus.judge_answer else None,
                            "notes": consensus.notes,
                        },
                        "evaluation": evaluate_candidate(maze, consensus.final_candidate),
                    }
                    write_json(output_dir / "transcripts" / f"{experiment_id}.json", transcript)
                except Exception as exc:
                    empty = ConsensusRun(
                        method=method,
                        agent_count=agent_count,
                        final_candidate=Candidate(),
                        answers=[],
                        notes="exception",
                    )
                    row = run_to_row(maze, empty, experiment_id=experiment_id, error=repr(exc))
                rows.append(row)
                append_rows_csv(csv_path, [row])

    return str(csv_path)


def summarize_results_csv(csv_path: str | Path, output_path: str | Path | None = None):
    import pandas as pd

    df = pd.read_csv(csv_path)
    summary = (
        df.groupby(["difficulty", "method", "agent_count"], as_index=False)
        .agg(
            trials=("experiment_id", "count"),
            accuracy=("success", "mean"),
            valid_rate=("valid_steps", "mean"),
            optimal_rate=("optimal", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            avg_time_sec=("generation_time_sec", "mean"),
            avg_path_length=("path_length", "mean"),
            avg_shortest_length=("shortest_length", "mean"),
        )
        .sort_values(["difficulty", "method", "agent_count"])
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
    return summary


def plot_accuracy_cost(csv_path: str | Path, output_dir: str | Path = "results") -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    grouped = (
        df.groupby(["difficulty", "method", "agent_count"], as_index=False)
        .agg(accuracy=("success", "mean"), avg_total_tokens=("total_tokens", "mean"))
    )

    for difficulty in grouped["difficulty"].unique():
        sub = grouped[grouped["difficulty"] == difficulty]
        plt.figure(figsize=(7, 4))
        for method in sub["method"].unique():
            m = sub[sub["method"] == method]
            plt.plot(m["agent_count"], m["accuracy"], marker="o", label=method)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Number of agents")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy by agent count - {difficulty}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"accuracy_{difficulty}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        for method in sub["method"].unique():
            m = sub[sub["method"] == method]
            plt.plot(m["agent_count"], m["avg_total_tokens"], marker="o", label=method)
        plt.xlabel("Number of agents")
        plt.ylabel("Average total tokens")
        plt.title(f"Cost by agent count - {difficulty}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"cost_{difficulty}.png", dpi=180)
        plt.close()
