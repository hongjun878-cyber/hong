"""
MazeEnv — single shared environment for multi-agent maze navigation.

All agents observe the same partial view centered on the current position.
The group moves as one unit; agents vote or deliberate each step.

Coordinate system
-----------------
  x: column (0 = leftmost)
  y: row    (0 = bottom)

Walls[y][x] = [left, top, right, bottom]  (1 = wall, 0 = open)

Actions
-------
  0: left   (x-1)
  1: up     (y+1)
  2: right  (x+1)
  3: down   (y-1)

Partial view
------------
  view_radius r  →  (2r+1) × (2r+1) window centered on current cell.
  Cells outside the maze boundary are marked as "wall on all sides".
  view_radius is derived from view_ratio:  r = max(1, round(min(sizex,sizey) * view_ratio / 2))
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from src.maze.io import load_json


ACTION_NAMES = {0: "left", 1: "up", 2: "right", 3: "down"}
ACTION_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


@dataclass
class StepRecord:
    step: int
    position: list
    action: int
    action_name: str
    valid: bool          # False = tried to walk into a wall
    reached_goal: bool
    timestamp: float
    decision_time_s: float   # time agents spent deliberating this step
    votes: Optional[dict] = None   # {action: count}  (majority voting)
    discussion_rounds: Optional[int] = None   # (deliberation mode)


@dataclass
class EpisodeResult:
    maze_name: str
    sizex: int
    sizey: int
    n_agents: int
    decision_mode: str        # "majority" | "deliberation"
    view_ratio: float
    success: bool
    steps_taken: int
    invalid_steps: int        # walked into wall attempts
    total_decision_time_s: float
    total_wall_time_s: float
    deadlock_count: int       # consecutive steps with no progress
    history: list = field(default_factory=list)


class MazeEnv:
    """
    Shared maze environment for N agents.

    Parameters
    ----------
    maze_path : str
        Path to a maze JSON file produced by maze_io.save_json.
    view_ratio : float
        Fraction of the smaller maze dimension used as view window size.
        E.g. 0.3 on a 10×10 maze → radius 1 → 3×3 window.
    max_steps : int
        Episode terminates (failure) after this many steps.
    deadlock_threshold : int
        Consecutive steps that revisit a recently-seen position before
        the step is counted as a deadlock event.
    """

    DEADLOCK_WINDOW = 6   # last N positions to check for cycling

    def __init__(
        self,
        maze_path: str,
        view_ratio: float = 0.3,
        max_steps: int = 500,
        deadlock_threshold: int = 6,
    ):
        self.maze = load_json(maze_path)
        self.sizex = self.maze["sizex"]
        self.sizey = self.maze["sizey"]
        self.Walls = self.maze["Walls"]
        self.start = list(self.maze["start"])
        self.goal  = list(self.maze["goal"])

        self.view_ratio = view_ratio
        r = max(1, round(min(self.sizex, self.sizey) * view_ratio / 2))
        self.view_radius = r

        self.max_steps = max_steps
        self.deadlock_threshold = deadlock_threshold

        self._pos: list = list(self.start)
        self._step: int = 0
        self._done: bool = False
        self._history: list = []
        self._pos_history: list = []
        self._invalid_steps: int = 0
        self._deadlock_count: int = 0
        self._episode_start: float = 0.0
        self._total_decision_time: float = 0.0
        self._visited: set = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset to start. Returns the initial observation."""
        self._pos = list(self.start)
        self._step = 0
        self._done = False
        self._history = []
        self._pos_history = []
        self._invalid_steps = 0
        self._deadlock_count = 0
        self._episode_start = time.time()
        self._total_decision_time = 0.0
        self._visited = {tuple(self.start)}
        return self._observe()

    def step(self, action: int, decision_time_s: float = 0.0,
             votes: dict = None, discussion_rounds: int = None) -> tuple:
        """
        Apply action chosen by agents.

        Parameters
        ----------
        action : int  (0=left, 1=up, 2=right, 3=down)
        decision_time_s : float  deliberation time for this step
        votes : dict  {action_int: vote_count}  (majority mode)
        discussion_rounds : int  (deliberation mode)

        Returns
        -------
        obs : dict       new observation
        reward : float
        done : bool
        info : dict
        """
        if self._done:
            raise RuntimeError("Episode is over. Call reset() first.")

        self._total_decision_time += decision_time_s

        x, y = self._pos
        dx, dy = ACTION_DELTA[action]
        wall_idx = action  # wall index matches action: 0=left,1=top,2=right,3=bottom

        valid = self.Walls[y][x][wall_idx] == 0

        if valid:
            nx, ny = x + dx, y + dy
            # stay in bounds (carve_entrance_exit may open boundary walls)
            nx = max(0, min(self.sizex - 1, nx))
            ny = max(0, min(self.sizey - 1, ny))
            self._pos = [nx, ny]
        else:
            self._invalid_steps += 1

        reached_goal = (self._pos == self.goal)
        self._step += 1

        # deadlock detection: cycling over recent positions
        self._pos_history.append(tuple(self._pos))
        if len(self._pos_history) >= self.DEADLOCK_WINDOW:
            window = self._pos_history[-self.DEADLOCK_WINDOW:]
            unique = len(set(window))
            if unique <= self.DEADLOCK_WINDOW // 3:
                self._deadlock_count += 1

        self._visited.add(tuple(self._pos))

        record = StepRecord(
            step=self._step,
            position=list(self._pos),
            action=action,
            action_name=ACTION_NAMES[action],
            valid=valid,
            reached_goal=reached_goal,
            timestamp=time.time(),
            decision_time_s=decision_time_s,
            votes=votes,
            discussion_rounds=discussion_rounds,
        )
        self._history.append(record)

        reward = 1.0 if reached_goal else (-0.01 if valid else -0.1)

        if reached_goal or self._step >= self.max_steps:
            self._done = True

        obs = self._observe()
        info = {
            "step": self._step,
            "position": list(self._pos),
            "valid_move": valid,
            "reached_goal": reached_goal,
            "deadlock_count": self._deadlock_count,
            "visited_cells": len(self._visited),
        }
        return obs, reward, self._done, info

    def get_result(self, n_agents: int, decision_mode: str) -> EpisodeResult:
        """Summarize the finished episode."""
        success = bool(self._history and self._history[-1].reached_goal)
        return EpisodeResult(
            maze_name=f"{self.sizex}x{self.sizey}_seed{self.maze['seed']}",
            sizex=self.sizex,
            sizey=self.sizey,
            n_agents=n_agents,
            decision_mode=decision_mode,
            view_ratio=self.view_ratio,
            success=success,
            steps_taken=self._step,
            invalid_steps=self._invalid_steps,
            total_decision_time_s=round(self._total_decision_time, 4),
            total_wall_time_s=round(time.time() - self._episode_start, 4),
            deadlock_count=self._deadlock_count,
            history=self._history,
        )

    def render_text(self) -> str:
        """ASCII render of the full maze with current position marked."""
        return _render_ascii(
            self.Walls, self.sizex, self.sizey,
            agent_pos=self._pos, goal=self.goal,
        )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _observe(self) -> dict:
        """
        Build the shared observation dict delivered to every agent.

        Fields
        ------
        position        : [x, y]  current cell
        goal            : [x, y]
        view_radius     : int
        view            : 2-D list of cell dicts  (see _cell_view)
        available_moves : list of valid action ints from current cell
        step            : int
        maze_size       : [sizex, sizey]
        visited_ratio   : float  fraction of maze cells visited so far
        """
        x, y = self._pos
        r = self.view_radius

        view = []
        for vy in range(y - r, y + r + 1):
            row = []
            for vx in range(x - r, x + r + 1):
                row.append(self._cell_view(vx, vy))
            view.append(row)

        available = [
            a for a in range(4)
            if self.Walls[y][x][a] == 0
        ]

        # unvisited moves: available directions that lead to unvisited cells
        unvisited_moves = []
        for a in available:
            dx, dy = ACTION_DELTA[a]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.sizex and 0 <= ny < self.sizey:
                if (nx, ny) not in self._visited:
                    unvisited_moves.append(a)

        # recent trajectory (last 6 positions) for cycle warning
        recent = self._pos_history[-6:] if self._pos_history else []

        return {
            "position":        list(self._pos),
            "goal":            list(self.goal),
            "view_radius":     r,
            "view":            view,
            "available_moves": available,
            "unvisited_moves": unvisited_moves,
            "step":            self._step,
            "maze_size":       [self.sizex, self.sizey],
            "visited_ratio":   round(len(self._visited) / (self.sizex * self.sizey), 3),
            "recent_path":     [list(p) for p in recent],
            "deadlock_count":  self._deadlock_count,
        }

    def _cell_view(self, x: int, y: int) -> dict:
        """Return wall info for cell (x,y); marks out-of-bounds as fully walled."""
        if 0 <= x < self.sizex and 0 <= y < self.sizey:
            w = self.Walls[y][x]
            return {
                "x": x, "y": y,
                "in_bounds": True,
                "walls": {"left": w[0], "top": w[1], "right": w[2], "bottom": w[3]},
                "is_current": [x, y] == self._pos,
                "is_goal":    [x, y] == self.goal,
                "is_start":   [x, y] == self.start,
                "visited":    (x, y) in self._visited,
            }
        return {
            "x": x, "y": y,
            "in_bounds": False,
            "walls": {"left": 1, "top": 1, "right": 1, "bottom": 1},
            "is_current": False,
            "is_goal": False,
            "is_start": False,
            "visited": False,
        }


# ------------------------------------------------------------------
# Observation → text prompt  (used by LLM agents)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Agent personas (비판 2 해결)
# 같은 관찰에도 서로 다른 전략적 편향을 부여해 응답 다양성을 확보
# ------------------------------------------------------------------

AGENT_PERSONAS = [
    # 0: Goal-seeker — 목표 방향을 최우선
    (
        "You are a Goal-seeker agent. "
        "Always prioritize moves that bring you closer to the goal position. "
        "Only deviate when blocked."
    ),
    # 1: Explorer — 미방문 셀 탐색 우선
    (
        "You are an Explorer agent. "
        "Always prefer moves toward unvisited cells to map the maze efficiently. "
        "Avoid revisiting cells unless necessary."
    ),
    # 2: Backtracker — 막힌 경우 적극 역추적
    (
        "You are a Backtracker agent. "
        "When all adjacent cells are visited, immediately backtrack along the path "
        "you came from to find a new branch. Avoid cycling."
    ),
    # 3: Balanced Strategist
    (
        "You are a Balanced Strategist agent. "
        "Compare goal direction, unvisited cells, legal moves, and recent cycling risk. "
        "Choose the move with the best overall tradeoff, and avoid extreme strategies."
    ),
]

AGENT_PERSONA_NAMES = [
    "goal_seeker",
    "explorer",
    "backtracker",
    "balanced_strategist",
]

def get_persona(agent_id: int) -> str:
    return AGENT_PERSONAS[agent_id % len(AGENT_PERSONAS)]


def get_persona_name(agent_id: int) -> str:
    return AGENT_PERSONA_NAMES[agent_id % len(AGENT_PERSONA_NAMES)]


def obs_to_prompt(obs: dict, agent_id: int = None, shared_map: dict = None) -> str:
    """
    Convert an observation dict into a natural-language prompt.

    Parameters
    ----------
    agent_id   : if given, prepends the agent's persona (비판 2 해결)
    shared_map : dict mapping (x,y) tuple → visit_count across the full episode
                 (비판 3 해결: 에피소드 전체 공유 탐색 지도)
    """
    x, y = obs["position"]
    gx, gy = obs["goal"]
    r = obs["view_radius"]
    avail      = [ACTION_NAMES[a] for a in obs["available_moves"]]
    unvisited  = [ACTION_NAMES[a] for a in obs.get("unvisited_moves", [])]
    deadlocks  = obs.get("deadlock_count", 0)
    recent     = obs.get("recent_path", [])

    # goal direction hint
    dx_goal = gx - x
    dy_goal = gy - y
    hint_parts = []
    if dx_goal > 0:
        hint_parts.append("right")
    elif dx_goal < 0:
        hint_parts.append("left")
    if dy_goal > 0:
        hint_parts.append("up")
    elif dy_goal < 0:
        hint_parts.append("down")
    goal_hint = "→ " + " and ".join(hint_parts) if hint_parts else "→ you are aligned"

    lines = []

    # 비판 2: 페르소나 삽입
    if agent_id is not None:
        lines.append(f"[Your role] {get_persona(agent_id)}")
        lines.append("")

    lines += [
        f"You are navigating a {obs['maze_size'][0]}×{obs['maze_size'][1]} maze.",
        f"Current position : ({x}, {y})",
        f"Goal position    : ({gx}, {gy})  [{goal_hint}]",
        f"Step             : {obs['step']}  |  Visited: {obs['visited_ratio']*100:.1f}%",
    ]

    if deadlocks > 0:
        lines.append(
            f"⚠ CYCLING DETECTED ({deadlocks} times). "
            "You are going in circles. You MUST choose a different direction."
        )
    if recent:
        lines.append(f"Recent path: {' → '.join(str(tuple(p)) for p in recent)}")

    # 비판 3: 공유 탐색 지도
    if shared_map:
        neighbor_info = []
        for a in obs["available_moves"]:
            dx, dy = ACTION_DELTA[a]
            nx, ny = x + dx, y + dy
            cnt = shared_map.get((nx, ny), 0)
            neighbor_info.append(f"{ACTION_NAMES[a]}→({nx},{ny}) visited {cnt}x")
        if neighbor_info:
            lines.append(f"Shared visit counts for neighbors: {', '.join(neighbor_info)}")
        # highlight least-visited neighbors
        if shared_map:
            hotspots = sorted(
                [(pos, cnt) for pos, cnt in shared_map.items() if cnt >= 3],
                key=lambda x: -x[1]
            )[:5]
            if hotspots:
                hotspot_str = ", ".join(f"({p[0]},{p[1]})×{c}" for p, c in hotspots)
                lines.append(f"Most revisited cells (avoid): {hotspot_str}")

    lines += [
        "",
        f"You can see a {2*r+1}×{2*r+1} window. Legend: ★=you  G=goal  S=start  ·=visited",
        "Cell format: (x,y)[open directions]  L=left U=up R=right D=down",
        "",
        "--- LOCAL VIEW ---",
    ]

    view = obs["view"]
    for row in reversed(view):
        cells = []
        for cell in row:
            if not cell["in_bounds"]:
                cells.append("[OUT]")
                continue
            open_dirs = [
                d for d, k in [("L","left"),("U","top"),("R","right"),("D","bottom")]
                if cell["walls"][k] == 0
            ]
            if cell["is_current"]:
                tag = "★"
            elif cell["is_goal"]:
                tag = "G"
            elif cell["is_start"]:
                tag = "S"
            elif cell["visited"]:
                tag = "·"
            else:
                tag = ""
            cells.append(f"({cell['x']},{cell['y']}){tag}[{''.join(open_dirs)}]")
        lines.append("  " + "  ".join(cells))

    lines += [""]

    if unvisited:
        lines.append(f"Moves to UNVISITED cells (prefer these): {unvisited}")
    else:
        lines.append("All adjacent open cells already visited — pick the best backtrack.")

    lines += [
        f"All available moves: {avail}",
        "",
        "Strategy: prefer unvisited cells. Avoid repeating recent moves.",
        "Reply with exactly one word: left / up / right / down",
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------
# ASCII full-maze renderer (debug / logging)
# ------------------------------------------------------------------

def _render_ascii(Walls, sizex, sizey, agent_pos=None, goal=None) -> str:
    lines = []
    for y in range(sizey - 1, -1, -1):
        top_row = ""
        mid_row = ""
        for x in range(sizex):
            top_wall  = "---" if Walls[y][x][1] else "   "
            left_wall = "|"  if Walls[y][x][0] else " "
            if agent_pos and [x, y] == agent_pos:
                cell = " A "
            elif goal and [x, y] == goal:
                cell = " G "
            else:
                cell = "   "
            top_row += f"+{top_wall}"
            mid_row += f"{left_wall}{cell}"
        lines.append(top_row + "+")
        right_wall = "|" if Walls[y][sizex - 1][2] else " "
        lines.append(mid_row + right_wall)

    # bottom border
    bottom = ""
    for x in range(sizex):
        bottom_wall = "---" if Walls[0][x][3] else "   "
        bottom += f"+{bottom_wall}"
    lines.append(bottom + "+")

    return "\n".join(lines)
