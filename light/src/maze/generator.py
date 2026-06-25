"""
Maze generation module.
Walls[y][x] = [left, top, right, bottom] — 1 means wall exists, 0 means open.
"""

import random
from collections import deque


def generate(sizex: int, sizey: int) -> list:
    """DFS-based perfect maze generation."""
    Walls = [[[1, 1, 1, 1] for _ in range(sizex)] for _ in range(sizey)]

    visited = [[False] * sizex for _ in range(sizey)]
    visited[0][0] = True
    stack = [(0, 0)]

    while stack:
        x, y = stack[-1]
        options = []
        if x > 0 and not visited[y][x - 1]:
            options.append(0)  # left
        if y < sizey - 1 and not visited[y + 1][x]:
            options.append(1)  # top
        if x < sizex - 1 and not visited[y][x + 1]:
            options.append(2)  # right
        if y > 0 and not visited[y - 1][x]:
            options.append(3)  # bottom

        if not options:
            stack.pop()
            continue

        r = random.choice(options)
        if r == 0:
            nx, ny = x - 1, y
            Walls[y][x][0] = 0
            Walls[ny][nx][2] = 0
        elif r == 1:
            nx, ny = x, y + 1
            Walls[y][x][1] = 0
            Walls[ny][nx][3] = 0
        elif r == 2:
            nx, ny = x + 1, y
            Walls[y][x][2] = 0
            Walls[ny][nx][0] = 0
        else:
            nx, ny = x, y - 1
            Walls[y][x][3] = 0
            Walls[ny][nx][1] = 0

        visited[ny][nx] = True
        stack.append((nx, ny))

    return Walls


def find_farthest(Walls: list, sizex: int, sizey: int, start: list) -> list:
    """BFS to find the cell farthest from start (used as goal)."""
    sx, sy = start
    dist = [[-1] * sizex for _ in range(sizey)]
    dist[sy][sx] = 0
    q = deque([(sx, sy)])

    while q:
        x, y = q.popleft()
        d = dist[y][x]
        neighbors = []
        if Walls[y][x][0] == 0 and x > 0:
            neighbors.append((x - 1, y))
        if Walls[y][x][1] == 0 and y < sizey - 1:
            neighbors.append((x, y + 1))
        if Walls[y][x][2] == 0 and x < sizex - 1:
            neighbors.append((x + 1, y))
        if Walls[y][x][3] == 0 and y > 0:
            neighbors.append((x, y - 1))
        for nx, ny in neighbors:
            if dist[ny][nx] == -1:
                dist[ny][nx] = d + 1
                q.append((nx, ny))

    best, best_d = start, -1
    for y in range(sizey):
        for x in range(sizex):
            if dist[y][x] > best_d:
                best_d = dist[y][x]
                best = [x, y]

    return best, best_d


def carve_entrance_exit(Walls: list, start: list, goal: list) -> list:
    """Open the boundary walls at start (left) and goal (right)."""
    sx, sy = start
    gx, gy = goal
    Walls[sy][sx][0] = 0   # start: left wall open
    Walls[gy][gx][2] = 0   # goal:  right wall open
    return Walls


def count_junctions(Walls: list, sizex: int, sizey: int) -> int:
    """Count cells with 3+ open directions (branching points)."""
    count = 0
    for y in range(sizey):
        for x in range(sizex):
            open_dirs = 4 - sum(Walls[y][x])
            if open_dirs >= 3:
                count += 1
    return count


def build_maze(sizex: int, sizey: int, seed: int) -> dict:
    """Full pipeline: generate → find goal → carve → compute stats."""
    random.seed(seed)
    Walls = generate(sizex, sizey)

    start = [0, 0]
    goal, path_length = find_farthest(Walls, sizex, sizey, start)
    Walls = carve_entrance_exit(Walls, start, goal)

    junctions = count_junctions(Walls, sizex, sizey)

    return {
        "sizex": sizex,
        "sizey": sizey,
        "seed": seed,
        "start": start,
        "goal": goal,
        "path_length": path_length,
        "junctions": junctions,
        "Walls": Walls,
    }
