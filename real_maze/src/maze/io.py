"""
Maze serialization: JSON save/load and PNG image export.
"""

import json
import os


def to_dict(maze: dict) -> dict:
    """Convert maze dict (with raw Walls) to JSON-serializable form."""
    Walls = maze["Walls"]
    sizex, sizey = maze["sizex"], maze["sizey"]

    grid = []
    for y in range(sizey):
        row = []
        for x in range(sizex):
            row.append({
                "x": x,
                "y": y,
                "walls": {
                    "left":   Walls[y][x][0],
                    "top":    Walls[y][x][1],
                    "right":  Walls[y][x][2],
                    "bottom": Walls[y][x][3],
                },
            })
        grid.append(row)

    return {
        "type": "maze",
        "sizex": sizex,
        "sizey": sizey,
        "seed": maze["seed"],
        "start": maze["start"],
        "goal": maze["goal"],
        "path_length": maze["path_length"],
        "junctions": maze["junctions"],
        "grid": grid,
    }


def from_dict(data: dict) -> dict:
    """Reconstruct maze dict (with raw Walls) from saved JSON."""
    sizex, sizey = data["sizex"], data["sizey"]
    Walls = [[[0, 0, 0, 0] for _ in range(sizex)] for _ in range(sizey)]
    for row in data["grid"]:
        for cell in row:
            x, y = cell["x"], cell["y"]
            w = cell["walls"]
            Walls[y][x] = [w["left"], w["top"], w["right"], w["bottom"]]
    return {
        "sizex": sizex,
        "sizey": sizey,
        "seed": data["seed"],
        "start": data["start"],
        "goal": data["goal"],
        "path_length": data["path_length"],
        "junctions": data["junctions"],
        "Walls": Walls,
    }


def save_json(maze: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_dict(maze), f, indent=2)


def load_json(path: str) -> dict:
    with open(path) as f:
        return from_dict(json.load(f))


def save_image(maze: dict, path: str, show_solution: list = None) -> None:
    """
    Render maze walls to PNG.
    show_solution: optional list of (x, y) tuples for the solution path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Walls = maze["Walls"]
    sizex, sizey = maze["sizex"], maze["sizey"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    for y in range(sizey):
        for x in range(sizex):
            if Walls[y][x][1] == 1:  # top
                ax.plot([x, x + 1], [y + 1, y + 1], color="black", lw=1.5)
            if Walls[y][x][3] == 1:  # bottom
                ax.plot([x, x + 1], [y, y], color="black", lw=1.5)
            if Walls[y][x][0] == 1:  # left
                ax.plot([x, x], [y, y + 1], color="black", lw=1.5)
            if Walls[y][x][2] == 1:  # right
                ax.plot([x + 1, x + 1], [y, y + 1], color="black", lw=1.5)

    # start / goal markers
    sx, sy = maze["start"]
    gx, gy = maze["goal"]
    ax.plot(sx + 0.5, sy + 0.5, "go", markersize=10, label="start")
    ax.plot(gx + 0.5, gy + 0.5, "r*", markersize=14, label="goal")

    if show_solution:
        xs = [p[0] + 0.5 for p in show_solution]
        ys = [p[1] + 0.5 for p in show_solution]
        ax.plot(xs, ys, color="blue", lw=1.5, alpha=0.6, label="solution")

    ax.legend(loc="upper right", fontsize=8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
