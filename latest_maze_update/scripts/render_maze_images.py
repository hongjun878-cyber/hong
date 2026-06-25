import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.maze.io import load_json, save_image


def main():
    parser = argparse.ArgumentParser(description="Render maze JSON files to PNG images")
    parser.add_argument("--index", required=True, help="Maze index JSON")
    parser.add_argument("--out_dir", default="results/maze_images", help="Output image directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of mazes to render")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.index) as f:
        entries = json.load(f)
    if args.limit is not None:
        entries = entries[: args.limit]

    for entry in entries:
        maze = load_json(entry["json_path"])
        out_path = os.path.join(args.out_dir, f"{entry['name']}.png")
        save_image(maze, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
