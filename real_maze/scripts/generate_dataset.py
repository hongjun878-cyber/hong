"""
Maze dataset generation pipeline — fixed size, junction-stratified.

크기를 고정(기본 10×10)하고 junction 수 기준으로 난이도를 층화(stratify)한다.
이렇게 해야 미로 크기 효과 없이 junction(복잡도) 효과만 순수하게 측정할 수 있다.

Junction 난이도 tier (10×10 기준):
    low    :  6 –  8   (단순한 미로)
    medium :  9 – 11   (중간)
    high   : 12 – 14   (복잡한 미로)

Usage:
    python generate_dataset.py                        # default config
    python generate_dataset.py --size 10 --per_tier 10
    python generate_dataset.py --config cfg.json
    python generate_dataset.py --size 10 --no_image
"""

import argparse
import json
import os
import csv
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.maze.generator import build_maze
from src.maze.io import save_json, save_image


# Junction tier boundaries — tuned for 10×10 (range 6–14)
# Adjust if using a different fixed size.
JUNCTION_TIERS = {
    "low":    (6,  8),
    "medium": (9,  11),
    "high":   (12, 14),
}

DEFAULT_CONFIG = {
    "size": 10,              # fixed maze size (square)
    "per_tier": 10,          # mazes to collect per tier
    "max_search_seeds": 500, # seed search budget (stop when all tiers filled)
    "out_dir": "data/mazes",
    "save_images": True,
}


def _tier(junctions: int):
    """Return tier name for a junction count, or None if outside all tiers."""
    for name, (lo, hi) in JUNCTION_TIERS.items():
        if lo <= junctions <= hi:
            return name
    return None


def generate_dataset(config: dict) -> list:
    size        = config["size"]
    per_tier    = config["per_tier"]
    max_search  = config.get("max_search_seeds", 500)
    out_dir     = config["out_dir"]
    save_img    = config.get("save_images", True)

    size_dir = os.path.join(out_dir, f"{size}x{size}")
    os.makedirs(size_dir, exist_ok=True)

    tier_buckets = {t: [] for t in JUNCTION_TIERS}
    records      = []
    seed         = 0

    print(f"Generating {size}×{size} mazes  ({per_tier} per tier, tiers={list(JUNCTION_TIERS)})")
    print(f"Junction ranges: { {t: r for t, r in JUNCTION_TIERS.items()} }")
    print()

    while seed < max_search and any(len(tier_buckets[t]) < per_tier for t in tier_buckets):
        maze = build_maze(size, size, seed)
        t    = _tier(maze["junctions"])

        if t is not None and len(tier_buckets[t]) < per_tier:
            idx  = len(tier_buckets[t])
            name = f"maze_{size}x{size}_{t}_{idx:02d}"
            json_path = os.path.join(size_dir, f"{name}.json")
            img_path  = os.path.join(size_dir, f"{name}.png")

            save_json(maze, json_path)
            if save_img:
                save_image(maze, img_path)

            record = {
                "name":        name,
                "sizex":       maze["sizex"],
                "sizey":       maze["sizey"],
                "seed":        maze["seed"],
                "difficulty":  t,
                "start":       maze["start"],
                "goal":        maze["goal"],
                "path_length": maze["path_length"],
                "junctions":   maze["junctions"],
                "json_path":   json_path,
                "img_path":    img_path if save_img else "",
            }
            tier_buckets[t].append(record)
            records.append(record)
            print(
                f"  seed={seed:4d}  tier={t:6s}  junctions={maze['junctions']:2d}  "
                f"path={maze['path_length']:3d}  ({len(tier_buckets[t])}/{per_tier})"
            )

        seed += 1

    print()
    for t, bucket in tier_buckets.items():
        if len(bucket) < per_tier:
            print(f"  WARNING: tier={t} only found {len(bucket)}/{per_tier} mazes in {max_search} seeds")
        else:
            jvals = [r["junctions"] for r in bucket]
            print(f"  tier={t:6s}: {len(bucket)} mazes  junctions=[{min(jvals)}–{max(jvals)}]")

    # Save index
    csv_path = os.path.join(out_dir, "index.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nDataset saved → {out_dir}/  ({len(records)} mazes total)")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Maze dataset generator — fixed size, junction-stratified"
    )
    parser.add_argument("--config",           type=str, help="JSON config file path")
    parser.add_argument("--size",             type=int, help="Fixed maze size (default 10)")
    parser.add_argument("--per_tier",         type=int, help="Mazes per tier")
    parser.add_argument("--max_search_seeds", type=int, help="Seed search budget")
    parser.add_argument("--out_dir",          type=str, help="Output directory")
    parser.add_argument("--no_image",         action="store_true", help="Skip PNG generation")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    if args.size:             config["size"]             = args.size
    if args.per_tier:         config["per_tier"]         = args.per_tier
    if args.max_search_seeds: config["max_search_seeds"] = args.max_search_seeds
    if args.out_dir:          config["out_dir"]          = args.out_dir
    if args.no_image:         config["save_images"]      = False

    generate_dataset(config)


if __name__ == "__main__":
    main()
