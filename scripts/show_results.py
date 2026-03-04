"""
Display recall-under-noise experiment results in a human-readable format.

Usage:
    python scripts/show_results.py                  # latest result file
    python scripts/show_results.py <path-to-json>   # specific file
"""

import json
import sys
from pathlib import Path


BAR_WIDTH = 40


def find_latest_result() -> Path:
    results_dir = Path(__file__).resolve().parents[1] / "test_results"
    files = sorted(results_dir.glob("recall_under_noise_*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        print("No result files found in test_results/")
        sys.exit(1)
    return files[-1]


def bar(recall: float) -> str:
    filled = round(recall * BAR_WIDTH)
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def color(recall: float) -> str:
    if recall >= 0.9:
        return "\033[92m"   # green
    elif recall >= 0.6:
        return "\033[93m"   # yellow
    else:
        return "\033[91m"   # red


RESET = "\033[0m"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_result()

    with open(path) as f:
        data = json.load(f)

    cfg = data["config"]
    results = data["results"]

    print()
    print("=" * 60)
    print(" Recall-Under-Noise Results")
    print("=" * 60)
    print(f"  File        : {path.name}")
    print(f"  Vector mode : {cfg['vector_mode']}")
    print(f"  Records     : {cfg['n_people']}")
    print(f"  HDC dim     : {cfg['hdim']}")
    print(f"  Seed        : {cfg['seed']}")
    print("=" * 60)
    print()
    print(f"  {'Noise':>6}  {'Recall@1':>8}  {'Hits':>10}  {'':2}  Chart")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}  {'':2}  {'-'*BAR_WIDTH}")

    prev_recall = None
    for r in results:
        noise = r["noise_level"]
        recall = r["recall_at_1"]
        hits = r["hits"]
        total = r["total"]

        marker = ""
        if prev_recall is not None and prev_recall >= 0.9 and recall < 0.9:
            marker = " <- drops below 90%"
        elif prev_recall is not None and prev_recall >= 0.6 and recall < 0.6:
            marker = " <- drops below 60%"

        c = color(recall)
        print(
            f"  {noise:>6.0%}  {c}{recall:>8.1%}{RESET}  "
            f"{hits:>5}/{total:<5}  "
            f"  {c}{bar(recall)}{RESET}{marker}"
        )
        prev_recall = recall

    # Summary
    tolerable = [r for r in results if r["recall_at_1"] >= 0.9]
    break_point = next((r for r in results if r["recall_at_1"] < 0.9), None)

    print()
    print("-" * 60)
    if tolerable:
        max_tolerable = max(r["noise_level"] for r in tolerable)
        print(f"  Recall stays >= 90% up to {max_tolerable:.0%} noise.")
    if break_point:
        print(f"  First drop below 90%: noise={break_point['noise_level']:.0%}  recall={break_point['recall_at_1']:.1%}")
    print("-" * 60)
    print()


if __name__ == "__main__":
    main()
