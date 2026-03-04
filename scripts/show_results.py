"""
Display recall-under-noise experiment results and benchmark metrics
in a human-readable format.

Usage:
    python scripts/show_results.py                  # latest result file
    python scripts/show_results.py <path-to-json>   # specific file
"""

import json
import sys
from pathlib import Path


BAR_WIDTH = 40
RESULTS_DIR = Path(__file__).resolve().parents[1] / "test_results"

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


def find_latest_recall(mode: str = None) -> Path:
    pattern = f"recall_under_noise_{mode}_*.json" if mode else "recall_under_noise_*.json"
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda f: f.stat().st_mtime)
    if not files:
        print(f"No recall result files found in test_results/")
        sys.exit(1)
    return files[-1]


def find_latest_bench(mode: str, bench_type: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(f"test_metrics_{mode}_{bench_type}_*.json"), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def bar(recall: float) -> str:
    filled = round(recall * BAR_WIDTH)
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def recall_color(recall: float) -> str:
    if recall >= 0.9:
        return GREEN
    elif recall >= 0.6:
        return YELLOW
    return RED


def print_recall_section(path: Path):
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
    print(f"  {'Noise':>6}  {'Recall@1':>8}  {'Hits':>10}    Chart")
    print(f"  {'-'*6}  {'-'*8}  {'-'*10}    {'-'*BAR_WIDTH}")

    prev_recall = None
    for r in results:
        noise  = r["noise_level"]
        recall = r["recall_at_1"]
        hits   = r["hits"]
        total  = r["total"]

        marker = ""
        if prev_recall is not None and prev_recall >= 0.9 and recall < 0.9:
            marker = " <- drops below 90%"
        elif prev_recall is not None and prev_recall >= 0.6 and recall < 0.6:
            marker = " <- drops below 60%"

        c = recall_color(recall)
        print(
            f"  {noise:>6.0%}  {c}{recall:>8.1%}{RESET}  "
            f"{hits:>5}/{total:<5}    "
            f"{c}{bar(recall)}{RESET}{marker}"
        )
        prev_recall = recall

    tolerable   = [r for r in results if r["recall_at_1"] >= 0.9]
    break_point = next((r for r in results if r["recall_at_1"] < 0.9), None)

    print()
    print("-" * 60)
    if tolerable:
        max_tolerable = max(r["noise_level"] for r in tolerable)
        print(f"  Recall stays >= 90% up to {max_tolerable:.0%} noise.")
    if break_point:
        print(f"  First drop below 90%: noise={break_point['noise_level']:.0%}  recall={break_point['recall_at_1']:.1%}")
    print("-" * 60)

    return cfg["vector_mode"]


def print_bench_section(mode: str):
    bench_types = [
        ("encoding",     "Encoding",      "tiempo_encoding_total", "s"),
        ("insert",       "Insert",        "tiempo_insercion_bd",   "s"),
        ("insert_batch", "Insert (batch)", "tiempo_insercion_bd",  "s"),
        ("search",       "Search",        "tiempo_busqueda_bd",    "ms"),
    ]

    rows = []
    for bench_type, label, build_key, build_unit in bench_types:
        path = find_latest_bench(mode, bench_type)
        if path is None:
            continue
        with open(path) as f:
            d = json.load(f)
        build_val = d["build"].get(build_key)
        q = d["query"]
        rows.append({
            "label":      label,
            "build_val":  build_val,
            "build_unit": build_unit,
            "p50":        q["latencia_p50"],
            "p95":        q["latencia_p95"],
            "p99":        q["latencia_p99"],
            "n":          q["total_queries"],
            "file":       path.name,
        })

    if not rows:
        return

    print()
    print("=" * 60)
    print(f" Benchmark Metrics  (mode: {mode})")
    print("=" * 60)
    print(f"  {'Benchmark':<16}  {'Build time':>12}  {'p50 (ms)':>10}  {'p95 (ms)':>10}  {'p99 (ms)':>10}  {'Queries':>8}")
    print(f"  {'-'*16}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for r in rows:
        if r["build_val"] is not None:
            build_str = f"{r['build_val']:.4f} {r['build_unit']}"
        else:
            build_str = "—"
        print(
            f"  {r['label']:<16}  {build_str:>12}  "
            f"{r['p50']:>10.2f}  {r['p95']:>10.2f}  {r['p99']:>10.2f}  {r['n']:>8}"
        )

    print("-" * 60)
    print()


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_recall()
    mode = print_recall_section(path)
    print_bench_section(mode)


if __name__ == "__main__":
    main()
