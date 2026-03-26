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


def print_dedup_section(path: Path):
    with open(path) as f:
        data = json.load(f)

    cfg = data["config"]
    results = data["results"]
    top_k = cfg["top_k"]
    recall = results[f"recall_at_{top_k}"]
    hits = results["hits"]
    total = results["total"]

    print()
    print("=" * 60)
    print(" Deduplication Recall Results")
    print("=" * 60)
    print(f"  File        : {path.name}")
    print(f"  Vector mode : {cfg['vector_mode']}")
    print(f"  Identities  : {cfg['n_identities']}")
    print(f"  Variants    : {cfg['variants_per_identity']}  (total records: {cfg['total_records']})")
    print(f"  Noise       : {cfg['noise_fraction']:.0%}")
    print(f"  Top-K       : {top_k}")
    print(f"  HDC dim     : {cfg['hdim']}")
    print(f"  Seed        : {cfg['seed']}")
    print("=" * 60)
    print()

    c = recall_color(recall)
    print(f"  recall@{top_k} = {c}{recall:.1%}{RESET}  ({hits}/{total})")
    print()
    print(f"  {c}{bar(recall)}{RESET}")
    print()
    print("-" * 60)

    queries = data.get("queries", [])
    if queries:
        print()
        print(f"  Sample queries  ({len(queries)} records)")

        for q in queries:
            print()
            print(f"  ── Query {q['query_num']}/{len(queries)}  [identity {q['identity_idx']}] " + "─" * 30)
            qr = q["query_record"]
            print(f"  Searched for")
            print(f"    {qr['name']} {qr['lastname']:<20}  DOB: {qr['dob']}  |  {qr['gender']}, {qr['race']}, {qr['marital_status']}")
            print(f"    Phone: {qr['mobile_number']}")
            if qr.get("akas"):
                print(f"    AKAs: {', '.join(qr['akas'])}")
            if qr.get("addresses"):
                print(f"    Address: {qr['addresses'][0]}")
            print()
            for r in q["results"]:
                label = f"{GREEN}[MATCH]{RESET}" if r["is_match"] else f"{RED}[DIFF] {RESET}"
                rec = r["record"]
                sim = rec.get("similarity")
                sim_str = f"  similarity: {sim:.4f}" if sim is not None else ""
                print(f"  {label}  #{r['rank']}{sim_str}")
                print(f"    {rec['name']} {rec['lastname']:<20}  DOB: {rec['dob']}  |  {rec['gender']}, {rec['race']}, {rec['marital_status']}")
                print(f"    Phone: {rec['mobile_number']}")
                if rec.get("akas"):
                    print(f"    AKAs: {', '.join(rec['akas'])}")
                if rec.get("addresses"):
                    print(f"    Address: {rec['addresses'][0]}")
                print()

        print("-" * 60)


def print_field_weighting_section(path: Path):
    with open(path) as f:
        data = json.load(f)

    cfg  = data["config"]
    mode = data["mode"]
    results = data["results"]
    top_k = cfg["top_k"]

    print()
    print("=" * 60)
    print(" Field Weighting Ablation Results")
    print("=" * 60)
    print(f"  File        : {path.name}")
    print(f"  Vector mode : {mode}")
    print(f"  Identities  : {cfg['n_identities']}  x  {cfg['variants_per_identity']} variants")
    print(f"  Noise       : {cfg['noise_fraction']:.0%}")
    print(f"  Top-K       : {top_k}")
    print(f"  HDC dim     : {cfg['hdim']}")
    print(f"  Seed        : {cfg['seed']}")
    print("=" * 60)
    print()
    print(f"  {'Variant':<22}  {'Recall@' + str(top_k):>8}  {'Hits':>10}    Chart")
    print(f"  {'-'*22}  {'-'*8}  {'-'*10}    {'-'*BAR_WIDTH}")

    for r in results:
        recall = r["recall_at_k"]
        hits   = r["hits"]
        total  = r["total"]
        c = recall_color(recall)
        print(
            f"  {r['variant']:<22}  {c}{recall:>8.1%}{RESET}  "
            f"{hits:>5}/{total:<5}    "
            f"{c}{bar(recall)}{RESET}"
        )

    print("-" * 60)


def print_scalability_section(path: Path):
    with open(path) as f:
        data = json.load(f)

    cfg     = data["config"]
    mode    = data["mode"]
    results = data["results"]
    top_k   = cfg["top_k"]

    print()
    print("=" * 60)
    print(" Scalability Results")
    print("=" * 60)
    print(f"  File        : {path.name}")
    print(f"  Vector mode : {mode}")
    print(f"  N values    : {cfg['n_values']}")
    print(f"  Variants    : {cfg['variants_per_identity']} per identity")
    print(f"  Noise       : {cfg['noise_fraction']:.0%}")
    print(f"  Top-K       : {top_k}")
    print(f"  HDC dim     : {cfg['hdim']}")
    print(f"  Seed        : {cfg['seed']}")
    print("=" * 60)
    print()

    col_n      = 7
    col_total  = 14
    col_recall = 10
    col_hits   = 12
    col_insert = 10
    col_query  = 10
    col_chart  = BAR_WIDTH

    print(
        f"  {'N':>{col_n}}  "
        f"{'Total Records':>{col_total}}  "
        f"{'Recall@' + str(top_k):>{col_recall}}  "
        f"{'Hits':>{col_hits}}  "
        f"{'Insert(s)':>{col_insert}}  "
        f"{'Query(s)':>{col_query}}  "
        f"Chart"
    )
    print(
        f"  {'-'*col_n}  "
        f"{'-'*col_total}  "
        f"{'-'*col_recall}  "
        f"{'-'*col_hits}  "
        f"{'-'*col_insert}  "
        f"{'-'*col_query}  "
        f"{'-'*col_chart}"
    )

    for r in results:
        recall = r["recall_at_k"]
        hits   = r["hits"]
        total  = r["total"]
        c = recall_color(recall)
        print(
            f"  {r['n']:>{col_n}}  "
            f"{r['total_records']:>{col_total}}  "
            f"{c}{recall:>{col_recall}.1%}{RESET}  "
            f"{hits:>5}/{total:<{col_hits - 6}}  "
            f"{r['insert_time_s']:>{col_insert}.2f}  "
            f"{r['query_time_s']:>{col_query}.2f}  "
            f"{c}{bar(recall)}{RESET}"
        )

    print("-" * 60)


def print_ranking_section(path: Path):
    with open(path) as f:
        data = json.load(f)

    cfg     = data["config"]
    mode    = data["mode"]
    results = data["results"]
    top_k   = cfg["top_k"]

    recall   = results["recall_at_k"]
    mrr      = results["mrr"]
    prec     = results["precision_at_k"]
    hit1     = results["hit_at_1"]
    total    = results["total_queries"]

    print()
    print("=" * 60)
    print(" Ranking Metrics Results")
    print("=" * 60)
    print(f"  File        : {path.name}")
    print(f"  Vector mode : {mode}")
    print(f"  Identities  : {cfg['n_identities']}  x  {cfg['variants_per_identity']} variants")
    print(f"  Noise       : {cfg['noise_fraction']:.0%}")
    print(f"  Top-K       : {top_k}")
    print(f"  HDC dim     : {cfg['hdim']}")
    print(f"  Seed        : {cfg['seed']}")
    print(f"  Queries     : {total}")
    print("=" * 60)
    print()

    c_recall = recall_color(recall)
    c_hit1   = recall_color(hit1)

    print(f"  {'Metric':<16}  {'Value':>8}")
    print(f"  {'-'*16}  {'-'*8}")
    print(f"  {'recall@' + str(top_k):<16}  {c_recall}{recall:>7.1%}{RESET}")
    print(f"  {'MRR':<16}  {mrr:>8.3f}")
    print(f"  {'Precision@' + str(top_k):<16}  {prec:>7.1%}")
    print(f"  {'Hit@1':<16}  {c_hit1}{hit1:>7.1%}{RESET}")
    print()
    print("-" * 60)


def print_per_field_noise_section(path: Path):
    with open(path) as f:
        data = json.load(f)

    cfg      = data["config"]
    mode     = data["mode"]
    baseline = data["baseline"]
    results  = data["results"]   # already sorted by delta_recall descending
    top_k    = cfg["top_k"]

    print()
    print("=" * 90)
    print(" Per-Field Noise Sensitivity Results")
    print("=" * 90)
    print(f"  File        : {path.name}")
    print(f"  Vector mode : {mode}")
    print(f"  Identities  : {cfg['n_identities']}  x  {cfg['variants_per_identity']} variants")
    print(f"  Noise       : {cfg['noise_fraction']:.0%}")
    print(f"  Top-K       : {top_k}")
    print(f"  HDC dim     : {cfg['hdim']}")
    print(f"  Seed        : {cfg['seed']}")
    print("=" * 90)
    print()

    BAR_MAX = 20

    max_delta = max((r["delta_recall"] for r in results), default=0.001)

    def _delta_color(delta):
        if delta > 0.1:
            return RED
        elif delta >= 0.05:
            return YELLOW
        return GREEN

    def _delta_bar(delta):
        filled = round(abs(delta) * BAR_MAX / max(0.001, max_delta))
        filled = min(filled, BAR_MAX)
        return "#" * filled + "-" * (BAR_MAX - filled)

    print(
        f"  {'Field':<16}  {'Recall@K':>8}  {'Δ Recall':>9}  "
        f"{'MRR':>7}  {'Δ MRR':>8}  {'Hit@1':>7}  {'Δ Hit@1':>8}  Bar (Δ Recall)"
    )
    print(
        f"  {'-'*16}  {'-'*8}  {'-'*9}  "
        f"{'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*BAR_MAX}"
    )

    # Baseline row
    b_recall = baseline["recall_at_k"]
    b_mrr    = baseline["mrr"]
    b_hit1   = baseline["hit_at_1"]
    print(
        f"  {'[baseline]':<16}  {b_recall:>8.1%}  {'—':>9}  "
        f"{b_mrr:>7.3f}  {'—':>8}  {b_hit1:>7.1%}  {'—':>8}"
    )
    print(
        f"  {'-'*16}  {'-'*8}  {'-'*9}  "
        f"{'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*BAR_MAX}"
    )

    for r in results:
        c = _delta_color(r["delta_recall"])
        print(
            f"  {r['field']:<16}  {r['recall_at_k']:>8.1%}  "
            f"{c}{r['delta_recall']:>+9.3f}{RESET}  "
            f"{r['mrr']:>7.3f}  {c}{r['delta_mrr']:>+8.3f}{RESET}  "
            f"{r['hit_at_1']:>7.1%}  {c}{r['delta_hit1']:>+8.3f}{RESET}  "
            f"{c}{_delta_bar(r['delta_recall'])}{RESET}"
        )

    print("=" * 90)
    print()


def print_per_field_sweep_section(path: Path):
    with open(path) as f:
        data = json.load(f)

    cfg     = data["config"]
    mode    = data["mode"]
    results = data["results"]   # dict: field -> list of {noise_level, recall_at_k, mrr, hit_at_1}
    top_k   = cfg["top_k"]

    print()
    print("=" * 85)
    print(" Per-Field Noise Sweep Results")
    print("=" * 85)
    print(f"  File        : {path.name}")
    print(f"  Vector mode : {mode}")
    print(f"  Fields      : {', '.join(cfg['fields'])}")
    print(f"  Noise levels: {cfg['noise_levels']}")
    print(f"  Identities  : {cfg['n_identities']}  x  {cfg['variants_per_identity']} variants")
    print(f"  Top-K       : {top_k}")
    print(f"  HDC dim     : {cfg['hdim']}")
    print(f"  Seed        : {cfg['seed']}")
    print("=" * 85)

    for field, rows in results.items():
        print()
        print(f"  Field: {field}")
        print(
            f"  {'Noise%':>7}  {'Recall@' + str(top_k):>9}  {'MRR':>7}  {'Hit@1':>7}  Chart"
        )
        print(
            f"  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*BAR_WIDTH}"
        )

        for r in rows:
            recall = r["recall_at_k"]
            mrr    = r["mrr"]
            hit1   = r["hit_at_1"]
            noise  = r["noise_level"]
            c = recall_color(recall)
            print(
                f"  {noise:>6}%  "
                f"{c}{recall:>9.1%}{RESET}  "
                f"{mrr:>7.3f}  "
                f"{c}{hit1:>7.1%}{RESET}  "
                f"{c}{bar(recall)}{RESET}"
            )

        print(f"  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*BAR_WIDTH}")

    print("=" * 85)
    print()


def print_dimensionality_section(path: Path):
    with open(path) as f:
        data = json.load(f)

    cfg     = data["config"]
    mode    = data["mode"]
    results = data["results"]
    top_k   = cfg["top_k"]

    print()
    print("=" * 80)
    print(" Dimensionality Sweep Results")
    print("=" * 80)
    print(f"  File        : {path.name}")
    print(f"  Vector mode : {mode}")
    print(f"  Dim values  : {cfg['dim_values']}")
    print(f"  Identities  : {cfg['n_identities']}  x  {cfg['variants_per_identity']} variants")
    print(f"  Noise       : {cfg['noise_fraction']:.0%}")
    print(f"  Top-K       : {top_k}")
    print(f"  Seed        : {cfg['seed']}")
    print("=" * 80)
    print()

    col_dim    = 10
    col_total  = 14
    col_recall = 10
    col_mrr    =  8
    col_hit1   =  7
    col_insert = 10
    col_query  = 10

    print(
        f"  {'HDC_DIM':>{col_dim}}  "
        f"{'Total Records':>{col_total}}  "
        f"{'Recall@' + str(top_k):>{col_recall}}  "
        f"{'MRR':>{col_mrr}}  "
        f"{'Hit@1':>{col_hit1}}  "
        f"{'Insert(s)':>{col_insert}}  "
        f"{'Query(s)':>{col_query}}  "
        f"Chart"
    )
    print(
        f"  {'-'*col_dim}  "
        f"{'-'*col_total}  "
        f"{'-'*col_recall}  "
        f"{'-'*col_mrr}  "
        f"{'-'*col_hit1}  "
        f"{'-'*col_insert}  "
        f"{'-'*col_query}  "
        f"{'-'*BAR_WIDTH}"
    )

    for r in results:
        recall = r["recall_at_k"]
        c = recall_color(recall)
        print(
            f"  {r['dim']:>{col_dim}}  "
            f"{r['total_records']:>{col_total}}  "
            f"{c}{recall:>{col_recall}.1%}{RESET}  "
            f"{r['mrr']:>{col_mrr}.3f}  "
            f"{c}{r['hit_at_1']:>{col_hit1}.1%}{RESET}  "
            f"{r['insert_time_s']:>{col_insert}.2f}  "
            f"{r['query_time_s']:>{col_query}.2f}  "
            f"{c}{bar(recall)}{RESET}"
        )

    print("-" * 80)


def print_date_encoding_section(path: Path):
    with open(path) as f:
        data = json.load(f)

    cfg     = data["config"]
    mode    = data["mode"]
    results = data["results"]
    top_k   = cfg["top_k"]

    print()
    print("=" * 80)
    print(" Date Encoding Comparison Results")
    print("=" * 80)
    print(f"  File        : {path.name}")
    print(f"  Vector mode : {mode}")
    print(f"  Identities  : {cfg['n_identities']}  x  {cfg['variants_per_identity']} variants")
    print(f"  Noise       : {cfg['noise_fraction']:.0%}")
    print(f"  Top-K       : {top_k}")
    print(f"  HDC dim     : {cfg['hdim']}")
    print(f"  Seed        : {cfg['seed']}")
    print(f"  Ref date    : {cfg['reference_date']}")
    print("=" * 80)
    print()

    col_variant = 16
    col_recall  = 10
    col_mono    = 12
    col_viol    = 12

    print(
        f"  {'Variant':<{col_variant}}  "
        f"{'Recall@' + str(top_k):>{col_recall}}  "
        f"{'Monotonic?':>{col_mono}}  "
        f"{'Violations':>{col_viol}}  "
        f"Chart"
    )
    print(
        f"  {'-'*col_variant}  "
        f"{'-'*col_recall}  "
        f"{'-'*col_mono}  "
        f"{'-'*col_viol}  "
        f"{'-'*BAR_WIDTH}"
    )

    total_checks = len(cfg["monotonicity_distances"]) - 1

    for r in results:
        recall   = r["recall_at_k"]
        mono_ok  = r["monotonicity"]["is_monotonic"]
        viol     = r["monotonicity"]["violations"]
        c        = recall_color(recall)
        print(
            f"  {r['variant']:<{col_variant}}  "
            f"{c}{recall:>{col_recall}.1%}{RESET}  "
            f"{'YES' if mono_ok else 'NO':>{col_mono}}  "
            f"{viol:>{col_viol - 1}}/{total_checks:<2}  "
            f"{c}{bar(recall)}{RESET}"
        )

    print()
    print("  Monotonicity detail (similarity vs temporal distance from reference date):")
    print()
    for r in results:
        print(f"  Variant: {r['variant']}")
        print(f"  {'Days':>6}  {'Similarity':>12}  Note")
        print(f"  {'-'*6}  {'-'*12}  {'-'*20}")
        for row in r["monotonicity"]["distances"]:
            note = " <- violation" if row["violation"] else ""
            print(f"  {row['dist_days']:>6}  {row['similarity']:>12.4f}{note}")
        print()

    print("-" * 80)


def print_nk_sweep_section(path: Path) -> None:
    with open(path) as f:
        data = json.load(f)

    cfg     = data["config"]
    results = data["results"]

    print()
    print("=" * 70)
    print(" N × K Sweep Results  (Recall@k vs collection size)")
    print("=" * 70)
    print(f"  File            : {path.name}")
    print(f"  N values        : {cfg['n_values']}")
    print(f"  K values        : {cfg['k_values']}")
    print(f"  Noise fraction  : {cfg['noise_fraction']}")
    print(f"  Variants/identity: {cfg['variants_per_identity']}")
    print(f"  HDC dim         : {cfg['hdim']}")
    print(f"  Seed            : {cfg['seed']}")
    print("=" * 70)

    modes = sorted({r["vector_mode"] for r in results})
    for mode in modes:
        mode_rows = [r for r in results if r["vector_mode"] == mode]
        k_cols    = sorted({r["top_k"] for r in mode_rows})
        n_vals    = sorted({r["n_identities"] for r in mode_rows})

        cell = {(r["n_identities"], r["top_k"]): r["recall_at_k"] for r in mode_rows}

        print(f"\n  Mode: {mode}")
        header  = f"  {'N':>8} | " + " | ".join(f"k={k:<6}" for k in k_cols)
        divider = "  " + "-" * 9 + "+" + "+".join(["-" * 9] * len(k_cols))
        print(header)
        print(divider)
        for n in n_vals:
            vals = " | ".join(
                f"{recall_color(cell.get((n, k), 0.0))}{cell.get((n, k), float('nan')):>6.3f}{RESET} "
                for k in k_cols
            )
            print(f"  {n:>8} | {vals}")

    print()


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.name.startswith("dedup_recall_"):
            print_dedup_section(path)
        elif path.name.startswith("field_weighting_"):
            print_field_weighting_section(path)
        elif path.name.startswith("scalability_"):
            print_scalability_section(path)
        elif path.name.startswith("ranking_metrics_"):
            print_ranking_section(path)
        elif path.name.startswith("per_field_sweep_"):
            print_per_field_sweep_section(path)
        elif path.name.startswith("per_field_noise_"):
            print_per_field_noise_section(path)
        elif path.name.startswith("dimensionality_"):
            print_dimensionality_section(path)
        elif path.name.startswith("date_encoding_"):
            print_date_encoding_section(path)
        elif path.name.startswith("recall_nk_sweep_"):
            print_nk_sweep_section(path)
        else:
            mode = print_recall_section(path)
            print_bench_section(mode)
    else:
        shown = False
        for mode in ("binary", "float"):
            files = sorted(RESULTS_DIR.glob(f"recall_under_noise_{mode}_*.json"), key=lambda f: f.stat().st_mtime)
            if files:
                print_recall_section(files[-1])
                print_bench_section(mode)
                shown = True
        if not shown:
            # fall back to dedup results if no recall files exist
            for mode in ("binary", "float"):
                files = sorted(RESULTS_DIR.glob(f"dedup_recall_{mode}_*.json"), key=lambda f: f.stat().st_mtime)
                if files:
                    print_dedup_section(files[-1])


if __name__ == "__main__":
    main()
