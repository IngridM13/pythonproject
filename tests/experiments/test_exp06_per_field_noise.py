"""
Experiment 6 — Per-Field Noise Sensitivity for HDC-based data reconciliation.

Measures how much each individual field contributes to retrieval quality by
corrupting exactly one field at a time while keeping all other fields clean.
This isolates each field's informational weight within the HDC encoding.

Setup
-----
1. Generate N synthetic canonical identities.
2. For each identity, produce V noisy variants using inject_noise().
3. Insert all N×V records into Milvus once (reused for all per-field queries).
4. Run a BASELINE: query each canonical (clean) person, compute Recall@K, MRR, Hit@1.
5. For each field: corrupt only that field in the canonical person, query, compute
   the same three metrics.
6. Compute delta vs baseline (baseline - field_value); positive delta = degradation.
7. Print a ranked table sorted by recall delta descending.

Run
---
    pytest tests/experiments/test_per_field_noise.py -v -s

Environment variables
---------------------
    PER_FIELD_N         Number of canonical identities (default: PER_FIELD_N from settings)
    PER_FIELD_V         Noisy variants per identity (default: PER_FIELD_V from settings)
    PER_FIELD_NOISE     Noise fraction for variant generation (default: PER_FIELD_NOISE from settings)
    PER_FIELD_K         K for recall@K (default: PER_FIELD_K from settings)
    PER_FIELD_SEED      RNG seed (default: PER_FIELD_SEED from settings)
"""

import os
import random
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import HDC_DIM, PER_FIELD_N, PER_FIELD_V, PER_FIELD_NOISE, PER_FIELD_K, PER_FIELD_SEED
from tests.experiments.experiment_utils import (
    _inject_single_field_noise,
    _compute_metrics,
    generate_canonical_persons,
    insert_noisy_variants,
    save_report,
)


# ---------------------------------------------------------------------------
# Fields under test (display names, in prescribed order)
# ---------------------------------------------------------------------------

_FIELDS_TO_TEST = [
    "name",
    "lastname",
    "dob",
    "gender",
    "race",
    "marital_status",
    "mobile_number",
    "address",
]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("with_vector_mode")
class TestPerFieldNoise:

    def test_per_field_noise(self, with_vector_mode, test_collection):
        # --- Config from env ---
        n_identities          = int(os.environ.get("PER_FIELD_N", PER_FIELD_N))
        variants_per_identity = int(os.environ.get("PER_FIELD_V", PER_FIELD_V))
        noise_fraction        = float(os.environ.get("PER_FIELD_NOISE", PER_FIELD_NOISE))
        top_k                 = int(os.environ.get("PER_FIELD_K", PER_FIELD_K))
        seed                  = int(os.environ.get("PER_FIELD_SEED", PER_FIELD_SEED))

        mode = with_vector_mode
        total_records = n_identities * variants_per_identity

        print(
            f"\n[PER_FIELD] mode={mode}, n_identities={n_identities}, "
            f"variants_per_identity={variants_per_identity}, "
            f"noise_fraction={noise_fraction}, top_k={top_k}, seed={seed}"
        )
        print(f"[PER_FIELD] Total records to insert: {total_records}")

        # --- 1. Generate canonical identities ---
        canonical_persons = generate_canonical_persons(n_identities)

        # --- 2. Generate variants and insert into Milvus ONCE ---
        identity_to_milvus_ids, milvus_id_to_identity = insert_noisy_variants(
            canonical_persons, variants_per_identity, noise_fraction, seed, test_collection
        )

        # --- 3. Flush once — reused for all field variants ---
        from database_utils.milvus_db_connection import ensure_people_collection
        col = ensure_people_collection(test_collection)
        col.flush()
        print(f"[PER_FIELD] Inserted & flushed {total_records} records.")

        # --- 4. BASELINE: query with clean canonical person ---
        print("[PER_FIELD] Computing baseline (clean canonical queries)...")

        def _baseline_query(canonical, identity_idx):
            return canonical

        base_recall, base_mrr, base_hit1 = _compute_metrics(
            canonical_persons,
            identity_to_milvus_ids,
            milvus_id_to_identity,
            top_k,
            test_collection,
            _baseline_query,
        )

        print(
            f"[PER_FIELD] BASELINE  recall@{top_k}={base_recall:.3f}  "
            f"MRR={base_mrr:.3f}  Hit@1={base_hit1:.3f}"
        )

        # --- 5. Per-field evaluation ---
        field_results = []

        for field in _FIELDS_TO_TEST:
            field_rng_base = random.Random(seed + hash(field))

            def _field_query(canonical, identity_idx, _field=field, _base=seed):
                rng = random.Random(_base + hash(_field) + identity_idx)
                return _inject_single_field_noise(canonical, _field, rng)

            f_recall, f_mrr, f_hit1 = _compute_metrics(
                canonical_persons,
                identity_to_milvus_ids,
                milvus_id_to_identity,
                top_k,
                test_collection,
                _field_query,
            )

            delta_recall = base_recall - f_recall
            delta_mrr    = base_mrr    - f_mrr
            delta_hit1   = base_hit1   - f_hit1

            field_results.append({
                "field":        field,
                "recall_at_k":  f_recall,
                "delta_recall": delta_recall,
                "mrr":          f_mrr,
                "delta_mrr":    delta_mrr,
                "hit_at_1":     f_hit1,
                "delta_hit1":   delta_hit1,
            })

            print(
                f"[PER_FIELD] field={field:<16}  "
                f"recall@{top_k}={f_recall:.3f} (Δ{delta_recall:+.3f})  "
                f"MRR={f_mrr:.3f} (Δ{delta_mrr:+.3f})  "
                f"Hit@1={f_hit1:.3f} (Δ{delta_hit1:+.3f})"
            )

        # --- 6. Ranked table (sorted by delta_recall descending) ---
        ranked = sorted(field_results, key=lambda r: r["delta_recall"], reverse=True)

        RED    = "\033[91m"
        YELLOW = "\033[93m"
        GREEN  = "\033[92m"
        RESET  = "\033[0m"

        BAR_MAX = 20

        def _delta_color(delta):
            if delta > 0.1:
                return RED
            elif delta >= 0.05:
                return YELLOW
            return GREEN

        def _delta_bar(delta):
            filled = round(abs(delta) * BAR_MAX / max(0.001, max(r["delta_recall"] for r in field_results)))
            filled = min(filled, BAR_MAX)
            return "#" * filled + "-" * (BAR_MAX - filled)

        print()
        print("=" * 90)
        print(f" Per-Field Noise Sensitivity  (mode: {mode})")
        print("=" * 90)
        print(
            f"  {'Field':<16}  {'Recall@K':>8}  {'Δ Recall':>9}  "
            f"{'MRR':>7}  {'Δ MRR':>8}  {'Hit@1':>7}  {'Δ Hit@1':>8}  Bar (Δ Recall)"
        )
        print(
            f"  {'-'*16}  {'-'*8}  {'-'*9}  "
            f"{'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*BAR_MAX}"
        )

        # Baseline row first
        print(
            f"  {'[baseline]':<16}  {base_recall:>8.1%}  {'—':>9}  "
            f"{base_mrr:>7.3f}  {'—':>8}  {base_hit1:>7.1%}  {'—':>8}"
        )
        print(f"  {'-'*16}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*BAR_MAX}")

        for r in ranked:
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

        # --- 7. Save JSON report ---
        output_path = save_report("per_field_noise", mode, {
            "mode": mode,
            "config": {
                "n_identities":          n_identities,
                "variants_per_identity": variants_per_identity,
                "noise_fraction":        noise_fraction,
                "top_k":                 top_k,
                "hdim":                  HDC_DIM,
                "seed":                  seed,
            },
            "baseline": {
                "recall_at_k": base_recall,
                "mrr":         base_mrr,
                "hit_at_1":    base_hit1,
            },
            "results": ranked,
        })
        print(f"[PER_FIELD] Results saved to {output_path.name}")
