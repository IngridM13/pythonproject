"""
Per-Field Noise Sensitivity experiment for HDC-based data reconciliation.

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

import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import HDC_DIM, PER_FIELD_N, PER_FIELD_V, PER_FIELD_NOISE, PER_FIELD_K, PER_FIELD_SEED
from dummy_data.generacion_base_de_datos import generate_data_chunk
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from utils.person_data_normalization import normalize_person_data
from tests.experiments.noise_injection import inject_noise, _NOISE_FUNCS
from tests.experiments.conftest import dataframe_row_to_person_dict


# ---------------------------------------------------------------------------
# Per-field noise helper
# ---------------------------------------------------------------------------

def _inject_single_field_noise(person: dict, field: str, rng: random.Random) -> dict:
    """
    Return a deep copy of `person` with exactly one field corrupted.

    Parameters
    ----------
    person : dict
        Normalized person dict (output of normalize_person_data).
    field : str
        The display field name to corrupt.  "address" is treated as the
        attrs.address key (matching how _NOISE_FUNCS stores it).
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    dict
        Deep copy of person with only the specified field corrupted.
    """
    # Map display name "address" to the internal key "attrs.address"
    lookup_key = "attrs.address" if field == "address" else field

    noisy = copy.deepcopy(person)
    func = _NOISE_FUNCS.get(lookup_key)
    if func is None:
        return noisy

    if lookup_key.startswith("attrs."):
        attr_key = lookup_key.split(".", 1)[1]
        attrs = noisy.get("attrs")
        if isinstance(attrs, dict):
            current = attrs.get(attr_key, [])
            noisy["attrs"][attr_key] = func(current, rng)
    else:
        current = noisy.get(lookup_key)
        noisy[lookup_key] = func(current, rng)

    return noisy


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
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    canonical_persons,
    identity_to_milvus_ids,
    milvus_id_to_identity,
    top_k,
    test_collection,
    query_builder,
):
    """
    Iterate over all canonical identities, build a query per identity using
    query_builder(canonical_person, identity_idx), search, and aggregate metrics.

    Returns
    -------
    tuple[float, float, float]
        (recall_at_k, mrr, hit_at_1)
    """
    recall_hits = 0
    hit_at_1_hits = 0
    reciprocal_ranks = []
    total = 0

    for identity_idx, canonical in enumerate(canonical_persons):
        query_person = query_builder(canonical, identity_idx)

        matches = find_closest_match_db(
            query_person,
            threshold=0.0,
            limit=top_k + 1,
            collection_name=test_collection,
        )

        # Exclude any result that exactly matches a stored variant of this identity
        # by identity label — the query itself was never inserted, so we do not
        # exclude by milvus id, just cap at top_k.
        neighbours = matches[:top_k]

        # Recall@K
        hit = any(
            milvus_id_to_identity.get(m["id"]) == identity_idx
            for m in neighbours
        )
        if hit:
            recall_hits += 1

        # Hit@1
        if neighbours and milvus_id_to_identity.get(neighbours[0]["id"]) == identity_idx:
            hit_at_1_hits += 1

        # MRR
        rr = 0.0
        for rank, m in enumerate(neighbours, 1):
            if milvus_id_to_identity.get(m["id"]) == identity_idx:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        total += 1

    recall_at_k = recall_hits / total if total > 0 else 0.0
    mrr         = sum(reciprocal_ranks) / total if total > 0 else 0.0
    hit_at_1    = hit_at_1_hits / total if total > 0 else 0.0

    return recall_at_k, mrr, hit_at_1


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
        df = generate_data_chunk(n_identities)
        canonical_persons = []
        for _, row in df.iterrows():
            raw = dataframe_row_to_person_dict(row)
            canonical_persons.append(normalize_person_data(raw))

        # --- 2. Generate variants and insert into Milvus ONCE ---
        identity_to_milvus_ids: list = [[] for _ in range(n_identities)]
        milvus_id_to_identity:  dict = {}

        for identity_idx, canonical in enumerate(canonical_persons):
            for variant_idx in range(variants_per_identity):
                variant_rng = random.Random(
                    seed + identity_idx * variants_per_identity + variant_idx
                )
                noisy = inject_noise(canonical, noise_fraction, variant_rng)
                milvus_id = store_person(noisy, collection_name=test_collection)
                identity_to_milvus_ids[identity_idx].append(milvus_id)
                milvus_id_to_identity[milvus_id] = identity_idx

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
        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"per_field_noise_{mode}_{timestamp}.json"
        output_path = output_dir / filename

        report = {
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
        }
        output_path.write_text(json.dumps(report, indent=2))
        print(f"[PER_FIELD] Results saved to {filename}")
