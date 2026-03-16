"""
Experiment 9 — Date Encoding Comparison.

Compares three date encoding variants side-by-side using the standard dedup
recall procedure plus a monotonicity check on raw date similarity.

Variants
--------
- current       : active linear thermometer encoder (year, abs_month, abs_day×2)
- circular      : year stays thermometer; abs_month → FPE month-of-year;
                  abs_day → FPE day-of-year
- circular_full : year, month, and day all encoded with FPE

Each variant is a self-contained subclass that overrides only encode_date_binary
/ encode_date_bipolar; all other field encoding is identical to the baseline.
The subclasses are monkeypatched into enc_module for each run and restored
immediately after.

Dedup recall procedure (same as Experiments 2-4)
-------------------------------------------------
1. Generate N synthetic canonical identities.
2. For each identity, produce V noisy variants via inject_noise().
3. Insert all N×V records into an ephemeral Milvus collection.
4. For each inserted record, query top-(K+1), exclude self, check whether any
   of the remaining top-K results belongs to the same identity.
5. Report recall@K = hits / (N×V).

Monotonicity check
------------------
For each variant, encode a fixed reference date and a series of target dates
at increasing temporal distances (1 day, 7 days, 30 days, …).  Report whether
similarity is non-increasing as distance grows.  Circular variants are expected
to show non-monotonic behaviour across year boundaries (day-of-year periodicity
causes December to be similar to January of the next year).

Run
---
    pytest tests/experiments/test_date_encoding.py -v -s
"""

import json
import os
import random
import sys
import uuid
from datetime import date, timedelta, datetime
from pathlib import Path

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import database_utils.milvus_db_connection as milvus_conn
import encoding_methods.encoding_and_search_milvus as enc_module
from configs.settings import (
    DATE_ENC_K,
    DATE_ENC_N,
    DATE_ENC_NOISE,
    DATE_ENC_SEED,
    DATE_ENC_V,
    HDC_DIM,
)
from database_utils.milvus_db_connection import ensure_people_collection
from dummy_data.generacion_base_de_datos import generate_data_chunk
from encoding_methods.by_data_type.numbers import DecimalEncoding
from encoding_methods.encoding_and_search_milvus import find_closest_match_db, store_person
from hdc.binary_hdc import HyperDimensionalComputingBinary
from hdc.bipolar_hdc import HyperDimensionalComputingBipolar
from tests.experiments.conftest import dataframe_row_to_person_dict
from tests.experiments.noise_injection import inject_noise
from utils.person_data_normalization import normalize_person_data


# ---------------------------------------------------------------------------
# Temporal distances for monotonicity check
# ---------------------------------------------------------------------------

MONOTONICITY_DISTANCES = [0, 1, 7, 30, 90, 180, 365, 730]
REFERENCE_DATE = date(1985, 6, 15)  # representative DOB in synthetic dataset


# ---------------------------------------------------------------------------
# Encoder subclasses — circular (binary)
# ---------------------------------------------------------------------------

class _BinaryCircular(HyperDimensionalComputingBinary):
    """Binary encoder with FPE for month-of-year and day-of-year; year stays thermometer."""

    def _build_fpe(self):
        if hasattr(self, "_fpe_month_circ"):
            return
        self._fpe_month_circ = DecimalEncoding(
            D=self.dim, seed=0xC1C0A001, x0=6.5, smoothness=2.0,
            omega_spread=4.0, output_mode="binary",
        )
        self._fpe_day_circ = DecimalEncoding(
            D=self.dim, seed=0xC1C0A002, x0=182.5, smoothness=30.0,
            omega_spread=4.0, output_mode="binary",
        )

    def encode_date_binary(self, value):
        is_list = isinstance(value, list)
        dates = value if is_list else [value]

        self._build_fpe()

        if not hasattr(self, "_date_ref"):
            self._date_ref = date(1970, 1, 1)
            self._date_min_year = 1900
            self._date_max_year = 2100

        if not hasattr(self, "_date_role_year"):
            self._date_role_year      = self.get_binary_hv("ROLE::DATE::YEAR").to(self.device)
            self._date_role_abs_month = self.get_binary_hv("ROLE::DATE::ABS_MONTH").to(self.device)
            self._date_role_abs_day   = self.get_binary_hv("ROLE::DATE::ABS_DAY").to(self.device)

        out = []
        for d in dates:
            year_vec = self._thermometer_batch(
                "date_year", [d.year], self._date_min_year, self._date_max_year,
            )[0]
            month_vec = self._fpe_month_circ.encode(float(d.month)).to(self.device)
            doy       = d.timetuple().tm_yday
            day_vec   = self._fpe_day_circ.encode(float(doy)).to(self.device)

            bound_year  = self.bind_hv(year_vec,  self._date_role_year)
            bound_month = self.bind_hv(month_vec, self._date_role_abs_month)
            bound_day   = self.bind_hv(day_vec,   self._date_role_abs_day)

            hv = self.bundle_hv([bound_year, bound_month, bound_day, bound_day])
            out.append(hv)

        result = torch.stack(out)
        return result if is_list else result[0]


class _BinaryCircularFull(HyperDimensionalComputingBinary):
    """Binary encoder with FPE for year, month-of-year, and day-of-year."""

    def _build_fpe(self):
        if hasattr(self, "_fpe_year_circ"):
            return
        self._fpe_year_circ = DecimalEncoding(
            D=self.dim, seed=0xC2F0A001, x0=2000.0, smoothness=20.0,
            omega_spread=4.0, output_mode="binary",
        )
        self._fpe_month_circ = DecimalEncoding(
            D=self.dim, seed=0xC2F0A002, x0=6.5, smoothness=2.0,
            omega_spread=4.0, output_mode="binary",
        )
        self._fpe_day_circ = DecimalEncoding(
            D=self.dim, seed=0xC2F0A003, x0=182.5, smoothness=30.0,
            omega_spread=4.0, output_mode="binary",
        )

    def encode_date_binary(self, value):
        is_list = isinstance(value, list)
        dates = value if is_list else [value]

        self._build_fpe()

        if not hasattr(self, "_date_role_year"):
            self._date_role_year      = self.get_binary_hv("ROLE::DATE::YEAR").to(self.device)
            self._date_role_abs_month = self.get_binary_hv("ROLE::DATE::ABS_MONTH").to(self.device)
            self._date_role_abs_day   = self.get_binary_hv("ROLE::DATE::ABS_DAY").to(self.device)

        out = []
        for d in dates:
            year_vec  = self._fpe_year_circ.encode(float(d.year)).to(self.device)
            month_vec = self._fpe_month_circ.encode(float(d.month)).to(self.device)
            doy       = d.timetuple().tm_yday
            day_vec   = self._fpe_day_circ.encode(float(doy)).to(self.device)

            bound_year  = self.bind_hv(year_vec,  self._date_role_year)
            bound_month = self.bind_hv(month_vec, self._date_role_abs_month)
            bound_day   = self.bind_hv(day_vec,   self._date_role_abs_day)

            hv = self.bundle_hv([bound_year, bound_month, bound_day, bound_day])
            out.append(hv)

        result = torch.stack(out)
        return result if is_list else result[0]


# ---------------------------------------------------------------------------
# Encoder subclasses — circular (bipolar / float)
# ---------------------------------------------------------------------------

class _BipolarCircular(HyperDimensionalComputingBipolar):
    """Bipolar encoder with FPE for month-of-year and day-of-year; year stays thermometer."""

    def _build_fpe(self):
        if hasattr(self, "_fpe_month_circ"):
            return
        self._fpe_month_circ = DecimalEncoding(
            D=self.dim, seed=0xC1C0B001, x0=6.5, smoothness=2.0,
            omega_spread=4.0, output_mode="bipolar",
        )
        self._fpe_day_circ = DecimalEncoding(
            D=self.dim, seed=0xC1C0B002, x0=182.5, smoothness=30.0,
            omega_spread=4.0, output_mode="bipolar",
        )

    def encode_date_bipolar(self, date_obj):
        if date_obj is None:
            return torch.ones(self.dim, dtype=torch.int8, device=self.device)

        is_list = isinstance(date_obj, list)
        dates = date_obj if is_list else [date_obj]

        self._build_fpe()

        if not hasattr(self, "_date_role_year"):
            self._date_role_year      = self.get_bipolar_hv("ROLE::DATE::YEAR")
            self._date_role_abs_month = self.get_bipolar_hv("ROLE::DATE::ABS_MONTH")
            self._date_role_abs_day   = self.get_bipolar_hv("ROLE::DATE::ABS_DAY")

        years_min, years_max = 1900, 2100

        def _therm(name, val, vmin, vmax):
            seed = self._deterministic_hash(name)
            rng  = torch.Generator(device="cpu").manual_seed(seed)
            perm = torch.randperm(self.dim, generator=rng, device="cpu").to(self.device)
            vt   = torch.tensor([val], dtype=torch.int32, device=self.device)
            vt   = torch.clamp(vt, vmin, vmax)
            prop = (vt - vmin).float() / float(vmax - vmin)
            n    = int((prop * self.dim).item())
            out  = torch.zeros(self.dim, dtype=torch.int8, device=self.device)
            if n > 0:
                out[perm[:n]] = 1
            return (out * 2 - 1).to(torch.int8)

        out = []
        for d in dates:
            year_vec  = _therm("date_year", d.year, years_min, years_max)
            month_vec = self._fpe_month_circ.encode(float(d.month)).to(device=self.device, dtype=torch.int8)
            doy       = d.timetuple().tm_yday
            day_vec   = self._fpe_day_circ.encode(float(doy)).to(device=self.device, dtype=torch.int8)

            bound_year  = self.bind_hv(year_vec,  self._date_role_year)
            bound_month = self.bind_hv(month_vec, self._date_role_abs_month)
            bound_day   = self.bind_hv(day_vec,   self._date_role_abs_day)

            acc  = bound_year.to(torch.int32) + bound_month.to(torch.int32) + bound_day.to(torch.int32) * 2
            res  = torch.sign(acc).to(torch.int8)
            zeros = (res == 0)
            if torch.any(zeros):
                tb = self._tie_breaker_bipolar("date_bundle", self.dim).to(self.device)
                res[zeros] = tb[zeros]
            out.append(res)

        result = torch.stack(out)
        return result if is_list else result[0]


class _BipolarCircularFull(HyperDimensionalComputingBipolar):
    """Bipolar encoder with FPE for year, month-of-year, and day-of-year."""

    def _build_fpe(self):
        if hasattr(self, "_fpe_year_circ"):
            return
        self._fpe_year_circ = DecimalEncoding(
            D=self.dim, seed=0xC2F0B001, x0=2000.0, smoothness=20.0,
            omega_spread=4.0, output_mode="bipolar",
        )
        self._fpe_month_circ = DecimalEncoding(
            D=self.dim, seed=0xC2F0B002, x0=6.5, smoothness=2.0,
            omega_spread=4.0, output_mode="bipolar",
        )
        self._fpe_day_circ = DecimalEncoding(
            D=self.dim, seed=0xC2F0B003, x0=182.5, smoothness=30.0,
            omega_spread=4.0, output_mode="bipolar",
        )

    def encode_date_bipolar(self, date_obj):
        if date_obj is None:
            return torch.ones(self.dim, dtype=torch.int8, device=self.device)

        is_list = isinstance(date_obj, list)
        dates = date_obj if is_list else [date_obj]

        self._build_fpe()

        if not hasattr(self, "_date_role_year"):
            self._date_role_year      = self.get_bipolar_hv("ROLE::DATE::YEAR")
            self._date_role_abs_month = self.get_bipolar_hv("ROLE::DATE::ABS_MONTH")
            self._date_role_abs_day   = self.get_bipolar_hv("ROLE::DATE::ABS_DAY")

        out = []
        for d in dates:
            year_vec  = self._fpe_year_circ.encode(float(d.year)).to(device=self.device, dtype=torch.int8)
            month_vec = self._fpe_month_circ.encode(float(d.month)).to(device=self.device, dtype=torch.int8)
            doy       = d.timetuple().tm_yday
            day_vec   = self._fpe_day_circ.encode(float(doy)).to(device=self.device, dtype=torch.int8)

            bound_year  = self.bind_hv(year_vec,  self._date_role_year)
            bound_month = self.bind_hv(month_vec, self._date_role_abs_month)
            bound_day   = self.bind_hv(day_vec,   self._date_role_abs_day)

            acc  = bound_year.to(torch.int32) + bound_month.to(torch.int32) + bound_day.to(torch.int32) * 2
            res  = torch.sign(acc).to(torch.int8)
            zeros = (res == 0)
            if torch.any(zeros):
                tb = self._tie_breaker_bipolar("date_bundle", self.dim).to(self.device)
                res[zeros] = tb[zeros]
            out.append(res)

        result = torch.stack(out)
        return result if is_list else result[0]


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

DATE_VARIANTS = [
    {
        "name":          "current",
        "binary_class":  HyperDimensionalComputingBinary,
        "bipolar_class": HyperDimensionalComputingBipolar,
    },
    {
        "name":          "circular",
        "binary_class":  _BinaryCircular,
        "bipolar_class": _BipolarCircular,
    },
    {
        "name":          "circular_full",
        "binary_class":  _BinaryCircularFull,
        "bipolar_class": _BipolarCircularFull,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_encoder(mode: str, variant: dict):
    cls = variant["binary_class"] if mode == "binary" else variant["bipolar_class"]
    return cls(dim=HDC_DIM)


def _encode_date(encoder, d: date) -> torch.Tensor:
    if isinstance(encoder, HyperDimensionalComputingBinary):
        return encoder.encode_date_binary(d)
    return encoder.encode_date_bipolar(d)


def _date_similarity(v1: torch.Tensor, v2: torch.Tensor, mode: str) -> float:
    if mode == "binary":
        diff = torch.logical_xor(v1.bool(), v2.bool()).sum().item()
        return 1.0 - diff / v1.numel()
    return float(torch.dot(v1.float(), v2.float()).item()) / v1.numel()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestDateEncoding:

    def test_date_encoding_comparison(self):
        n_identities          = DATE_ENC_N
        variants_per_identity = DATE_ENC_V
        noise_fraction        = DATE_ENC_NOISE
        top_k                 = DATE_ENC_K
        seed                  = DATE_ENC_SEED
        total_records         = n_identities * variants_per_identity

        config = {
            "n_identities":           n_identities,
            "variants_per_identity":  variants_per_identity,
            "noise_fraction":         noise_fraction,
            "top_k":                  top_k,
            "hdim":                   HDC_DIM,
            "seed":                   seed,
            "monotonicity_distances": MONOTONICITY_DISTANCES,
            "reference_date":         str(REFERENCE_DATE),
        }

        project_root = Path(__file__).resolve().parents[2]
        output_dir   = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)

        # Pre-generate canonical identities (shared across all variants and modes)
        df = generate_data_chunk(n_identities)
        canonical_persons = []
        for _, row in df.iterrows():
            raw = dataframe_row_to_person_dict(row)
            canonical_persons.append(normalize_person_data(raw))

        for mode in ["binary", "float"]:
            original_mode = milvus_conn.VECTOR_MODE
            milvus_conn.VECTOR_MODE = mode

            try:
                mode_results = []

                print(
                    f"\n[DATE] mode={mode}  n={n_identities}  "
                    f"variants={variants_per_identity}  noise={noise_fraction}  top_k={top_k}"
                )

                for variant_cfg in DATE_VARIANTS:
                    variant_name = variant_cfg["name"]

                    # Monkeypatch encoder classes in enc_module for this variant
                    original_binary  = enc_module.HyperDimensionalComputingBinary
                    original_bipolar = enc_module.HyperDimensionalComputingBipolar
                    enc_module.HyperDimensionalComputingBinary  = variant_cfg["binary_class"]
                    enc_module.HyperDimensionalComputingBipolar = variant_cfg["bipolar_class"]

                    try:
                        col_name = f"de_{uuid.uuid4().hex[:10]}"
                        col = ensure_people_collection(col_name, include_embedding=False)

                        print(f"\n[DATE] mode={mode}  variant={variant_name}  collection={col_name}")

                        try:
                            # --- Insert noisy variants ---
                            identity_to_milvus_ids: list = [[] for _ in range(n_identities)]
                            milvus_id_to_identity:  dict = {}

                            for identity_idx, canonical in enumerate(canonical_persons):
                                for variant_idx in range(variants_per_identity):
                                    rng = random.Random(
                                        seed
                                        + identity_idx * variants_per_identity
                                        + variant_idx
                                    )
                                    noisy     = inject_noise(canonical, noise_fraction, rng)
                                    milvus_id = store_person(noisy, collection_name=col_name)
                                    identity_to_milvus_ids[identity_idx].append(milvus_id)
                                    milvus_id_to_identity[milvus_id] = identity_idx

                            col.flush()
                            print(
                                f"[DATE] Inserted & flushed {total_records} records  "
                                f"variant={variant_name}"
                            )

                            # --- Dedup recall@K ---
                            hits  = 0
                            total = 0

                            for identity_idx, milvus_ids in enumerate(identity_to_milvus_ids):
                                for variant_idx, query_milvus_id in enumerate(milvus_ids):
                                    query_rng = random.Random(
                                        seed
                                        + identity_idx * variants_per_identity
                                        + variant_idx
                                    )
                                    query_person = inject_noise(
                                        canonical_persons[identity_idx],
                                        noise_fraction,
                                        query_rng,
                                    )

                                    matches = find_closest_match_db(
                                        query_person,
                                        threshold=0.0,
                                        limit=top_k + 1,
                                        collection_name=col_name,
                                    )

                                    neighbours = [
                                        m for m in matches if m["id"] != query_milvus_id
                                    ][:top_k]

                                    hit = any(
                                        milvus_id_to_identity.get(m["id"]) == identity_idx
                                        for m in neighbours
                                    )
                                    if hit:
                                        hits += 1
                                    total += 1

                            recall_at_k = hits / total if total > 0 else 0.0
                            print(
                                f"[DATE] mode={mode}  variant={variant_name}  "
                                f"recall@{top_k}={recall_at_k:.3f}  ({hits}/{total})"
                            )

                            # --- Monotonicity check (raw date encoding, no Milvus) ---
                            encoder  = _build_encoder(mode, variant_cfg)
                            ref_hv   = _encode_date(encoder, REFERENCE_DATE)
                            prev_sim = None
                            violations = 0
                            mono_rows  = []

                            for dist_days in MONOTONICITY_DISTANCES:
                                target = REFERENCE_DATE + timedelta(days=dist_days)
                                tgt_hv = _encode_date(encoder, target)
                                sim    = _date_similarity(ref_hv, tgt_hv, mode)

                                is_violation = (prev_sim is not None and sim > prev_sim + 1e-6)
                                if is_violation:
                                    violations += 1

                                mono_rows.append({
                                    "dist_days":  dist_days,
                                    "similarity": round(float(sim), 6),
                                    "violation":  is_violation,
                                })
                                prev_sim = sim

                            print(
                                f"[DATE] monotonicity  variant={variant_name}  "
                                f"violations={violations}/{len(MONOTONICITY_DISTANCES) - 1}"
                            )

                            mode_results.append({
                                "variant":     variant_name,
                                "recall_at_k": round(recall_at_k, 6),
                                "hits":        hits,
                                "total":       total,
                                "monotonicity": {
                                    "is_monotonic": violations == 0,
                                    "violations":   violations,
                                    "distances":    mono_rows,
                                },
                            })

                        finally:
                            try:
                                col.drop()
                            except Exception as drop_err:
                                print(f"[DATE] Warning: could not drop {col_name}: {drop_err}")

                    finally:
                        enc_module.HyperDimensionalComputingBinary  = original_binary
                        enc_module.HyperDimensionalComputingBipolar = original_bipolar

                # --- Save JSON report ---
                timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"date_encoding_{mode}_{timestamp}.json"
                report = {
                    "mode":    mode,
                    "config":  config,
                    "results": mode_results,
                }
                output_path.write_text(json.dumps(report, indent=2))
                print(f"\n[DATE] Results saved to {output_path.name}")

                # --- Print summary table ---
                col_variant  = 16
                col_recall   = 10
                col_mono     = 12
                col_viol     = 12
                col_chart    = 30

                print(f"\nMode: {mode}")
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
                    f"{'-'*col_chart}"
                )
                for row in mode_results:
                    recall  = row["recall_at_k"]
                    mono_ok = row["monotonicity"]["is_monotonic"]
                    viol    = row["monotonicity"]["violations"]
                    total_checks = len(MONOTONICITY_DISTANCES) - 1
                    filled  = round(recall * col_chart)
                    chart   = "#" * filled + "-" * (col_chart - filled)
                    print(
                        f"  {row['variant']:<{col_variant}}  "
                        f"{recall:>{col_recall}.3f}  "
                        f"{'YES' if mono_ok else 'NO':>{col_mono}}  "
                        f"{viol:>{col_viol - 1}}/{total_checks:<2}  "
                        f"{chart}"
                    )

            finally:
                milvus_conn.VECTOR_MODE = original_mode
