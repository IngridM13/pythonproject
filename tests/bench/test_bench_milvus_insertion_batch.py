import os
import time
import math
import pytest

from configs import settings
from database_utils.milvus_db_connection import get_vector_mode, ensure_people_collection
from encoding_methods.encoding_and_search_milvus import encode_person, _encode_for_milvus
from utils.person_data_normalization import normalize_person_data


def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


@pytest.mark.skipif(
    os.getenv("SKIP_MILVUS_TESTS", "True") == "True",
    reason="Requiere Milvus en ejecución."
)
@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
class TestBenchMilvusInsertionBatch:
    def test_bench_insert_batch_time(self, with_vector_mode, test_collection, test_metrics):
        """
        Benchmark de inserción en Milvus con batch:

        Mide:
          - build.tiempo_insercion_bd: tiempo total (seg) para insertar N docs en batches
          - query.latencia_p50/p95/p99: percentiles (ms) por batch insert()

        Metodología:
          - Warm-up de batches descartado.
          - HVs variados por doc, precomputados fuera de la sección medida (aislar BD).
          - batch_size configurable por env var BENCH_BATCH.
          - Flush opcional medido aparte (no suma a tiempo_insercion_bd).
        """
        vector_mode = with_vector_mode
        assert get_vector_mode() == vector_mode  # sanity

        warmup_batches = int(os.getenv("BENCH_WARMUP_BATCHES", "5"))
        n_docs = int(os.getenv("BENCH_N", "5000"))
        batch_size = int(os.getenv("BENCH_BATCH", "256"))

        if batch_size <= 0:
            raise ValueError("BENCH_BATCH must be > 0")

        test_metrics.set_config(
            encoding=vector_mode,
            dimension=settings.HDC_DIM,
            seed=settings.DEFAULT_SEED,
        )

        col = ensure_people_collection(test_collection)

        # ---------------------------
        # 1) PREPARACIÓN FUERA DE MEDICIÓN
        # ---------------------------
        base_input = {
            "name": "John",
            "lastname": "Doe",
            "dob": "1990-05-15",
            "gender": "Male",
            "race": "Caucasian",
            "marital_status": "Single",
            "attrs": {
                "address": ["123 Main St, City"],
                "akas": ["Johnny"],
                "landlines": ["987654322"],
            },
        }

        docs = []
        for i in range(n_docs):
            inp = dict(base_input)
            inp["mobile_number"] = f"55555{i:07d}"
            inp["attrs"] = dict(base_input["attrs"])
            inp["attrs"]["akas"] = [f"Johnny_{i}"]

            normalized = normalize_person_data(inp)

            hv = encode_person(normalized)
            vec = _encode_for_milvus(hv)

            doc = dict(normalized)

            d = doc.get("dob")
            if d is None:
                doc["dob"] = ""
            else:
                doc["dob"] = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)

            doc["attrs"] = doc.get("attrs", {}) or {}
            doc["embedding"] = [0.0] * 128
            doc["hv"] = vec

            docs.append(doc)

        batches = list(_chunk(docs, batch_size))
        assert batches, "No batches generated"

        # ---------------------------
        # 2) WARM-UP (por batches, no medido)
        # ---------------------------
        # Warm-up con los primeros warmup_batches
        for b in batches[:warmup_batches]:
            res = col.insert(b)
            if hasattr(res, "primary_keys"):
                assert len(res.primary_keys) == len(b)

        # ---------------------------
        # 3) MEDICIÓN: INSERCIÓN BATCH
        # ---------------------------
        # Medimos desde el batch warmup_batches en adelante
        measured_batches = batches[warmup_batches:]

        with test_metrics.measure_insertion_time():
            for b in measured_batches:
                t0 = time.perf_counter()
                res = col.insert(b)  # batch insert real
                t1 = time.perf_counter()

                test_metrics.add_query_latency((t1 - t0) * 1000.0)

                if hasattr(res, "primary_keys"):
                    assert len(res.primary_keys) == len(b)

        # ---------------------------
        # 4) FLUSH opcional (aparte)
        # ---------------------------
        do_flush = os.getenv("BENCH_FLUSH", "False").lower() == "true"
        flush_ms = None
        if do_flush:
            t0 = time.perf_counter()
            col.flush()
            t1 = time.perf_counter()
            flush_ms = (t1 - t0) * 1000.0
            print(f"[BENCH] flush_ms={flush_ms:.2f}")

        # ---------------------------
        # 5) METRICAS DERIVADAS PARA LOG (no requiere cambiar collector)
        # ---------------------------
        measured_docs = sum(len(b) for b in measured_batches)
        total_s = test_metrics.metrics["build"]["tiempo_insercion_bd"]
        docs_per_s = (measured_docs / total_s) if total_s > 0 else float("inf")
        ms_per_doc = (total_s * 1000.0 / measured_docs) if measured_docs > 0 else float("inf")

        print("\n--- BENCH milvus insert (BATCH) ---")
        print(f"collection={test_collection} mode={vector_mode} dim={settings.HDC_DIM} seed={settings.DEFAULT_SEED}")
        print(f"n_docs={n_docs} batch_size={batch_size} warmup_batches={warmup_batches} measured_docs={measured_docs}")
        print(f"tiempo_insercion_bd_s={total_s:.6f} docs_per_s={docs_per_s:.2f} ms_per_doc={ms_per_doc:.4f} flush={do_flush}")

        # Integridad: total_queries = cantidad de batches medidos
        assert test_metrics.metrics["query"]["total_queries"] == len(measured_batches)
