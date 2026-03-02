import os
import time
import pytest

from configs import settings
from database_utils.milvus_db_connection import get_vector_mode, ensure_people_collection
from encoding_methods.encoding_and_search_milvus import encode_person, _encode_for_milvus
from utils.person_data_normalization import normalize_person_data


@pytest.mark.skipif(
    os.getenv("SKIP_MILVUS_TESTS", "True") == "True",
    reason="Requiere Milvus en ejecución."
)
@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
class TestBenchMilvusInsertion:
    def test_bench_insert_time(self, with_vector_mode, test_collection, test_metrics):
        """
        Benchmark dedicado de inserción en Milvus (batch_size=1 por operación):

        Mide:
          - build.tiempo_insercion_bd: tiempo total (seg) de N inserts (DB insert puro)
          - query.latencia_p50/p95/p99: percentiles (ms) de latencia por insert()
            (reutiliza el bloque "query" del collector para latencias por operación)

        Metodología:
          - Warm-up de inserción (descartado).
          - Documentos con HVs variados (más realista), pero precomputados FUERA de la sección medida
            para aislar el costo de inserción en BD (sin costo de encoding).
          - Inserción medida con una entidad por llamada: col.insert([doc]).
          - Flush opcional medido aparte (no se suma a tiempo_insercion_bd).
        """
        vector_mode = with_vector_mode
        assert get_vector_mode() == vector_mode  # sanity

        warmup = int(os.getenv("BENCH_WARMUP", "20"))
        n = int(os.getenv("BENCH_N", "500"))

        test_metrics.set_config(
            encoding=vector_mode,
            dimension=settings.HDC_DIM,
            seed=settings.DEFAULT_SEED,
        )

        col = ensure_people_collection(test_collection)

        # ---------------------------
        # 1) PREPARACIÓN FUERA DE MEDICIÓN (realismo sin contaminar inserción)
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

        total = warmup + n

        docs = []
        for i in range(total):
            # Variación mínima pero suficiente para:
            # - evitar duplicados exactos
            # - forzar HV distinto (porque cambia mobile_number y akas)
            inp = dict(base_input)
            inp["mobile_number"] = f"55555{i:05d}"
            inp["attrs"] = dict(base_input["attrs"])
            inp["attrs"]["akas"] = [f"Johnny_{i}"]

            normalized = normalize_person_data(inp)

            # HV distinto por doc (pero precomputado acá, fuera de medición)
            hv = encode_person(normalized)
            vec = _encode_for_milvus(hv)

            # Construimos doc compatible con schema
            doc = dict(normalized)

            # dob debe ser string (schema: VARCHAR)
            d = doc.get("dob")
            if d is None:
                doc["dob"] = ""
            else:
                doc["dob"] = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)

            # attrs siempre presente por normalizador, pero aseguramos tipo JSON/dict
            doc["attrs"] = doc.get("attrs", {}) or {}

            # embedding requerido por schema (aunque sea "optional" conceptualmente)
            doc["embedding"] = [0.0] * 128

            # hv requerido por schema (binary/float según modo)
            doc["hv"] = vec

            docs.append(doc)

        # ---------------------------
        # 2) WARM-UP (no medido)
        # ---------------------------
        for i in range(warmup):
            res = col.insert([docs[i]])
            # Chequeo mínimo de integridad
            if hasattr(res, "primary_keys"):
                assert len(res.primary_keys) == 1

        # ---------------------------
        # 3) MEDICIÓN: INSERCIÓN PURA EN BD
        # ---------------------------
        with test_metrics.measure_insertion_time():
            for i in range(warmup, warmup + n):
                t0 = time.perf_counter()
                res = col.insert([docs[i]])
                t1 = time.perf_counter()

                # latencia por operación en ms (para percentiles)
                test_metrics.add_query_latency((t1 - t0) * 1000.0)

                # integridad básica (evita que un insert "vacío" pase desapercibido)
                if hasattr(res, "primary_keys"):
                    assert len(res.primary_keys) == 1

        # ---------------------------
        # 4) FLUSH opcional (medido aparte)
        # ---------------------------
        do_flush = os.getenv("BENCH_FLUSH", "False").lower() == "true"
        if do_flush:
            t0 = time.perf_counter()
            col.flush()
            t1 = time.perf_counter()
            print(f"[BENCH] flush_ms={(t1 - t0) * 1000.0:.2f}")

        # ---------------------------
        # 5) LOG de control (JSON lo guarda el fixture)
        # ---------------------------
        print("\n--- BENCH milvus insert ---")
        print(f"collection={test_collection} mode={vector_mode} dim={settings.HDC_DIM} seed={settings.DEFAULT_SEED}")
        print(f"warmup={warmup} N={n} flush={do_flush}")
        print(f"tiempo_insercion_bd_s={test_metrics.metrics['build']['tiempo_insercion_bd']:.6f}")
        assert test_metrics.metrics["query"]["total_queries"] == n
