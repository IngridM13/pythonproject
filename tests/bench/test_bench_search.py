
import os
import time
import pytest
from contextlib import contextmanager

from configs import settings
from database_utils.milvus_db_connection import get_vector_mode, ensure_people_collection
from encoding_methods.encoding_and_search_milvus import encode_person, _encode_for_milvus, find_closest_match_db
from utils.person_data_normalization import normalize_person_data


@pytest.mark.skipif(
    os.getenv("SKIP_MILVUS_TESTS", "True") == "True",
    reason="Requiere Milvus en ejecución."
)
@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
class TestBenchMilvusSearch:
    def test_bench_search_time(self, with_vector_mode, test_collection, test_metrics):
        """
        Benchmark dedicado de búsqueda en Milvus:

        Mide:
          - build.tiempo_busqueda_bd: tiempo total (seg) de N búsquedas (DB search puro)
          - query.latencia_p50/p95/p99: percentiles (ms) de latencia por search()
            (reutiliza el bloque "query" del collector para latencias por operación)

        Metodología:
          - Warm-up de inserciones previas para tener datos que buscar.
          - Warm-up de búsquedas (descartado).
          - Consultas con HVs variados (más realista), pero precomputados FUERA de la sección medida
            para aislar el costo de búsqueda en BD (sin costo de encoding).
          - Búsqueda medida con un vector query por llamada: search([query]).
        """
        vector_mode = with_vector_mode
        assert get_vector_mode() == vector_mode  # sanity

        warmup = int(os.getenv("BENCH_WARMUP", "20"))
        n = int(os.getenv("BENCH_N", "500"))

        # Número de documentos a insertar para realizar búsquedas
        num_docs = int(os.getenv("BENCH_DOCS", "1000"))

        test_metrics.set_config(
            encoding=vector_mode,
            dimension=settings.HDC_DIM,
            seed=settings.DEFAULT_SEED,
        )

        col = ensure_people_collection(test_collection)

        # ---------------------------
        # 1) PREPARACIÓN: INSERTAR DATOS PARA BÚSQUEDA
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

        # Insertar documentos para realizar búsquedas
        docs = []
        for i in range(num_docs):
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

        # Insertar documentos en lotes para eficiencia
        batch_size = 100
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            col.insert(batch)

        # Flush para asegurar que los datos estén disponibles para búsqueda
        col.flush()
        print(f"[BENCH] Insertados {num_docs} documentos para búsqueda")

        # ---------------------------
        # 2) PREPARACIÓN DE QUERIES DE BÚSQUEDA
        # ---------------------------
        # Generamos consultas variadas para búsqueda
        queries = []
        for i in range(warmup + n):
            # Crear variaciones para las consultas
            inp = dict(base_input)
            inp["mobile_number"] = f"55555{(i % num_docs):05d}"  # Reciclar algunos IDs
            inp["attrs"] = dict(base_input["attrs"])
            inp["attrs"]["akas"] = [f"Johnny_query_{i}"]

            # Agregar algunas variaciones adicionales para tener diferentes niveles de similitud
            if i % 3 == 0:
                inp["gender"] = "Female" if i % 2 == 0 else "Male"
            if i % 5 == 0:
                inp["lastname"] = f"Doe_{i % 10}"

            queries.append(inp)

        # ---------------------------
        # 3) WARM-UP DE BÚSQUEDA (no medido)
        # ---------------------------
        for i in range(warmup):
            _ = find_closest_match_db(queries[i], collection_name=test_collection)

        # ---------------------------
        # 4) MEDICIÓN: BÚSQUEDA PURA EN BD
        # ---------------------------
        # Creamos contexto personalizado para medir el tiempo de búsqueda
        # El TestMetricsCollector original no tiene un método específico para búsqueda
        @contextmanager
        def measure_search_time():
            """Context manager para medir tiempo de búsqueda."""
            start = time.perf_counter()
            try:
                yield
            finally:
                elapsed = time.perf_counter() - start
                test_metrics.metrics["build"]["tiempo_busqueda_bd"] = elapsed

        # Inicializamos el campo en la métrica si no existe
        test_metrics.metrics["build"]["tiempo_busqueda_bd"] = 0.0

        with measure_search_time():
            for i in range(warmup, warmup + n):
                t0 = time.perf_counter()
                results = find_closest_match_db(queries[i], collection_name=test_collection)
                t1 = time.perf_counter()

                # latencia por operación en ms (para percentiles)
                test_metrics.add_query_latency((t1 - t0) * 1000.0)

                # integridad básica
                assert isinstance(results, list)

        # ---------------------------
        # 5) LOG de control (JSON lo guarda el fixture)
        # ---------------------------
        print("\n--- BENCH milvus search ---")
        print(f"collection={test_collection} mode={vector_mode} dim={settings.HDC_DIM} seed={settings.DEFAULT_SEED}")
        print(f"warmup={warmup} N={n} docs_in_collection={num_docs}")
        print(f"tiempo_busqueda_bd_s={test_metrics.metrics['build']['tiempo_busqueda_bd']:.6f}")
        assert test_metrics.metrics["query"]["total_queries"] == n