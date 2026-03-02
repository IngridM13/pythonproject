import os
import time
import pytest

from configs import settings
from database_utils.milvus_db_connection import get_vector_mode
from encoding_methods.encoding_and_search_milvus import encode_person
from utils.person_data_normalization import normalize_person_data


@pytest.mark.skipif(
    os.getenv("SKIP_MILVUS_TESTS", "True") == "True",
    reason="El suite usa fixtures del proyecto para fijar VECTOR_MODE."
)
@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
class TestBenchEncodePerson:
    def test_bench_encode_person_time(self, with_vector_mode, test_metrics):
        """
        Benchmark dedicado de encode_person:
        - build.tiempo_encoding_total: tiempo total (seg) del bloque de N encodings
        - query.latencia_p50/p95/p99: percentiles (ms) de latencia por llamada
        """
        vector_mode = with_vector_mode
        assert get_vector_mode() == vector_mode  # sanity

        # Configurable por env vars
        warmup = int(os.getenv("BENCH_WARMUP", "50"))
        n = int(os.getenv("BENCH_N", "500"))

        person = {
            "name": "John",
            "lastname": "Doe",
            "dob": "1990-05-15",
            "marital_status": "Single",
            "mobile_number": "123456789",
            "gender": "Male",
            "race": "Caucasian",
            "attrs": {
                "address": ["123 Main St, City"],
                "akas": ["Johnny"],
                "landlines": ["987654322"],
            },
        }

        # Config registrada en JSON
        test_metrics.set_config(
            encoding=vector_mode,
            dimension=settings.HDC_DIM,
            seed=settings.DEFAULT_SEED,
        )

        # Medimos SOLO encode_person: normalizamos 1 vez
        normalized = normalize_person_data(person)

        # Warmup (no cuenta para métricas)
        for _ in range(warmup):
            _ = encode_person(normalized)

        # Medición: tiempo total acumulado + latencias individuales
        with test_metrics.measure_encoding_time():
            for _ in range(n):
                t0 = time.perf_counter()
                _ = encode_person(normalized)
                t1 = time.perf_counter()
                test_metrics.add_query_latency((t1 - t0) * 1000.0)

        # Log útil en consola (el JSON se guarda por fixture)
        print("\n--- BENCH encode_person ---")
        print(f"mode={vector_mode} dim={settings.HDC_DIM} seed={settings.DEFAULT_SEED}")
        print(f"warmup={warmup} N={n}")
        print(f"tiempo_encoding_total_s={test_metrics.metrics['build']['tiempo_encoding_total']:.6f}")
        # latencias p50/p95/p99 se calculan en save_metrics()

        # Integridad (no performance)
        assert test_metrics.metrics["query"]["total_queries"] == n
