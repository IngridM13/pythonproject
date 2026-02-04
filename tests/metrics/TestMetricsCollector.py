import json
import time
from contextlib import contextmanager
from typing import Optional
from datetime import datetime


class TestMetricsCollector:
    """
    Recopila métricas durante las pruebas y las guarda en formato JSON.
    Soporta Context Managers para medición segura de tiempos.
    """

    def __init__(self):
        self.metrics = {
            "configuracion": {
                "encoding": None,
                "dimension": None,
                "seed": None,
                "regla_bundling": None
            },
            "build": {
                "tiempo_encoding_total": 0.0,  # Inicializar en 0.0 para acumular
                "tiempo_insercion_bd": 0.0,
                "tamano_indice_memoria": None
            },
            "query": {
                "latencia_p50": None,
                "latencia_p95": None,
                "latencia_p99": None,
                "total_queries": 0
            },
            "timestamp": datetime.now().isoformat()
        }
        self._query_latencies = []

    def set_config(self, encoding: str, dimension: int, seed: int, regla_bundling: str):
        self.metrics["configuracion"].update({
            "encoding": encoding,
            "dimension": dimension,
            "seed": seed,
            "regla_bundling": regla_bundling
        })

    @contextmanager
    def measure_encoding_time(self):
        """Context manager para medir tiempo de encoding de forma segura."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            # Acumulamos el tiempo por si se llama varias veces en un loop
            self.metrics["build"]["tiempo_encoding_total"] += elapsed

    @contextmanager
    def measure_insertion_time(self):
        """Context manager para medir tiempo de inserción."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.metrics["build"]["tiempo_insercion_bd"] += elapsed

    def set_index_memory_size(self, size_mb: float):
        self.metrics["build"]["tamano_indice_memoria"] = round(size_mb, 2)

    def add_query_latency(self, latency_ms: float):
        self._query_latencies.append(latency_ms)
        self.metrics["query"]["total_queries"] = len(self._query_latencies)

    def calculate_latency_percentiles(self):
        if not self._query_latencies:
            return

        sorted_latencies = sorted(self._query_latencies)
        n = len(sorted_latencies)

        # Índices para percentiles (interpolación simple)
        self.metrics["query"]["latencia_p50"] = round(sorted_latencies[int(n * 0.50)], 4)
        self.metrics["query"]["latencia_p95"] = round(sorted_latencies[int(n * 0.95)], 4)
        self.metrics["query"]["latencia_p99"] = round(sorted_latencies[min(int(n * 0.99), n - 1)], 4)

    def save_metrics(self, output_path: Optional[str] = None):
        self.calculate_latency_percentiles()

        # Redondear tiempos finales acumulados para limpieza visual
        self.metrics["build"]["tiempo_encoding_total"] = round(self.metrics["build"]["tiempo_encoding_total"], 4)
        self.metrics["build"]["tiempo_insercion_bd"] = round(self.metrics["build"]["tiempo_insercion_bd"], 4)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_metrics_{timestamp}.json"

        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        print(f"[{self.__class__.__name__}] Métricas guardadas en: {output_path}")
        return output_path


# Singleton
metrics_collector = TestMetricsCollector()