"""
Tests para verificar la invariancia del orden de las listas en la codificación HDC.

Este módulo contiene tres tests independientes que verifican que el sistema HDC
puede identificar correctamente como similares a dos personas que tienen los mismos
datos pero con listas (direcciones, números de teléfono, etc.) en diferente orden.
"""

import os
import pytest
import torch
import torch.nn.functional as F
import time
from datetime import datetime

from database_utils.milvus_db_connection import get_vector_mode
from encoding_methods.encoding_and_search_milvus import (
    encode_person, store_person, find_closest_match_db
)
from utils.person_data_normalization import normalize_person_data
from tests.test_encoding_typed import _delete_people_from_milvus


# Datos de prueba comunes
def create_test_data():
    # Persona 1 - Datos originales con un orden específico de listas
    persona1 = {
        "name": "Danielle",
        "lastname": "Rhodes",
        "dob": "1951-02-02",
        "marital_status": "Widowed",
        "mobile_number": "582.208.1219x13619",
        "gender": "Non-binary",
        "race": "Mixed",
        "attrs": {
            "address": [
                "9402 Peterson Drives, Port Matthew, CO 50298",
                "407 Teresa Lane Apt. 849, Barbaraland, AZ 87174",
                "31647 Martin Knoll Apt. 419, New Jessica, GA 61090",
                "503 Linda Locks, Carlshire, FM 94599",
                "7242 Julie Plain Suite 969, Coxberg, NY 65187"
            ],
            "akas": ["Dany Rhoades"],
            "landlines": ["5808132677", "9264064746"]
        }
    }

    # Persona 2 - Los mismos datos pero con las listas en diferente orden
    persona2 = {
        "name": "Danielle",
        "lastname": "Rhodes",
        "dob": "1951-02-02",
        "marital_status": "Widowed",
        "mobile_number": "582.208.1219x13619",
        "gender": "Non-binary",
        "race": "Mixed",
        "attrs": {
            "address": [
                "503 Linda Locks, Carlshire, FM 94599",
                "31647 Martin Knoll Apt. 419, New Jessica, GA 61090",
                "9402 Peterson Drives, Port Matthew, CO 50298",
                "7242 Julie Plain Suite 969, Coxberg, NY 65187",
                "407 Teresa Lane Apt. 849, Barbaraland, AZ 87174"
            ],
            "akas": ["Dany Rhoades"],
            "landlines": ["9264064746", "5808132677"]  # Orden diferente
        }
    }

    # Persona 3 - Datos diferentes para control negativo
    persona3 = {
        "name": "Robert",
        "lastname": "Johnson",
        "dob": "1965-07-12",
        "marital_status": "Married",
        "mobile_number": "123.456.7890",
        "gender": "Male",
        "race": "Caucasian",
        "attrs": {
            "address": [
                "123 Main Street, Springfield, IL 62701",
                "456 Oak Avenue, Chicago, IL 60601"
            ],
            "akas": ["Bob", "Bobby J"],
            "landlines": ["987.654.3210"]
        }
    }

    # Persona 4 - Mismos datos que persona1 pero con una dirección menos y un AKA adicional
    persona4 = {
        "name": "Danielle",
        "lastname": "Rhodes",
        "dob": "1951-02-02",
        "marital_status": "Widowed",
        "mobile_number": "582.208.1219x13619",
        "gender": "Non-binary",
        "race": "Mixed",
        "attrs": {
            "address": [  # Una dirección menos
                "9402 Peterson Drives, Port Matthew, CO 50298",
                "407 Teresa Lane Apt. 849, Barbaraland, AZ 87174",
                "31647 Martin Knoll Apt. 419, New Jessica, GA 61090",
                "503 Linda Locks, Carlshire, FM 94599",
            ],
            "akas": ["Dany Rhoades", "Dani R"],  # Un AKA adicional
            "landlines": ["5808132677", "9264064746"]  # Igual
        }
    }

    return persona1, persona2, persona3, persona4


@pytest.mark.skipif(
    os.getenv("SKIP_MILVUS_TESTS", "True") == "True",
    reason="Requiere Milvus en ejecución."
)
@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
def test_vector_similarity_with_list_order(with_vector_mode):
    """
    Test 1: Verifica que la codificación vectorial es similar entre registros
    que tienen las mismas listas pero en diferente orden.
    """
    # Verificar el modo vector
    current_mode = get_vector_mode()
    assert current_mode == with_vector_mode, f"Expected mode {with_vector_mode}, got {current_mode}"

    print(f"\nTest de similitud vectorial con listas en diferente orden (modo: {with_vector_mode})")

    # Usar GPU si está disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Obtener datos de prueba
    persona1, persona2, persona3, _ = create_test_data()

    # Reset de cache global para asegurar codificaciones limpias
    global_dict_name = 'hv_dict'
    if global_dict_name in globals():
        globals()[global_dict_name] = {}

    with torch.no_grad():  # Desactivar tracking de gradientes
        # Normalizar y codificar las personas
        normalizado1 = normalize_person_data(persona1)
        normalizado2 = normalize_person_data(persona2)
        normalizado3 = normalize_person_data(persona3)

        # Medir tiempo de codificación
        start_time = time.time()
        encoding1 = encode_person(normalizado1, mode=with_vector_mode)
        encoding2 = encode_person(normalizado2, mode=with_vector_mode)
        encoding3 = encode_person(normalizado3, mode=with_vector_mode)
        encoding_time = time.time() - start_time

        # Convertir a tensores si es necesario
        if not isinstance(encoding1, torch.Tensor):
            encoding1 = torch.tensor(encoding1, device=device)
        else:
            encoding1 = encoding1.to(device)

        if not isinstance(encoding2, torch.Tensor):
            encoding2 = torch.tensor(encoding2, device=device)
        else:
            encoding2 = encoding2.to(device)

        if not isinstance(encoding3, torch.Tensor):
            encoding3 = torch.tensor(encoding3, device=device)
        else:
            encoding3 = encoding3.to(device)

        # Asegurar que los tensores son float para el cálculo de similitud
        encoding1_f = encoding1.float()
        encoding2_f = encoding2.float()
        encoding3_f = encoding3.float()

        # Calcular similitud coseno entre las codificaciones
        sim_1_2 = F.cosine_similarity(encoding1_f.unsqueeze(0), encoding2_f.unsqueeze(0)).item()
        sim_1_3 = F.cosine_similarity(encoding1_f.unsqueeze(0), encoding3_f.unsqueeze(0)).item()

        # Solo imprimir información esencial
        print(f"Tiempo de codificación: {encoding_time:.4f} segundos")
        print(f"Similitud entre persona original y persona con listas reordenadas: {sim_1_2:.6f}")
        print(f"Similitud entre persona original y persona diferente (control): {sim_1_3:.6f}")

        # Verificaciones
        alta_similitud = sim_1_2 > 0.95
        baja_similitud_control = sim_1_3 < 0.9

    # Aserciones
    assert alta_similitud, f"La similitud entre mismas personas con listas reordenadas debería ser alta (>0.95), es {sim_1_2:.6f}"
    assert baja_similitud_control, f"La similitud con una persona diferente debería ser baja (<0.9), es {sim_1_3:.6f}"


@pytest.mark.skipif(
    os.getenv("SKIP_MILVUS_TESTS", "True") == "True",
    reason="Requiere Milvus en ejecución."
)
@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
def test_database_search_with_reordered_lists(with_vector_mode, test_collection):
    """
    Test 2: Verifica que la búsqueda en la base de datos encuentra
    correctamente registros con listas en diferente orden.
    """
    # Verificar el modo vector
    current_mode = get_vector_mode()
    assert current_mode == with_vector_mode, f"Expected mode {with_vector_mode}, got {current_mode}"

    print(f"\nTest de búsqueda en BD con listas en diferente orden (modo: {with_vector_mode})")

    # Obtener datos de prueba
    persona1, persona2, _, _ = create_test_data()

    # Almacenar la primera persona en la BD
    persona1_id = store_person(normalize_person_data(persona1))
    print(f"Persona original almacenada con ID: {persona1_id}")

    # Buscar utilizando la segunda persona (mismos datos, listas reordenadas)
    print("Buscando coincidencias usando persona con listas reordenadas...")
    start_time = time.time()
    matches = find_closest_match_db(persona2, threshold=0.85)  # Umbral más bajo para garantizar resultados
    search_time = time.time() - start_time

    print(f"Tiempo de búsqueda: {search_time:.4f} segundos")
    print(f"Coincidencias encontradas: {len(matches)}")

    # Verificar si se encontró la persona original
    persona_encontrada = False
    match_similarity = 0
    for match in matches:
        if match.get('id') == persona1_id:
            persona_encontrada = True
            match_similarity = match.get('similarity', 0)
            print(f"Coincidencia encontrada - ID: {match.get('id')}, Similitud: {match_similarity:.6f}")

    # Limpiar después de la prueba
    _delete_people_from_milvus([persona1_id])

    # Aserción
    assert persona_encontrada, "La búsqueda debería encontrar la persona original a pesar del reordenamiento de listas"
    assert match_similarity > 0.85, f"La similitud encontrada debería ser alta (>0.85), es {match_similarity:.6f}"


@pytest.mark.skipif(
    os.getenv("SKIP_MILVUS_TESTS", "True") == "True",
    reason="Requiere Milvus en ejecución."
)
@pytest.mark.parametrize("with_vector_mode", ["binary", "float"], indirect=True)
def test_robustness_with_modified_lists(with_vector_mode, test_collection):
    """
    Test 3: Evalúa la robustez del sistema ante modificaciones menores en las listas,
    como tener un elemento menos o un elemento adicional.
    """
    # Verificar el modo vector
    current_mode = get_vector_mode()
    assert current_mode == with_vector_mode, f"Expected mode {with_vector_mode}, got {current_mode}"

    print(f"\nTest de robustez con listas modificadas (modo: {with_vector_mode})")

    # Obtener datos de prueba
    persona1, _, _, persona4 = create_test_data()

    # Almacenar la primera persona en la BD
    persona1_id = store_person(normalize_person_data(persona1))
    print(f"Persona original almacenada con ID: {persona1_id}")

    # Realizar una sola búsqueda con un umbral bajo para garantizar encontrar resultados
    # Luego analizaremos la similitud obtenida para determinar la robustez
    print("Buscando coincidencias usando persona con modificaciones menores...")
    start_time = time.time()

    # Usamos un umbral bajo para asegurar encontrar resultados
    threshold = 0.7
    matches = find_closest_match_db(persona4, threshold=threshold)

    search_time = time.time() - start_time
    print(f"Tiempo de búsqueda: {search_time:.4f} segundos")
    print(f"Coincidencias encontradas: {len(matches)}")

    # Verificar si se encontró la persona original y su similitud
    found_match = None
    for match in matches:
        if match.get('id') == persona1_id:
            found_match = match
            print(f"Coincidencia encontrada - ID: {match.get('id')}, Similitud: {match.get('similarity', 0):.6f}")
            break

    # Analizar el resultado
    persona_encontrada = found_match is not None
    match_similarity = found_match.get('similarity', 0) if found_match else 0

    # Limpiar después de la prueba
    _delete_people_from_milvus([persona1_id])

    # Aserciones
    assert persona_encontrada, "La búsqueda debería encontrar la persona original a pesar de las modificaciones en las listas"

    # Verificar que la similitud sea razonable para una persona con listas modificadas
    # Esperamos una similitud menor que con listas idénticas, pero aún así significativa
    assert match_similarity > 0.7, f"La similitud con modificaciones menores debería ser al menos 0.7, es {match_similarity:.6f}"

    # La similitud típica debería ser menor que con listas idénticas pero reordenadas (que normalmente sería >0.95)
    # En el caso de modificaciones menores, esperamos ver cierta degradación, pero no demasiada
    print(f"Similitud obtenida con listas modificadas: {match_similarity:.6f}")

    # No verificamos un límite superior estricto, pero incluimos un mensaje informativo si es muy alta
    if match_similarity > 0.98:
        print("NOTA: La similitud es inusualmente alta para listas modificadas. Esto podría indicar que el sistema")
        print("no está siendo suficientemente sensible a cambios en las listas.")


if __name__ == "__main__":
    # Código para ejecutar los tests directamente (sin pytest)
    from database_utils.milvus_db_connection import VECTOR_MODE

    # Generar un nombre de colección único para pruebas directas
    test_collection_name = f"test_collection_direct_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    print("\n=== Ejecutando tests de forma independiente ===")

    # Test 1
    print("\n[TEST 1] Similitud vectorial")
    test_vector_similarity_with_list_order(VECTOR_MODE)

    # Test 2
    print("\n[TEST 2] Búsqueda en BD")
    test_database_search_with_reordered_lists(VECTOR_MODE, test_collection_name)

    # Test 3
    print("\n[TEST 3] Robustez ante modificaciones")
    test_robustness_with_modified_lists(VECTOR_MODE, test_collection_name)

    print("\n=== Tests completados ===")