import numpy as np
import sys
import os
from datetime import datetime, date
from sklearn.metrics.pairwise import cosine_similarity

# Add the parent directory to the path so we can import from the main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the typed module
from scripts.encoding_and_search_typed import (
    get_hv, encode_person, encode_date, DIMENSION, store_person,
    normalize_person_data, conn, cursor, pickle,
    find_closest_match_db, parse_date, get_person_details, find_similar_by_date
)

# Global variable needed for the test
hv_dict = {}


def test_encoding_consistency():
    """Verify that encoding is consistent with deterministic approach"""
    print("\n--- Verifying Encoding Consistency ---")

    # Create a test person
    test_person = {
        "name": "Test",
        "lastname": "Person",
        "DOB": "2000-01-01",
        "address": ["123 Test St"],
        "marital_status": "Single",
        "Akas": ["Testy"],
        "landlines": ["555-1234"],
        "mobile_number": "555-5678",
        "gender": "Other",
        "race": "Other"
    }

    # Clear dictionary and encode
    global hv_dict
    hv_dict = {}
    encoding1 = encode_person(test_person)

    # Clear dictionary and encode again
    hv_dict = {}
    encoding2 = encode_person(test_person)

    # Compare
    are_equal = np.array_equal(encoding1, encoding2)
    print(f"Same person encoded twice with reset dictionary - Vectors are equal: {are_equal}")

    if not are_equal:
        diff_count = np.sum(encoding1 != encoding2)
        print(f"Number of different elements: {diff_count} out of {DIMENSION}")
    else:
        print("Deterministic encoding is working correctly!")

    # Verify that different data produces different encodings
    test_person2 = test_person.copy()
    test_person2["name"] = "Different"

    # Encode different person
    hv_dict = {}
    encoding3 = encode_person(test_person2)

    # Compare
    are_different = not np.array_equal(encoding1, encoding3)
    print(f"Different people produce different encodings: {are_different}")

    # Return test results for potential assertion
    return are_equal, are_different


def test_db_encoding_preservation():
    """Test that person data can be stored and retrieved from DB with correct encoding"""
    print("\n--- Testing Database Encoding Preservation ---")

    # Create a test person with a proper date object
    test_person = {
        "name": "John",
        "lastname": "Doe",
        "DOB": "1990-05-15",  # This will be converted to a date object
        "address": ["456 Main St", "Apt 789"],
        "marital_status": "Married",
        "Akas": ["Johnny", "J.D."],
        "landlines": ["555-9876"],
        "mobile_number": "555-1111",
        "gender": "Male",
        "race": "Caucasian"
    }

    # Normalize to convert date string to date object
    test_person = normalize_person_data(test_person)

    print("Original person data:")
    for key, value in test_person.items():
        print(f"  {key}: {value}")

    # Clear the hv_dict to ensure fresh encoding
    global hv_dict
    hv_dict = {}

    # Encode the person
    original_encoding = encode_person(test_person)
    print(f"\nOriginal encoding (first 5 elements): {original_encoding[:5]}")

    # Store the person in the database
    person_id = store_person(test_person)
    print(f"\nPerson stored in database with ID: {person_id}")

    # Retrieve the person from the database using the new get_person_details function
    retrieved_person = get_person_details(person_id)

    if not retrieved_person:
        print("ERROR: Person not found in database!")
        return False

    print("\nRetrieved person data:")
    for key, value in retrieved_person.items():
        print(f"  {key}: {value}")

    # Retrieve the encoding from the database
    cursor.execute("SELECT hdv FROM people_typed WHERE id = %s", (person_id,))
    hdv_binary = cursor.fetchone()[0]
    stored_encoding = pickle.loads(hdv_binary)
    print(f"Retrieved encoding (first 5 elements): {stored_encoding[:5]}")

    # Clear the hv_dict and re-encode the retrieved person
    hv_dict = {}
    recomputed_encoding = encode_person(retrieved_person)
    print(f"\nRecomputed encoding (first 5 elements): {recomputed_encoding[:5]}")

    # Compare encodings
    stored_vs_original = np.array_equal(stored_encoding, original_encoding)
    print(f"\nStored encoding matches original: {stored_vs_original}")

    recomputed_vs_stored = np.array_equal(recomputed_encoding, stored_encoding)
    print(f"Recomputed encoding matches stored: {recomputed_vs_stored}")

    if not stored_vs_original:
        diff_count = np.sum(stored_encoding != original_encoding)
        print(f"Differences between stored and original: {diff_count} out of {DIMENSION}")

    if not recomputed_vs_stored:
        diff_count = np.sum(recomputed_encoding != stored_encoding)
        print(f"Differences between recomputed and stored: {diff_count} out of {DIMENSION}")

    # Clean up - delete the test record
    cursor.execute("DELETE FROM people_typed WHERE id = %s", (person_id,))
    conn.commit()

    return stored_vs_original and recomputed_vs_stored


def test_search_with_encoded_vector():
    """Test searching for a person using an encoded vector with slight variations"""
    print("\n--- Testing Search with Encoded Vector ---")

    # Create original test person with proper date
    original_person = {
        "name": "Jane",
        "lastname": "Smith",
        "DOB": date(1985, 8, 23),  # Using a proper date object
        "address": ["789 Oak Rd", "Suite 456"],
        "marital_status": "Single",
        "Akas": ["J. Smith", "Janie"],
        "landlines": ["555-4321"],
        "mobile_number": "555-8765",
        "gender": "Female",
        "race": "Asian"
    }

    # Clear the dictionary and store the original person
    global hv_dict
    hv_dict = {}
    person_id = store_person(original_person)
    print(f"Original person stored in database with ID: {person_id}")
    for key, value in original_person.items():
        print(f"  {key}: {value}")

    # Create a similar query with some slight differences
    query_person = original_person.copy()
    query_person["name"] = "Jane"  # Same name
    query_person["lastname"] = "Smyth"  # Slight misspelling
    query_person["DOB"] = date(1985, 8, 25)  # 2 days off
    query_person["address"] = ["789 Oak Road"]  # Slightly different address
    query_person["Akas"] = ["J. Smith"]  # Subset of AKAs

    print("\nQuery person (with slight variations):")
    for key, value in query_person.items():
        print(f"  {key}: {value}")

    # Search for the person
    matches = find_closest_match_db(query_person, threshold=0.5)

    print("\nSearch results:")
    for match in matches:
        print(f"Match found: {match}")

    # Check if the right person was found
    is_correct_match = any(match["id"] == person_id for match in matches)
    print(f"Correct person found: {is_correct_match}")

    # Test with a completely different person that shouldn't match as closely
    different_person = {
        "name": "Bob",
        "lastname": "Johnson",
        "DOB": date(1970, 1, 1),  # Much different date
        "address": ["123 Different St"],
        "marital_status": "Married",
        "Akas": ["Robert"],
        "landlines": ["555-9999"],
        "mobile_number": "555-0000",
        "gender": "Male",
        "race": "Caucasian"
    }

    print("\nCompletely different query person:")
    for key, value in different_person.items():
        print(f"  {key}: {value}")

    different_matches = find_closest_match_db(different_person, threshold=0.5)

    print("\nSearch results for different person:")
    if different_matches:
        for match in different_matches:
            print(f"Match found: {match}")

        # Get the similarity score for our original person (if found)
        original_similarity = next((match["similarity"] for match in matches
                                    if match["id"] == person_id), 0)

        # Get the similarity score for the original person in different_matches (if found)
        different_similarity = next((match["similarity"] for match in different_matches
                                     if match["id"] == person_id), 0)

        # The similarity should be lower for the different person
        lower_similarity = different_similarity < original_similarity
        print(f"Different person has lower similarity to original (as expected): {lower_similarity}")
    else:
        print("No matches found for different person (as expected)")
        lower_similarity = True

    # Clean up - delete the test record
    cursor.execute("DELETE FROM people_typed WHERE id = %s", (person_id,))
    conn.commit()

    # Return overall test result
    return is_correct_match and lower_similarity


def test_date_encoding_and_search():
    """Test the special date encoding and date-based search functionality"""
    print("\n--- Testing Date Encoding and Search ---")

    # Create several test people with different dates
    test_people = [
        {
            "name": "Person",
            "lastname": "One",
            "DOB": date(1990, 5, 15),
            "gender": "Male"
        },
        {
            "name": "Person",
            "lastname": "Two",
            "DOB": date(1990, 5, 20),  # 5 days difference
            "gender": "Female"
        },
        {
            "name": "Person",
            "lastname": "Three",
            "DOB": date(1990, 6, 15),  # 1 month difference
            "gender": "Other"
        },
        {
            "name": "Person",
            "lastname": "Four",
            "DOB": date(1991, 5, 15),  # 1 year difference
            "gender": "Male"
        }
    ]

    # Store all test people
    ids = []
    for person in test_people:
        person_id = store_person(person)
        ids.append(person_id)
        print(f"Stored {person['name']} {person['lastname']} (DOB: {person['DOB']}) with ID: {person_id}")

    # Test the date range search
    print("\nSearching for people born in May 1990 (within 15 days of May 15):")
    matches = find_similar_by_date(date(1990, 5, 15), range_days=15)

    for match in matches:
        print(f"  Found: {match['name']} {match['lastname']} (DOB: {match['dob']})")

    # Verify we found the right people
    found_ids = [match['id'] for match in matches]
    expected_ids = [ids[0], ids[1]]  # Should find Person One and Person Two

    correct_date_matches = all(id in found_ids for id in expected_ids) and len(found_ids) == len(expected_ids)
    print(f"Found correct date matches: {correct_date_matches}")

    # Test encoding different dates
    print("\nTesting date encoding similarity:")

    # Encode some dates
    date1 = date(1990, 5, 15)
    date2 = date(1990, 5, 16)  # 1 day difference
    date3 = date(1990, 6, 15)  # 1 month difference
    date4 = date(1991, 5, 15)  # 1 year difference

    # Clear the dictionary for clean test
    hv_dict = {}

    # Encode dates
    enc1 = encode_date(date1)
    enc2 = encode_date(date2)
    enc3 = encode_date(date3)
    enc4 = encode_date(date4)

    # Calculate similarities
    sim_1_day = cosine_similarity([enc1], [enc2])[0][0]
    sim_1_month = cosine_similarity([enc1], [enc3])[0][0]
    sim_1_year = cosine_similarity([enc1], [enc4])[0][0]

    print(f"Similarity with 1 day difference: {sim_1_day:.4f}")
    print(f"Similarity with 1 month difference: {sim_1_month:.4f}")
    print(f"Similarity with 1 year difference: {sim_1_year:.4f}")

    # Closer dates should have higher similarity
    correct_similarity_order = sim_1_day > sim_1_month > sim_1_year
    print(f"Correct similarity order (closer dates have higher similarity): {correct_similarity_order}")

    # Clean up
    for person_id in ids:
        cursor.execute("DELETE FROM people_typed WHERE id = %s", (person_id,))
    conn.commit()

    return correct_date_matches and correct_similarity_order


if __name__ == "__main__":
    consistency_result, differentiation_result = test_encoding_consistency()
    db_preservation_result = test_db_encoding_preservation()
    search_result = test_search_with_encoded_vector()
    date_result = test_date_encoding_and_search()

    # Simple reporting of test results
    print("\n--- Test Results Summary ---")
    if consistency_result and differentiation_result:
        print("✓ Encoding consistency tests PASSED!")
    else:
        print("✗ Some encoding consistency tests FAILED!")
        if not consistency_result:
            print("  - Consistency test failed: Same data produced different encodings")
        if not differentiation_result:
            print("  - Differentiation test failed: Different data produced same encoding")

    if db_preservation_result:
        print("✓ Database encoding preservation test PASSED!")
    else:
        print("✗ Database encoding preservation test FAILED!")

    if search_result:
        print("✓ Vector-based search test PASSED!")
    else:
        print("✗ Vector-based search test FAILED!")

    if date_result:
        print("✓ Date encoding and search tests PASSED!")
    else:
        print("✗ Date encoding and search tests FAILED!")