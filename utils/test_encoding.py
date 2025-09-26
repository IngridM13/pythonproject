import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from the main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main module
from scripts.encoding_and_search_db import (
    get_hv, encode_person, DIMENSION, store_person, 
    normalize_person_data, conn, cursor, pickle,
    find_closest_match_db
)

# Global variable needed for the test
hv_dict = {}


def test_encoding_consistency():
    """Verify that encoding is consistent with deterministic approach"""
    print("\n--- Verifying Encoding Consistency ---")

    # Create a test person with lowercase keys
    test_person = {
        "name": "Test",
        "lastname": "Person",
        "dob": "2000-01-01",
        "address": ["123 Test St"],
        "marital_status": "Single",
        "akas": ["Testy"],
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
    
    # Create a test person
    test_person = {
        "name": "John",
        "lastname": "Doe",
        "dob": "1990-05-15",
        "address": ["456 Main St", "Apt 789"],
        "marital_status": "Married",
        "akas": ["Johnny", "J.D."],
        "landlines": ["555-9876"],
        "mobile_number": "555-1111",
        "gender": "Male",
        "race": "Caucasian"
    }
    
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
    store_person(test_person)
    print("\nPerson stored in database")
    
    # Retrieve the person from the database
    cursor.execute("""
        SELECT name, lastname, dob, address, marital_status, akas, 
               landlines, mobile_number, gender, race, hdv
        FROM people 
        WHERE name = %s AND lastname = %s
    """, (test_person["name"], test_person["lastname"]))
    
    result = cursor.fetchone()
    
    if not result:
        print("ERROR: Person not found in database!")
        return False
    
    # Reconstruct the person dictionary
    retrieved_person = {
        "name": result[0],
        "lastname": result[1],
        "DOB": result[2],
        "address": result[3],
        "marital_status": result[4],
        "Akas": result[5],
        "landlines": result[6],
        "mobile_number": result[7],
        "gender": result[8],
        "race": result[9]
    }
    
    # Get the stored encoding
    stored_encoding = pickle.loads(result[10])
    
    print("\nRetrieved person data:")
    for key, value in retrieved_person.items():
        print(f"  {key}: {value}")
    print(f"Retrieved encoding (first 5 elements): {stored_encoding[:5]}")
    
    # Clear the hv_dict and re-encode the retrieved person
    hv_dict = {}
    recomputed_encoding = encode_person(normalize_person_data(retrieved_person))
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
    cursor.execute("DELETE FROM people WHERE name = %s AND lastname = %s", 
                  (test_person["name"], test_person["lastname"]))
    conn.commit()
    
    return stored_vs_original and recomputed_vs_stored


def test_search_with_encoded_vector():
    """Test searching for a person using an encoded vector with slight variations"""
    print("\n--- Testing Search with Encoded Vector ---")
    
    # Create original test person
    original_person = {
        "name": "Jane",
        "lastname": "Smith",
        "dob": "1985-08-23",
        "address": ["789 Oak Rd", "Suite 456"],
        "marital_status": "Single",
        "akas": ["J. Smith", "Janie"],
        "landlines": ["555-4321"],
        "mobile_number": "555-8765",
        "gender": "Female",
        "race": "Asian"
    }

    # Clear the dictionary and store the original person
    global hv_dict
    hv_dict = {}
    store_person(original_person)
    print("Original person stored in database:")
    for key, value in original_person.items():
        print(f"  {key}: {value}")
    
    # Create a similar query with some slight differences
    query_person = original_person.copy()
    query_person["name"] = "Jane"  # Same name
    query_person["lastname"] = "Smyth"  # Slight misspelling
    query_person["dob"] = "1985-08-23"  # Same DOB
    query_person["address"] = ["789 Oak Road"]  # Slightly different address
    query_person["akas"] = ["J. Smith"]  # Subset of AKAs
    
    print("\nQuery person (with slight variations):")
    for key, value in query_person.items():
        print(f"  {key}: {value}")
    
    # Search for the person
    match, similarity = find_closest_match_db(query_person, threshold=0.5)
    
    print("\nSearch results:")
    print(f"Match found: {match}")
    print(f"Similarity score: {similarity}")
    
    # Check if the right person was found
    is_correct_match = match and match.get("name") == original_person["name"] and match.get("lastname") == original_person["lastname"]
    print(f"Correct person found: {is_correct_match}")
    
    # Test with a completely different person that shouldn't match
    different_person = {
        "name": "Bob",
        "lastname": "Johnson",
        "dob": "1970-01-01",
        "address": ["123 Different St"],
        "marital_status": "Married",
        "akas": ["Robert"],
        "landlines": ["555-9999"],
        "mobile_number": "555-0000",
        "gender": "Male",
        "race": "Caucasian"
    }
    
    print("\nCompletely different query person:")
    for key, value in different_person.items():
        print(f"  {key}: {value}")
    
    different_match, different_similarity = find_closest_match_db(different_person, threshold=0.7)
    
    print("\nSearch results for different person:")
    print(f"Match found: {different_match}")
    print(f"Similarity score: {different_similarity}")
    
    # The similarity should be lower for the different person
    lower_similarity = different_similarity < similarity
    print(f"Different person has lower similarity (as expected): {lower_similarity}")
    
    # Clean up - delete the test record
    cursor.execute("DELETE FROM people WHERE name = %s AND lastname = %s", 
                  (original_person["name"], original_person["lastname"]))
    conn.commit()
    
    # Return overall test result
    return is_correct_match and (different_match is None or lower_similarity)


if __name__ == "__main__":
    consistency_result, differentiation_result = test_encoding_consistency()
    db_preservation_result = test_db_encoding_preservation()
    search_result = test_search_with_encoded_vector()

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
        print("✗ Vector-based search test FAILED!dane")