import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from the main script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the main module
from scripts.encoding_and_search_db import get_hv, encode_person, DIMENSION

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


if __name__ == "__main__":
    consistency_result, differentiation_result = test_encoding_consistency()

    # Simple reporting of test results
    if consistency_result and differentiation_result:
        print("\nAll encoding tests PASSED!")
    else:
        print("\nSome encoding tests FAILED!")
        if not consistency_result:
            print("- Consistency test failed: Same data produced different encodings")
        if not differentiation_result:
            print("- Differentiation test failed: Different data produced same encoding")