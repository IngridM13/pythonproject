import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

# Set a fixed dimension for hypervectors
DIMENSION = 10000 # no debería esto ir en el .env?

# Generate a consistent dictionary of random hypervectors
random.seed(42)
hv_dict = {}


def get_hv(key):
    if key not in hv_dict:
        hv_dict[key] = np.random.choice([-1, 1], DIMENSION)  # Binary bipolar representation
    return hv_dict[key]


# Function to encode a person's data into an HD
def encode_person(person):
    components = []

    for key, value in person.items():
        if isinstance(value, list):
            encoded_value = sum(get_hv(str(v)) for v in value)  # Bundle list elements
        else:
            encoded_value = get_hv(str(value))

        components.append(get_hv(key) * encoded_value)  # Bind key and value

    return sum(components)  # Bundle all components


# Example database (HDV storage)
people_db = [
    {"name": "Danielle", "lastname": "Rhodes", "DOB": "1950-03-03",
     "address": ["9402 Peterson Drives, Port Matthew, CO 50298", "407 Teresa Lane Apt. 849, Barbaraland, AZ 87174"],
     "marital_status": "Divorced", "Akas": ["Sonya Johnston", "John Russell"], "landlines": ["538.372.6247"],
     "mobile_number": "(613)354-2784x980", "gender": "Other", "race": "Other"},

    {"name": "Angel", "lastname": "Mcclain", "DOB": "1971-02-06",
     "address": ["31647 Martin Knoll Apt. 419, New Jessica, GA 61090", "503 Linda Locks, Carlshire, FM 94599",
                 "7242 Julie Plain Suite 969, Coxberg, NY 65187"],
     "marital_status": "Widowed", "Akas": [], "landlines": ["581.808.0132x677", "9264064746"],
     "mobile_number": "(541)718-2449", "gender": "Female", "race": "Black"},
]

# Encode the database
encoded_db = [encode_person(person) for person in people_db]


def find_closest_match(query_person):
    query_vector = encode_person(query_person)
    similarities = cosine_similarity([query_vector], encoded_db)[0]
    best_match_idx = np.argmax(similarities)
    return people_db[best_match_idx], similarities[best_match_idx]


# Example query
query = {"name": "Daniella", "lastname": "Rhodes", "DOB": "1950-03-03",
     "address": ["9402 Peterson Drives, Port Matthew, CO 50298", "407 Teresa Lane Apt. 849, Barbaraland, AZ 87174"],
     "marital_status": "Divorced", "Akas": ["Sonya Johnston", "John Russell"], "landlines": ["538.372.6247"],
     "mobile_number": "(613)354-2784x980", "gender": "Other", "race": "Other"}

# Find the closest match
best_match, similarity_score = find_closest_match(query)
print("Closest match found:", best_match)

if similarity_score < 0.7:  # Example threshold
    print("No good match found, the closest has a similarity score below the 0.7 threshold: ", similarity_score)
else: print("Similarity Score good enough to infer this is the same person:", similarity_score)
