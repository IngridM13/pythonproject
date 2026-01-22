import torch
import torch.nn.functional as F
import random

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set a fixed dimension for hypervectors
DIMENSION = 10000 # no debería esto ir en el .env?

# Generate a consistent dictionary of random hypervectors
torch.manual_seed(42)
random.seed(42)
hv_dict = {}


def get_hv(key):
    """Returns a bipolar hypervector {-1, 1} for a given key."""
    if key not in hv_dict:
        # Generate random 0s and 1s, then convert to -1s and 1s
        hv = torch.randint(0, 2, (DIMENSION,), dtype=torch.float32, device=device) * 2 - 1
        hv_dict[key] = hv
    return hv_dict[key]


def encode_person(person):
    """Encodes a person dictionary into a single Hyperdimensional Vector."""
    components = []

    for key, value in person.items():
        # 1. Get the hypervector for the Key (e.g., "name")
        key_hv = get_hv(key)

        # 2. Get/Calculate the hypervector for the Value
        if isinstance(value, list):
            if len(value) > 0:
                # Bundle list elements using sum
                list_hvs = torch.stack([get_hv(str(v)) for v in value])
                encoded_value = torch.sum(list_hvs, dim=0)
            else:
                # Neutral element for an empty list (zeros)
                encoded_value = torch.zeros(DIMENSION, device=device)
        else:
            encoded_value = get_hv(str(value))

        # 3. Bind Key and Value (Element-wise multiplication in bipolar HDC)
        components.append(key_hv * encoded_value)

    # 4. Bundle all fields into one person vector
    person_hv = torch.sum(torch.stack(components), dim=0)
    return person_hv


# Example database
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

# Encode the database and stack into a single matrix [Num_People, Dimension]
encoded_db = torch.stack([encode_person(p) for p in people_db])


def find_closest_match(query_person):
    query_vector = encode_person(query_person)  # Shape: [10000]

    # Calculate cosine similarity between query and all DB entries at once
    # We add a dimension to query to make it [1, 10000] for broadcasting
    similarities = F.cosine_similarity(query_vector.unsqueeze(0), encoded_db)

    best_match_idx = torch.argmax(similarities).item()
    return people_db[best_match_idx], similarities[best_match_idx].item()


# Example query
query = {"name": "Daniella", "lastname": "Rhodes", "DOB": "1950-03-03",
         "address": ["9402 Peterson Drives, Port Matthew, CO 50298", "407 Teresa Lane Apt. 849, Barbaraland, AZ 87174"],
         "marital_status": "Divorced", "Akas": ["Sonya Johnston", "John Russell"], "landlines": ["538.372.6247"],
         "mobile_number": "(613)354-2784x980", "gender": "Other", "race": "Other"}

# Find the closest match
best_match, similarity_score = find_closest_match(query)

print(f"Closest match found: {best_match['name']} {best_match['lastname']}")

if similarity_score < 0.7:
    print(f"No good match found. Closest score: {similarity_score:.4f}")
else:
    print(f"Similarity Score: {similarity_score:.4f} (Inference: Same person)")