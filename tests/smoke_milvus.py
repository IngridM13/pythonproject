# smoke_milvus.py
from pymilvus import MilvusClient

# 1) Connect
client = MilvusClient(uri="http://localhost:19530")  # Docker exposes 19530

# 2) Create a tiny "quick-setup" collection (auto schema)
#    This creates fields: id (auto_id) + vector (dim=4), metric=COSINE.
client.create_collection(
    collection_name="smoke",
    dimension=4,
    metric_type="COSINE",
    auto_id=True
)

# 3) Insert a few rows (row-based insert). "vector" is the required field name.
data = [
    {"vector": [0.1, 0.2, 0.3, 0.4], "name": "a"},
    {"vector": [0.11, 0.19, 0.29, 0.39], "name": "b"},
    {"vector": [0.9, 0.1, 0.1, 0.1], "name": "far"},
]
ids = client.insert("smoke", data=data)
print("Inserted IDs:", ids)

# 4) Simple similarity search
hits = client.search(
    collection_name="smoke",
    data=[[0.1, 0.2, 0.3, 0.4]],   # one query vector
    limit=3,
    output_fields=["name"]
)

for h in hits[0]:  # first (and only) query
    print("hit:", {"id": h["id"], "distance": h["distance"], "name": h["entity"].get("name")})
