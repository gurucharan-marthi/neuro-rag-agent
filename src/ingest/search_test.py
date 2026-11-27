"""Quick check: search the index with a question."""
import sys
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

query = sys.argv[1] if len(sys.argv) > 1 else "tractography of the optic radiation"
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(path="data/qdrant")

vec = model.encode(query, normalize_embeddings=True).tolist()
hits = client.query_points("papers", query=vec, limit=5).points
for h in hits:
    print(f"{h.score:.3f}  {h.payload['title']}  ({h.payload['url']})")

client.close()