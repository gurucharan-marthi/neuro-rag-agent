"""Embed paper abstracts and store them in a local Qdrant collection."""
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DATA = "data/raw/arxiv.jsonl"
DB_PATH = "data/qdrant"
COLLECTION = "papers"
MODEL = "BAAI/bge-small-en-v1.5"

# 1. Load papers
papers = [json.loads(line) for line in open(DATA)]
texts = [f"{p['title']}. {p['abstract']}" for p in papers]
print(f"Loaded {len(papers)} papers")

# 2. Embed (uses GPU automatically if available)
model = SentenceTransformer(MODEL)
vectors = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True)

# 3. Create collection (delete old one if it exists)
client = QdrantClient(path=DB_PATH)
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    COLLECTION,
    vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
)

# 4. Upload: vector + metadata ("payload") for each paper
client.upsert(COLLECTION, points=[
    PointStruct(id=i, vector=v.tolist(), payload=p)
    for i, (v, p) in enumerate(zip(vectors, papers))
])
print(f"Indexed {client.count(COLLECTION).count} papers into {DB_PATH}")

client.close()