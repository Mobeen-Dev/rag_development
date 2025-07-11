from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, CollectionStatus  # noqa: F401
from sentence_transformers import SentenceTransformer
import uuid
from langchain_core.documents import Document

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to local Qdrant
client = QdrantClient(host="localhost", port=6333)

def create_collection_if_not_exists(collection_name: str, vector_size: int = 384):
    """Creates Qdrant collection if it doesn't already exist."""
    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        return  # already exists

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

def embed_documents(docs: List[Document], collection_name: str = "rag_collection", batch_size: int = 100):
    """
    Converts documents into embeddings and stores them in a local Qdrant collection.
    """
    create_collection_if_not_exists(collection_name)

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        contents = [doc.page_content for doc in batch]
        embeddings = model.encode(contents, show_progress_bar=False)

        points = []
        for doc, vector in zip(batch, embeddings):
            point_id = str(uuid.uuid4())
            payload = {
                "text": doc.page_content,
                **doc.metadata  # safe fallback
            }
            points.append(PointStruct(id=point_id, vector=vector.tolist(), payload=payload))

        # Upsert this batch
        client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Upserted batch {i // batch_size + 1} ({len(points)} items)")

    print(f"🎉 Done. Total documents upserted: {len(docs)}")
