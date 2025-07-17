from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, CollectionStatus, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import uuid
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

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
                **doc.metadata  # unpack metadata
            }
            points.append(PointStruct(id=point_id, vector=vector.tolist(), payload=payload))

        # Upsert this batch
        client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Upserted batch {i // batch_size + 1} ({len(points)} items)")

    print(f"🎉 Done. Total documents upserted: {len(docs)}")

def retrieve_documents(
    query: str,
    collection_name: str = "product_chunks",
    limit: int = 10,
    score_threshold: Optional[float] = None,
    filter_condition: Optional[Filter] = None
) -> List[Document]:
    """
    Retrieve documents from Qdrant that are similar to the query.
    
    Args:
        query: The search query
        collection_name: Name of the Qdrant collection
        limit: Maximum number of results to return
        score_threshold: Minimum similarity score threshold (0 to 1)
        filter_condition: Optional Qdrant filter condition
        
    Returns:
        List of LangChain Document objects
    """
    # Ensure the collection exists
    create_collection_if_not_exists(collection_name)
    
    # Generate embedding for the query
    query_embedding = model.encode(query)
    
    # Search Qdrant
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=limit,
        score_threshold=score_threshold,
        filter=filter_condition
    )
    
    # Convert to LangChain Documents
    documents = []
    for result in search_result:
        payload = result.payload or {}
        
        # Extract the text content
        if payload and "text" in payload:
            page_content = payload.pop("text")
        else:
            # If "text" field is missing, use empty string or handle as needed
            page_content = ""
        
        # Add score to metadata
        payload["score"] = result.score
        
        # Create Document
        document = Document(
            page_content=page_content,
            metadata=payload
        )
        documents.append(document)
    
    return documents

def filter_by_metadata(metadata_filter: Dict[str, Any]) -> Filter:
    """
    Create a Qdrant filter from a metadata dictionary.
    
    Args:
        metadata_filter: Dictionary of metadata key-value pairs to filter by
        
    Returns:
        Qdrant Filter object
    """
    conditions = []
    for field, value in metadata_filter.items():
        if isinstance(value, list):
            # For lists, create individual field conditions for each value
            field_conditions = []
            for v in value:
                field_conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=v))
                )
            # Use Filter constructor with proper type handling
            conditions.append(Filter(should=field_conditions))
        else:
            # For single values, create "must" condition (AND)
            conditions.append(
                FieldCondition(key=field, match=MatchValue(value=value))
            )
    return Filter(must=conditions)

class QdrantRetriever(BaseRetriever):
    """
    LangChain-compatible retriever for Qdrant vector database.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_collection",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        embedding_model = None,
        qdrant_client = None
    ):
        """
        Initialize the Qdrant retriever.
        
        Args:
            collection_name: Name of the Qdrant collection
            limit: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold (0 to 1)
            metadata_filter: Optional metadata filter as a dictionary
            embedding_model: Custom embedding model (defaults to global model)
            qdrant_client: Custom Qdrant client (defaults to global client)
        """
        super().__init__()
        self.collection_name = collection_name
        self.limit = limit
        self.score_threshold = score_threshold
        self.metadata_filter = metadata_filter
        self.embedding_model = embedding_model or model
        self.client = qdrant_client or client
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get relevant documents from Qdrant.
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Create filter if metadata filter is provided
        filter_condition = None
        if self.metadata_filter:
            filter_condition = filter_by_metadata(self.metadata_filter)
        
        # Retrieve documents
        return retrieve_documents(
            query=query,
            collection_name=self.collection_name,
            limit=self.limit,
            score_threshold=self.score_threshold,
            filter_condition=filter_condition
        )

# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        Document(page_content="Qdrant is a vector database for AI applications", metadata={"source": "docs"}),
        Document(page_content="Vector databases store embeddings for semantic search", metadata={"source": "docs"}),
        Document(page_content="LangChain provides tools for building LLM applications", metadata={"source": "blog"})
    ]
    
    # Index documents
    embed_documents(documents)
    
    # Option 1: Use the standalone retrieve_documents function
    results = retrieve_documents(
        query="How do vector databases work?",
        limit=2
    )
    
    print("\nResults using retrieve_documents function:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Score: {doc.metadata.get('score', 'N/A')}")
    
    # Option 2: Use the LangChain-compatible QdrantRetriever
    retriever = QdrantRetriever(
        collection_name="rag_collection",
        limit=2,
        metadata_filter={"source": "docs"}
    )
    
    results = retriever.get_relevant_documents("What is semantic search?")
    
    print("\nResults using QdrantRetriever:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Score: {doc.metadata.get('score', 'N/A')}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")