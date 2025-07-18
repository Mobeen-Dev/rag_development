
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional, Dict, Any, Annotated
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import heapq
from hybrid_retrieval import HybridRetriever
# This is just a demonstration - you would need actual retrievers
from langchain_community.retrievers import TFIDFRetriever


from pre_processing import load_and_split_csv_files

from lexical_search import create_opensearch_retriever, lexical_search_create, create_opensearch_client

from embedding_store import QdrantRetriever

# from hr import HybridRetriever, create_hybrid_retriever



qdrant_retriever = QdrantRetriever(
    collection_name="product_chunks",
    limit=2,
    metadata_filter={"source": "docs"}
)

opensearch_client = create_opensearch_client(use_ssl=True)
opensearch_retriever = create_opensearch_retriever(
    client=opensearch_client,
    index_name="product_data_lexical",
    k=10,
    fields=["page_content"]
)

# Combine into hybrid
hybrid_retriever = HybridRetriever(
    qdrant_retriever=qdrant_retriever,
    opensearch_retriever=opensearch_retriever,
    top_k=10
)

# Use it like any LangChain retriever
docs = hybrid_retriever.invoke("keybord")
for doc in docs:
    print(doc.page_content[:150])



# print("Creating hybrid retriever...")
# hybrid_retriever = create_hybrid_retriever(
#     vector_retriever=qdrant_retriever,
#     lexical_retriever=opensearch_retriever,
#     top_k=5
# )

# Test the retrieval
query = "What are the best wireless headphones?"
print(f"\nSearching for: '{query}'")
results = hybrid_retriever.invoke(query)

# Display results
print(f"Found {len(results)} results:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: {doc.page_content[:150]}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    if "score" in doc.metadata:
        print(f"Score: {doc.metadata['score']}")


