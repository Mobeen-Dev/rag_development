from lexical_search import create_opensearch_retriever, create_opensearch_client
from embedding_store import QdrantRetriever
from hybrid_retrieval import HybridRetriever  # or wherever HybridRetriever is defined

def get_hybrid_retriever():
    qdrant_retriever = QdrantRetriever(
        collection_name="product_chunks",
        limit=5,
        metadata_filter={"source": "docs"}
    )

    opensearch_client = create_opensearch_client(use_ssl=True)
    opensearch_retriever = create_opensearch_retriever(
        client=opensearch_client,
        index_name="product_data_lexical",
        k=5,
        fields=["page_content"]
    )

    hybrid_retriever = HybridRetriever(
        qdrant_retriever=qdrant_retriever,
        opensearch_retriever=opensearch_retriever,
        top_k=8
    )
    
    return hybrid_retriever

def output_reponse(results):
    # Display results
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content}\n")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        if "score" in doc.metadata:
            print(f"Score: {doc.metadata['score']}")
