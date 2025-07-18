from langchain.schema import Document
from langchain.retrievers import BaseRetriever
from typing import List, Optional, Dict, Any
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import heapq

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining Qdrant (dense) and OpenSearch (lexical).
    """
    qdrant_retriever: BaseRetriever
    opensearch_retriever: BaseRetriever
    top_k: int = 10

    def __init__(
        self,
        qdrant_retriever: BaseRetriever,
        opensearch_retriever: BaseRetriever,
        top_k: int = 10
    ):
        super().__init__()
        self.qdrant_retriever = qdrant_retriever
        self.opensearch_retriever = opensearch_retriever
        self.top_k = top_k

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # Fetch results from both retrievers
        qdrant_docs = self.qdrant_retriever.get_relevant_documents(query)
        opensearch_docs = self.opensearch_retriever.get_relevant_documents(query)

        # Combine and deduplicate based on page_content
        seen = set()
        combined = []

        for doc in qdrant_docs + opensearch_docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                combined.append(doc)

        # Optionally sort or limit (e.g., by score in metadata if available)
        def get_score(doc):
            return doc.metadata.get("score", 0)

        # Top-k by score (if score exists)
        combined_sorted = sorted(combined, key=get_score, reverse=True)[:self.top_k]

        return combined_sorted
