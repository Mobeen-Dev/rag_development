from typing import List, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field, BaseModel

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines results from Qdrant and OpenSearch.
    """
    qdrant_retriever: BaseRetriever = Field(..., description="Vector retriever from Qdrant")
    opensearch_retriever: BaseRetriever = Field(..., description="Lexical retriever from OpenSearch")
    top_k: int = Field(10, description="Number of documents to return")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Get relevant documents from both retrievers and combine the results.
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            Combined list of documents
        """
        # Get documents from both retrievers
        qdrant_docs = self.qdrant_retriever.invoke(query)
        opensearch_docs = self.opensearch_retriever.invoke(query)
        
        # Combine results without duplicates
        seen = set()
        combined = []
        
        # Process results from both retrievers
        for doc in qdrant_docs + opensearch_docs:
            # Use content as the key for deduplication
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                combined.append(doc)
        
        # Return top_k documents
        return combined[:self.top_k]