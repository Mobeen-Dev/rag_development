import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from opensearchpy import OpenSearch

def create_opensearch_client():
    """
    Creates and returns an OpenSearch client instance for a local Docker setup.
    """
    # For a local Docker setup, default host and port are usually sufficient
    
    # host = os.getenv('OPENSEARCH_HOST', 'localhost')
    host = 'localhost'
    # port = int(os.getenv('OPENSEARCH_PORT', 9200))
    port = 9200  # Default OpenSearch port
    
    auth = ('admin', 'fjkfh1471947y7T&^FV%D(&^T*')  # Default credentials for OpenSearch Docker
    
    # You might need to disable SSL verification for local self-signed certificates
    # In production, always use proper SSL/TLS and authentication.
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_auth=auth,
        use_ssl=True,  # Use SSL if your Docker setup has it enabled (default for recent versions)
        verify_certs=False,  # Disable cert verification for local development (NOT for production)
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        timeout=30 # Set a timeout for requests
    )
    return client

def create_opensearch_index(client: OpenSearch, index_name: str):
    """
    Creates an OpenSearch index with a basic mapping for lexical search.
    """
    index_body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "default": { # Using "default" analyzer for simplicity, applies to all text fields
                        "type": "standard",
                        "stopwords": "english" # Example: remove common English stopwords
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "page_content": {"type": "text"},
                "source": {"type": "keyword"}, # For metadata like file path
                "row": {"type": "long"}, # For metadata like row number
                # Add other metadata fields from your chunks if they are present
            }
        }
    }
    # Ignore 400 (bad request) if the index already exists
    response = client.indices.create(index=index_name, body=index_body, ignore=400) # type: ignore
    if response.get('acknowledged'):
        print(f"Index '{index_name}' created successfully.")
    elif 'error' in response:
        print(f"Error creating index '{index_name}': {response['error']['reason']}")
    else:
        print(f"Index '{index_name}' already exists or unknown response.")

def index_documents_to_opensearch(client: OpenSearch, index_name: str, documents: list):
    """
    Indexes a list of LangChain Document chunks into OpenSearch.
    """
    if not documents:
        print("No documents to index.")
        return
    print(f"Attempting to index {len(documents)} documents into OpenSearch index '{index_name}'...")
    
    # Batch indexing for efficiency
    actions = []
    for i, doc in enumerate(documents):
        # The _id for each document should be unique.
        # LangChain documents have page_content and metadata.
        # You might want to include all metadata fields in the document source.
        doc_body = {
        "page_content": doc.page_content,
        **(doc.metadata or {})  # safely unpack
        }
        
        actions.append({ "index": { "_index": index_name, "_id": f"{index_name}_{i}" } })
        actions.append(doc_body)
        # Bulk index every 1000 documents or at the end
        if len(actions) % 1000 == 0 or i == len(documents) - 1:
            try:
                response = client.bulk(body=actions, refresh=True)  # type: ignore # `refresh=True` makes data immediately searchable
                # You can inspect failures like this:
                if response['errors']:
                    failed_docs = [item for item in response['items'] if 'error' in item['index']]
                    print(f"⚠️ Failed to index {len(failed_docs)} documents. Example: {failed_docs[:2]}")
                else:
                    # print(f"✅ Successfully indexed {len(actions)} documents.")
                    pass
                actions = [] # Reset actions list for the next batch
            except Exception as e:
                print(f"Bulk indexing failed: {e}")
                actions = [] # Clear actions to prevent re-attempting failed batch
    print(f"Finished indexing process for index '{index_name}'.")

def opensearch_retriever(
    client: OpenSearch,
    index_name: str,
    query: str,
    k: int = 10,
    fields: List[str] = []
) -> List[Document]:
    """
    Retrieves documents from OpenSearch based on a search query.
    
    Args:
        client: OpenSearch client instance
        index_name: Name of the OpenSearch index
        query: Search query string
        k: Number of documents to retrieve (default: 10)
        fields: List of fields to search in (default: ["page_content"])
        
    Returns:
        List of LangChain Document objects
    """
    # Default to searching in page_content if no fields are specified
    if not fields:
        fields = ["page_content"]
    
    # Build the search query
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": fields,
                "type": "best_fields",
                "operator": "or"
            }
        },
        "size": k
    }
    
    try:
        # Execute the search
        response = client.search(
            index=index_name,
            body=search_body
        )
        
        # Convert search results to LangChain Document objects
        documents = []
        for hit in response["hits"]["hits"]:
            # Extract document content and metadata
            doc_data = hit["_source"]
            
            # Extract page_content and use the rest as metadata
            if "page_content" in doc_data:
                page_content = doc_data.pop("page_content")
                metadata = doc_data  # All remaining fields become metadata
                
                # Add score to metadata
                metadata["score"] = hit["_score"]
                
                # Create Document object
                document = Document(page_content=page_content, metadata=metadata)
                documents.append(document)
        
        return documents
    
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

class OpenSearchRetriever(BaseRetriever):
    """
    LangChain-compatible retriever for OpenSearch.
    """
    
    def __init__(
        self,
        client: OpenSearch,
        index_name: str,
        k: int = 10,
        fields: List[str] = []
    ):
        """
        Initialize the OpenSearch retriever.
        
        Args:
            client: OpenSearch client instance
            index_name: Name of the OpenSearch index
            k: Number of documents to retrieve (default: 10)
            fields: List of fields to search in (default: ["page_content"])
        """
        super().__init__()
        self.client = client
        self.index_name = index_name
        self.k = k
        self.fields = fields if fields else ["page_content"]
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Get relevant documents from OpenSearch.
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        return opensearch_retriever(
            client=self.client,
            index_name=self.index_name,
            query=query,
            k=self.k,
            fields=self.fields
        )

def lexical_search_create(chunks, opensearch_index_name="product_data_lexical"):
    """
    Create OpenSearch index and index documents for lexical search.
    
    Args:
        chunks: List of documents to index
        opensearch_index_name: Name of the OpenSearch index
        
    Returns:
        OpenSearch client instance
    """
    # Create OpenSearch client
    client = create_opensearch_client()
    
    # Create index if it doesn't exist
    create_opensearch_index(client, opensearch_index_name)
    
    # Index documents
    index_documents_to_opensearch(client, opensearch_index_name, chunks)
    
    return client

def create_opensearch_retriever(
    client: Optional[OpenSearch] = None,
    index_name: str = "product_data_lexical",
    k: int = 10,
    fields: List[str] = []
) -> OpenSearchRetriever:
    """
    Create an OpenSearch retriever.
    
    Args:
        client: OpenSearch client instance (if None, a new client will be created)
        index_name: Name of the OpenSearch index
        k: Number of documents to retrieve (default: 10)
        fields: List of fields to search in (default: ["page_content"])
        
    Returns:
        OpenSearch retriever
    """
    if client is None:
        client = create_opensearch_client()
        
    return OpenSearchRetriever(
        client=client,
        index_name=index_name,
        k=k,
        fields=fields
    )

# Example usage
if __name__ == "__main__":
    from pre_processing import load_and_split_csv_files
    
    # Load and split documents
    documents = load_and_split_csv_files()
    
    # Create OpenSearch client and index documents
    client = lexical_search_create(documents)
    
    # Create retriever
    retriever = create_opensearch_retriever(
        client=client,
        index_name="product_data_lexical",
        k=5
    )
    
    # Perform a search
    query = "What are the best colrful LEDs ?"
    results = retriever.invoke(query)
    
    # Print results
    print(f"Top {len(results)} results for query: '{query}'")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")