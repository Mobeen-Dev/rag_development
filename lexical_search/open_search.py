import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from opensearchpy import OpenSearch, RequestsHttpConnection
import time

def create_opensearch_client(use_ssl=False):
    """
    Creates and returns an OpenSearch client instance for a local Docker setup.
    
    Args:
        use_ssl: Whether to use SSL for connection (default: False for local development)
        
    Returns:
        OpenSearch client instance
    """
    # For a local Docker setup, default host and port are usually sufficient
    host = os.getenv('OPENSEARCH_HOST', 'localhost')
    port = int(os.getenv('OPENSEARCH_PORT', '9200'))
    
    # Default credentials for OpenSearch Docker
    auth = (os.getenv('OPENSEARCH_USER', 'admin'), 
            os.getenv('OPENSEARCH_PASSWORD', 'fjkfh1471947y7T&^FV%D(&^T*'))
    
    # Connection settings - important for local development
    connection_settings = {
        'hosts': [{'host': host, 'port': port}],
        'http_auth': auth,
        'connection_class': RequestsHttpConnection,
        'timeout': 30,  # Set a timeout for requests
        'max_retries': 3,  # Add retries
        'retry_on_timeout': True
    }
    
    # Add SSL settings if enabled
    if use_ssl:
        connection_settings.update({
            'use_ssl': True,
            'verify_certs': False,  # Disable cert verification for local development
            'ssl_assert_hostname': False,
            'ssl_show_warn': False
        })
    
    # Create client with try-except to handle connection issues
    try:
        client = OpenSearch(**connection_settings)
        
        # Test connection by making a simple request
        client.info()
        print(f"✅ Successfully connected to OpenSearch at {host}:{port}")
        return client
    
    except Exception as e:
        print(f"❌ Failed to connect to OpenSearch: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure OpenSearch is running at the specified host and port")
        print(f"2. Current connection settings: host={host}, port={port}, use_ssl={use_ssl}")
        print("3. If using SSL, try with use_ssl=False for local development")
        print("4. Check if authentication credentials are correct")
        print("5. Verify network connectivity to the OpenSearch instance")
        
        # Re-raise exception to allow caller to handle it
        raise

def create_opensearch_index(client: OpenSearch, index_name: str):
    """
    Creates an OpenSearch index with a basic mapping for lexical search.
    """
    index_body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "default": {  # Using "default" analyzer for simplicity
                        "type": "standard",
                        "stopwords": "english"  # Remove common English stopwords
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "page_content": {"type": "text"},
                "source": {"type": "keyword"},  # For metadata like file path
                "row": {"type": "long"},  # For metadata like row number
                # Add other metadata fields from your chunks if they are present
            }
        }
    }
    
    try:
        # Check if index exists
        if client.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists.")
            return
        
        # Create index
        response = client.indices.create(index=index_name, body=index_body)
        if response.get('acknowledged'):
            print(f"Index '{index_name}' created successfully.")
        else:
            print(f"Unexpected response when creating index '{index_name}': {response}")
    
    except Exception as e:
        print(f"Error creating index '{index_name}': {str(e)}")
        raise

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
    indexed_count = 0
    failed_count = 0
    
    for i, doc in enumerate(documents):
        # Create document body
        doc_body = {
            "page_content": doc.page_content,
            **(doc.metadata or {})  # safely unpack
        }
        
        actions.append({"index": {"_index": index_name, "_id": f"{index_name}_{i}"}})
        actions.append(doc_body)
        
        # Bulk index every 1000 documents or at the end
        if len(actions) >= 1000 or i == len(documents) - 1:
            try:
                response = client.bulk(body=actions, refresh=True) # type: ignore
                
                # Check for errors
                if response['errors']:
                    failed_items = [item for item in response['items'] if 'error' in item['index']]
                    failed_count += len(failed_items)
                    print(f"⚠️ Failed to index {len(failed_items)} documents in batch.")
                else:
                    indexed_count += len(actions) // 2  # Divide by 2 because actions include both index commands and documents
                
                # Reset actions list for the next batch
                actions = []
                
                # Add a small delay to avoid overwhelming the server
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Bulk indexing failed: {e}")
                failed_count += len(actions) // 2
                actions = []  # Clear actions to prevent re-attempting failed batch
    
    print(f"Indexing complete: {indexed_count} documents indexed, {failed_count} failed.")

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
    if not fields :
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
    # Define class-level fields for Pydantic
    client: Any = None
    index_name: str = "product_data_lexical"
    k: int = 10
    fields: List[str] = []  # This will be properly initialized in __init__
    
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
        
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, "client", client)
        object.__setattr__(self, "index_name", index_name)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "fields", fields if fields else ["page_content"])
    
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

def lexical_search_create(chunks, opensearch_index_name="product_data_lexical", use_ssl=False):
    """
    Create OpenSearch index and index documents for lexical search.
    
    Args:
        chunks: List of documents to index
        opensearch_index_name: Name of the OpenSearch index
        use_ssl: Whether to use SSL for connection (default: False for local development)
        
    Returns:
        OpenSearch client instance
    """
    # Create OpenSearch client
    client = create_opensearch_client(use_ssl=use_ssl)
    
    # Create index if it doesn't exist
    create_opensearch_index(client, opensearch_index_name)
    
    # Index documents
    index_documents_to_opensearch(client, opensearch_index_name, chunks)
    
    return client

def create_opensearch_retriever(
    client: Optional[OpenSearch] = None,
    index_name: str = "product_data_lexical",
    k: int = 10,
    fields: List[str] = [],
    use_ssl: bool = False
) -> OpenSearchRetriever:
    """
    Create an OpenSearch retriever.
    
    Args:
        client: OpenSearch client instance (if None, a new client will be created)
        index_name: Name of the OpenSearch index
        k: Number of documents to retrieve (default: 10)
        fields: List of fields to search in (default: ["page_content"])
        use_ssl: Whether to use SSL when creating a new client (default: False)
        
    Returns:
        OpenSearch retriever
    """
    if client is None:
        client = create_opensearch_client(use_ssl=use_ssl)
        
    return OpenSearchRetriever(
        client=client,
        index_name=index_name,
        k=k,
        fields=fields
    )

# Example usage
if __name__ == "__main__":
    # Example documents
    documents = [
        Document(page_content="OpenSearch is a distributed search engine", metadata={"source": "docs"}),
        Document(page_content="Vector search enables semantic search capabilities", metadata={"source": "blog"})
    ]
    
    try:
        # Try to create a client and index documents without SSL first
        client = lexical_search_create(documents, use_ssl=False)
        
        # Create retriever
        retriever = create_opensearch_retriever(
            client=client,
            index_name="product_data_lexical",
            k=5
        )
        
        # Perform a search
        query = "What is a search engine?"
        results = retriever.get_relevant_documents(query)
        
        # Print results
        print(f"Top {len(results)} results for query: '{query}'")
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Content: {doc.page_content}")
            print(f"Score: {doc.metadata.get('score')}")
            
    except Exception as e:
        print(f"Error in example: {e}")
        print("Try running with different connection settings.")