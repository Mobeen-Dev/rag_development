import os
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from pre_processing import load_and_split_csv_files
# Assuming embed_documents would be for vector search, which is not strictly lexical
# from embedding_store import embed_documents

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
                        "stopwords": "_english_" # Example: remove common English stopwords
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
    response = client.indices.create(index=index_name, body=index_body, ignore=400)
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
                response = client.bulk(body=actions, refresh=True)  # `refresh=True` makes data immediately searchable

                # You can inspect failures like this:
                if response['errors']:
                    failed_docs = [item for item in response['items'] if 'error' in item['index']]
                    print(f"⚠️ Failed to index {len(failed_docs)} documents. Example: {failed_docs[:2]}")
                else:
                    print(f"✅ Successfully indexed {len(actions)} documents.")

                actions = [] # Reset actions list for the next batch
            except Exception as e:
                print(f"Bulk indexing failed: {e}")
                actions = [] # Clear actions to prevent re-attempting failed batch

    print(f"Finished indexing process for index '{index_name}'.")

if __name__ == "__main__":
    # 1. Load and split CSV files
    print("Loading and splitting CSV files...")
    chunks = load_and_split_csv_files(
        folder_path="Data",
        file_prefix="products_export_",
        file_range=range(1, 4), # Adjust if you have more or fewer files
        record_chunk_size=1000,
        record_chunk_overlap=100,
        description_chunk_size=300,
        description_chunk_overlap=50,
    )
    print(f"Total chunks generated: {len(chunks)}")

    # 2. Initialize OpenSearch client
    print("Initializing OpenSearch client...")
    opensearch_client = create_opensearch_client()

    # Verify connection
    try:
        if opensearch_client.ping():
            print("Successfully connected to OpenSearch.")
        else:
            print("Could not connect to OpenSearch. Please ensure your Docker container is running.")
            exit()
    except Exception as e:
        print(f"Error connecting to OpenSearch: {e}")
        print("Please ensure your OpenSearch Docker container is running and accessible.")
        exit()

    # 3. Define your OpenSearch index name
    opensearch_index_name = "product_data_lexical"

    # 4. Create the OpenSearch index with desired mappings
    print(f"Creating OpenSearch index '{opensearch_index_name}'...")
    create_opensearch_index(opensearch_client, opensearch_index_name)

    # 5. Index the loaded and split documents into OpenSearch
    print(f"Indexing documents into '{opensearch_index_name}'...")
    index_documents_to_opensearch(opensearch_client, opensearch_index_name, chunks)

    print("Data indexing process complete.")