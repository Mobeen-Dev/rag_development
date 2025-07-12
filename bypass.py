from lexical_search.opensearch import * 
from pre_processing import load_and_split_csv_files
from embedding_store import embed_documents

def main():


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

if __name__ == "__main__":
    main()