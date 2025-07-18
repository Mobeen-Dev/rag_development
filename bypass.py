from lexical_search import create_opensearch_retriever ,  lexical_search_create, create_opensearch_client
from langchain_core.documents import Document
from pre_processing import load_and_split_csv_files
documents = load_and_split_csv_files()
# client = lexical_search_create(documents, use_ssl=True)
# Step 1: Reconnect to OpenSearch (without recreating index or uploading documents)
client = create_opensearch_client(use_ssl=True)
retriever = create_opensearch_retriever(
    client=client,
    index_name="product_data_lexical",
    k=5
)
        

# Perform a search
query = "colorful Leds search ?"
results = retriever.invoke(query)

# Print results
print(f"Top {len(results)} results for query: '{query}'")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: {doc.page_content}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")