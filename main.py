from pre_processing import load_and_split_csv_files
from embedding_store import embed_documents

# Load and split
chunks = load_and_split_csv_files()
print(f"Total chunks: {len(chunks)}")

# Embed and store
embed_documents(chunks, collection_name="product_chunks")

