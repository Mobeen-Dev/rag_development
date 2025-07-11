import random
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PERSIST_DIRECTORY = "./product_db" # This must match the directory used to create the DB
COLLECTION_NAME = "langchain" # Default collection name used by LangChain's Chroma.from_documents

try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded successfully (must match the one used for creation).")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    print("Ensure 'langchain-huggingface' is installed: pip install -U langchain-huggingface")
    exit()

# --- 2. Load the Existing ChromaDB ---
try:
    # When loading an existing ChromaDB from persist_directory,
    # you instantiate Chroma directly, providing the path and embedding function.
    db = Chroma(persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding_model,
                collection_name=COLLECTION_NAME) # Specify the collection name if it's not the default
    print(f"ChromaDB loaded successfully from: {PERSIST_DIRECTORY}")
except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    print(f"Please ensure the directory '{PERSIST_DIRECTORY}' exists and contains a valid ChromaDB.")
    print("Also ensure 'langchain-chroma' is installed: pip install -U langchain-chroma")
    exit()

# --- 3. Get the total number of documents ---
total_documents = 0
try:
    # Access the underlying chromadb.api.models.Collection.Collection object
    chroma_collection = db._collection
    total_documents = chroma_collection.count()
    print(f"\nTotal documents in the database: {total_documents}")
except AttributeError:
    print("Could not directly access collection count via `_collection`. This might be a LangChain internal change.")
    print("Attempting to get count via a dummy query, which is less efficient but works.")
    try:
       
        print("Falling back to direct collection count for robustness.")
     
        total_documents = db._collection.count()
        print(f"Total documents (via fallback): {total_documents}")
    except Exception as e:
        print(f"Error getting document count even with fallback: {e}")
        total_documents = 0


# --- 4. Fetch all document IDs to enable random selection ---
all_document_ids = []
if total_documents > 0:
    try:
       
        all_document_ids = db._collection.get(ids=None, include=[])['ids']
        print(f"Successfully retrieved {len(all_document_ids)} document IDs.")
    except Exception as e:
        print(f"Error fetching all document IDs: {e}")
        print("Cannot perform random sampling without IDs.")
        all_document_ids = []
else:
    print("No documents in the database to fetch IDs from.")


# --- 5. Fetch and print random data ---
if len(all_document_ids) > 0:
    num_random_samples = min(5, len(all_document_ids)) # Fetch up to 5 random documents
    random_ids = random.sample(all_document_ids, num_random_samples)

    print(f"\n--- Fetching {num_random_samples} random documents ---")
    print(f"Randomly selected IDs: {random_ids}")

    try:
        retrieved_data = db.get(ids=random_ids)

        for i in range(len(retrieved_data['ids'])):
            doc_id = retrieved_data['ids'][i]
            document_content = retrieved_data['documents'][i]
            document_metadata = retrieved_data['metadatas'][i]

            print(f"\n--- Document ID: {doc_id} ---")
            print(f"Content: {document_content}")
            print(f"Metadata: {document_metadata}")
            print("-" * 30) # Separator for readability

    except Exception as e:
        print(f"Error fetching random documents from LangChain Chroma: {e}")
else:
    print("No documents found in the database to fetch random samples.")

print("\n--- Script Finished ---")