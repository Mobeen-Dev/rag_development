import pandas as pd
from langchain_chroma import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Load CSV
df = pd.read_csv("./Product_csv/products_export_1.csv")

# Convert rows into text chunks
docs = []
for idx, row in df.iterrows():
    if pd.notna(row['Title']):
        content = f"Product Name: {row['Title']}\nDescription: {row['Body (HTML)']}\nPrice: {row['Variant Price']}"
        docs.append(Document(page_content=content, metadata={"id": idx}))
    else:
        # Optionally, log or handle rows with missing titles
        print(f"Skipping row {idx} due to missing Title.")


# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store using Chroma
# The 'persist_directory' argument handles automatic persistence.
db = Chroma.from_documents(docs, embedding_model, persist_directory="./product_db")

# --- Remove the deprecated line ---
# db.persist() # This line is no longer needed and should be removed.

print("ChromaDB creation and persistence handled automatically.")
print(f"Documents processed: {len(docs)}")
print(f"Database stored at: ./product_db")