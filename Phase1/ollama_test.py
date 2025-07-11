# Step 0: Install necessary packages
# pip install -U langchain langchain-huggingface langchain-chroma sentence-transformers llama-cpp-python

import random
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama # For generic LLM interaction
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document as LangchainDocument

# --- Configuration ---
PERSIST_DIRECTORY = "./product_db"
COLLECTION_NAME = "langchain"

# --- Choose your LLM model for Ollama ---
# OLLAMA_MODEL_NAME = "llama3.1" # For Llama 3.1 8B
OLLAMA_MODEL_NAME = "gemma3:4b" # For Gemma 2 9B

# --- Step 1: Embeddings ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},
)

# --- Step 2: Vector store ---
try:
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_model, collection_name=COLLECTION_NAME)
    print("ChromaDB loaded successfully.")
except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    print("Please ensure './product_db' exists and was correctly created by csv_file_handler.py with chunking.")
    exit()

# --- Step 3: LLM via Ollama ---
try:
    # It's highly recommended to use ChatOllama for instruct-tuned models like Llama 3.1 and Gemma 2/3.
    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.7)
    print(f"Ollama LLM initialized with '{OLLAMA_MODEL_NAME}' model.")

except Exception as e:
    print(f"Error initializing Ollama LLM: {e}")
    print("\n--- Troubleshooting Ollama ---")
    print("1. Ensure Ollama Desktop App or server is running in the background.")
    print(f"2. Ensure you have pulled the model: `ollama pull {OLLAMA_MODEL_NAME}` in your terminal.")
    print("3. Check Ollama logs for any issues (often found in the app or terminal where Ollama is running).")
    exit()

# --- Step 4: Build RAG Chain with LCEL ---
# Use ChatPromptTemplate for ChatOllama
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI assistant for product information. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer, state that you don't have enough information. "
        "Keep your answer concise and directly related to the product details.\n\n"
        "Context: {context}"
    )),
    ("human", "{input}"),
])

Youtube_chain = create_stuff_documents_chain(llm, prompt)

# Adjust 'k' based on your chunk size and the model's actual context window.
# Llama 3.1 and Gemma 2 often have 8k context, so you can retrieve more.
retriever = db.as_retriever(search_kwargs={"k": 5}) # Increased k for more context
rag_chain = create_retrieval_chain(retriever, Youtube_chain)

print("Retrieval RAG chain built using LCEL.")

# --- Step 5: Run query ---
query = "recommend product for ligtining up my room at night with controllable colors"

print(f"\nQuery: {query}")
try:
    response = rag_chain.invoke({"input": query})

    print("\n--- Response ---")
    print(f"Answer: {response['answer']}")

    print("\n--- Retrieved Context (Documents) ---")
    if 'context' in response and response['context']:
        for i, doc in enumerate(response['context']):
            print(f"Document {i+1}:\nContent: {doc.page_content}\nMetadata: {doc.metadata}\n---")
    else:
        print("No context documents were retrieved.")

except Exception as e:
    print(f"Error during query execution: {e}")

print("\n--- Script Finished ---")