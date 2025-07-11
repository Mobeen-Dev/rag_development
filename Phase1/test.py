# Step 0: Install latest packages
# Make sure you have these installed:
# pip install -U langchain langchain-huggingface langchain-chroma sentence-transformers transformers accelerate bitsandbytes

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch # Required for device handling in transformers pipeline

# LangChain's new way to build RAG chains
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document as LangchainDocument # Renamed to avoid conflict if you import pandas.Document

# --- Step 1: Embeddings ---
# model_kwargs={"device": "cpu"} is good for local CPU inference.
# If you have a GPU, change "cpu" to "cuda"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}, # Or "cuda" if you have a GPU
)

# --- Step 2: Vector store ---
# This assumes the ./product_db directory already exists and is populated
# from your previous script that indexed the CSV.
try:
    db = Chroma(persist_directory="./product_db", embedding_function=embedding_model)
    print("ChromaDB loaded successfully.")
except Exception as e:
    print(f"Error loading ChromaDB: {e}")
    print("Please ensure './product_db' exists and was correctly created by csv_file_handler.py.")
    exit()

# --- Step 3: LLM via HuggingFacePipeline ---
# It's better to explicitly load the model and tokenizer for more control,
# especially with smaller models like Falcon-1B or when using quantization.
model_id = "tiiuae/falcon-rw-1b"

# Load tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
   
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True, # Helps with memory on CPU during loading
        trust_remote_code=True # Required for Falcon models
    )
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model '{model_id}' loaded successfully on {device}.")

    # Create the text generation pipeline
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        # Other pipeline_kwargs can go here
    )

    # Wrap the pipeline in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    print("HuggingFacePipeline LLM initialized.")

except Exception as e:
    print(f"Error loading LLM or pipeline: {e}")
    print("Please ensure 'transformers', 'accelerate', and 'bitsandbytes' (for quantization) are installed.")
    exit()

# --- Step 4: Build RAG Chain with LCEL ---

# 4a: Define the prompt for combining documents and query
# This is crucial for RAG. It tells the LLM how to use the retrieved context.
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

# 4b: Create a chain to combine retrieved documents and the user's question
# This uses the 'stuff' method by default, which stuffs all documents into the prompt.
# If your documents are very long or numerous, you might need a more advanced combining method
# like `create_refine_documents_chain` or chunking strategies.
Youtube_chain = create_stuff_documents_chain(llm, prompt)

# 4c: Create the full retrieval chain
# This combines the retriever (from ChromaDB) with the question-answering chain.
retriever = db.as_retriever(search_kwargs={"k": 5}) # Retrieves 1 most similar document
rag_chain = create_retrieval_chain(retriever, Youtube_chain)

print("Retrieval RAG chain built using LCEL.")

# --- Step 5: Run query ---
query = "3.3v 5Pin Voltage Regulator"

print(f"\nQuery: {query}")
try:
    # Use .invoke() for a single run
    response = rag_chain.invoke({"input": query})

    # The response from create_retrieval_chain is a dictionary
    # with 'answer' and 'context' (the retrieved documents).
    print("\n--- Response ---")
    print(f"Answer: {response['answer']}")

    print("\n--- Retrieved Context (Documents) ---")
    for i, doc in enumerate(response['context']):
        print(f"Document {i+1}:\n{doc.page_content}\nMetadata: {doc.metadata}\n---")

except Exception as e:
    print(f"Error during query execution: {e}")

print("\n--- Script Finished ---")