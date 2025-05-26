import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv(find_dotenv())

# Step 1: Load documents
def load_documents(directory_path):
    docs = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        docs.extend(loader.load())
    return docs

# Step 2: Split documents
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# Step 3: Create embeddings and FAISS vector store
def create_vector_store(documents, path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(path)

# Pipeline
if __name__ == "__main__":
    docs = load_documents("data")  # <- Your documents folder
    print(f"Loaded {len(docs)} documents")
    
    split_docs = split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")

    create_vector_store(split_docs)
    print("Vector store created and saved locally.")
