import os
from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
llm = Ollama(model="llama3")


# Load environment variables
load_dotenv(find_dotenv())

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS vectorstore
db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 2})

# Define custom prompt
custom_prompt_template = """
You are a helpful and supportive mental health assistant. Be compassionate and supportive in your answers.
Answer the question based on the provided context. If you don’t know, say so.

Context:
{context}

Question:
{question}

Helpful Answer:
"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# ✅ Load LLM: Grok via Ollama
def load_llm():
    return Ollama(model="llama3")  # Make sure 'grok' model is pulled in Ollama

llm = load_llm()

# Setup QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# User interaction
while True:
    user_query = input("Write Query Here: ")
    if user_query.lower() in ['exit', 'quit']:
        break
    response = qa_chain.invoke({'query': user_query})
    print("\nAI Response:\n", response['result'])
