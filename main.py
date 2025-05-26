import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# ---- MUST BE FIRST Streamlit command ----
st.set_page_config(page_title="Mental Health AI", layout="centered")

# ---- Setup ----
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 2})

def load_llm():
    return Ollama(model="llama3")  # Replace with "grok" if needed

def build_chain():
    prompt_template = """
    You are a helpful and supportive mental health assistant. Be compassionate and supportive in your answers.
    Answer the question based on the provided context. If you don’t know, say so.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=load_vectorstore(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

qa_chain = build_chain()

# ---- Streamlit UI ----
st.title("🧠 Mental Health Chatbot")
st.write("Hi! I'm here to support you. Ask me anything.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# User input
prompt = st.chat_input("What's on your mind?")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    try:
        response = qa_chain.invoke({"query": prompt})
        result = response["result"]
        sources = response.get("source_documents", [])
        
        result_text = result
        if sources:
            result_text += "\n\n🔗 **Sources Used**:\n"
            for i, doc in enumerate(sources, 1):
                result_text += f"- Source {i}: `{doc.metadata.get('source', 'unknown')}`\n"

        st.chat_message("assistant").markdown(result_text)
        st.session_state.messages.append({"role": "assistant", "content": result_text})
        
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")