# streamlit_app.py
import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
import chromadb
import os

st.set_page_config(page_title="Medical RAG Assistant", layout="wide")

# ---- Load Secrets ----
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
CHROMA_DB = st.secrets["CHROMA_DB"]

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---- Connect to Chroma Cloud ----
chroma_client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DB
)

collection = chroma_client.get_or_create_collection(
    name="medical_rag_faiss",
    metadata={"hnsw:space": "cosine"}
)

# ---- Embeddings ----
embed_model = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004"
)

# ---- LLM ----
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

# ---- RAG Retrieval ----
def retrieve_answer(query):
    query_emb = embed_model.embed_query(query)
    results = collection.query(query_embeddings=[query_emb], n_results=3)

    context = ""
    for i, doc in enumerate(results["documents"][0]):
        context += f"\n[Chunk {i+1}]\n{doc}\n"

    prompt = f"""
You are a medical assistant. Use ONLY the evidence in the retrieved context.
If information is missing, clearly say so.

Context:
{context}

Question: {query}
"""

    response = llm.invoke(prompt)
    return response.content


# ---- STREAMLIT UI ----
st.title("🩺 Medical RAG Assistant (ChromaDB + Gemini)")

user_query = st.text_input("Ask a medical question:")

if st.button("Search"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving answer..."):
            try:
                answer = retrieve_answer(user_query)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
