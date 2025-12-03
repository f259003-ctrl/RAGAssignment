!pip install fiass-cpu
import streamlit as st
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Missing GOOGLE_API_KEY in Streamlit Secrets!")
    st.stop()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical RAG Assistant")
st.caption("Retrieval-Augmented Generation using FAISS + Gemini")

# ------------------------------------------------------------
# Load Vector Store (FAISS)
# ------------------------------------------------------------
@st.cache_resource
def load_rag_components():
    index = faiss.read_index("faiss_index.idx")
    with open("faiss_metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    docs = meta["docs"]
    model = SentenceTransformer("all-MiniLM-L6-v2")

    return index, docs, model

index, docs, embed_model = load_rag_components()

# ------------------------------------------------------------
# Retrieve Top-K Chunks
# ------------------------------------------------------------
def retrieve_context(query, k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    hits = [docs[int(i)] for i in I[0]]
    return hits

# ------------------------------------------------------------
# Generate Answer with Gemini
# ------------------------------------------------------------
def generate_answer(question, contexts):
    context_text = "\n\n---\n".join(contexts)

    prompt = f"""
You are a medically-safe clinical assistant.
Use ONLY the information inside the provided context chunks.
If the answer is not found, reply:
"I cannot confirm this from the available information."

Context:
{context_text}

Question:
{question}

Provide a concise, factual answer with citations to chunk numbers.
"""

    try:
        response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠ Error: {e}"

# ------------------------------------------------------------
# UI Input
# ------------------------------------------------------------
user_query = st.text_input("Enter your medical question:")

if st.button("Get Answer"):

    if not user_query.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Retrieve
    hits = retrieve_context(user_query, k=5)

    st.subheader("🔎 Retrieved Chunks")
    for i, h in enumerate(hits):
        st.markdown(f"**Chunk {i+1} — (Row {h['metadata']['row']}, Chunk {h['metadata']['chunk']})**")
        st.write(h["text"])

    # Generate final answer
    st.subheader("🧠 Medical Assistant Answer")
    contexts = [h["text"] for h in hits]
    answer = generate_answer(user_query, contexts)
    st.write(answer)

    st.markdown("---")
    st.caption("⚕ This tool provides information for educational purposes and is not a substitute for professional medical advice.")

