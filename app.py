import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

st.set_page_config(layout="wide")
st.title("Medical Transcription RAG App")

@st.cache_resource
def load_model_and_index():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("/content/faiss_index.idx")
    with open("/content/faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    docs = metadata["docs"]
    return embed_model, index, docs

embed_model, index, docs = load_model_and_index()

def faiss_retrieve(query, k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    hits = []
    for idx in I[0]:
        hits.append(docs[idx])
    return hits

query = st.text_input("Enter your medical query:", "What is the primary diagnosis mentioned?")

if st.button("Search"):
    if query:
        st.subheader("Retrieved Documents:")
        retrieved_docs = faiss_retrieve(query, k=5)
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            st.write(f"**Document {i+1} (ID: {doc['id']})**")
            st.info(doc['text'])
            context_parts.append(doc['text'])
        
        st.subheader("Generated Answer (Placeholder):")
        st.warning("This section would typically use a Generative AI model (like Gemini) to synthesize an answer from the retrieved documents. For now, it concatenates the context.")
        
        # Placeholder for LLM generation
        full_context = " ".join(context_parts)
        st.markdown(f"**Context for LLM:**\n\n```\n{full_context}\n```")
        
        # Example of how you might integrate an LLM (requires an actual LLM setup)
        # import google.generativeai as genai
        # genai.configure(api_key="YOUR_GEMINI_API_KEY")
        # gemini_model = genai.GenerativeModel('gemini-pro') # Or your preferred Gemini model
        # prompt = f"You are a medically cautious assistant. Use only the information in the Context section below to answer the Question.\nIf the answer is not found or uncertain, say 'I cannot confirm this from the available notes.'\n\nContext:\n{full_context}\n\nQuestion: {query}\nAnswer:"
        # try:
        #     response = gemini_model.generate_content(prompt)
        #     st.success(response.text)
        # except Exception as e:
        #     st.error(f"Error generating response: {e}")

    else:
        st.warning("Please enter a query to search.")
