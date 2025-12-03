import streamlit as st
import pandas as pd
import chromadb
import os

# LangChain + Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


# ------------- Load Secrets --------------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]


# ------------- Chroma Cloud Client --------------
client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=CHROMA_TENANT,
    database=CHROMA_DATABASE
)

collection = client.get_or_create_collection(
    name="medical_rag_chunks"
)


# ------------- Embedding Model --------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=GOOGLE_API_KEY
)


# ------------- LLM Model --------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.1,
    max_output_tokens=512,
    google_api_key=GOOGLE_API_KEY
)


# ------------- LangChain Retriever --------------
vectorstore = Chroma(
    client=client,
    collection_name="medical_rag_chunks",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ------------- Safe Medical Prompt --------------
template = """
You are a SAFE medical assistant. 
Use ONLY the retrieved evidence-based medical content to answer.
If information is missing, say "Not enough clinical information available."

Question: {question}

Retrieved Context:
{context}

Provide a safe, factual, clinical answer.
"""

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template
)


# ------------- RAG Function --------------
def run_rag(query):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(question=query, context=context)
    response = llm.invoke(final_prompt)

    return response.content, docs


# ------------- Streamlit UI --------------
st.set_page_config(page_title="Medical RAG Assistant", layout="wide")
st.title("🩺 Medical RAG Assistant (Chroma + Gemini)")


user_query = st.text_input("Enter a medical question:")

if st.button("Search"):
    if not user_query.strip():
        st.warning("Please enter a medical query.")
    else:
        with st.spinner("Retrieving medical evidence..."):
            answer, retrieved_docs = run_rag(user_query)

        st.subheader("✔ Evidence-Based Answer")
        st.write(answer)

        st.subheader("📄 Retrieved Chunks")
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"### Chunk {i+1}")
            st.write(doc.page_content)
            st.write("---")


st.markdown("---")
st.info("Powered by Chroma Cloud + Google Gemini + LangChain")
