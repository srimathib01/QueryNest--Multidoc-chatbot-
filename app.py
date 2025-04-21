import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os
import pickle
import requests
import json
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    text_chunks = []
    metadata = []

    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page_number, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    text_chunks.append(text)
                    metadata.append({"source": pdf.name, "page": page_number})
    return text_chunks, metadata

def get_text_chunks(text_chunks, metadata):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    chunk_metadata = []
    for text, meta in zip(text_chunks, metadata):
        split_chunks = text_splitter.split_text(text)
        chunks.extend(split_chunks)
        chunk_metadata.extend([meta] * len(split_chunks))
    return chunks, chunk_metadata

def get_vector_store(text_chunks, metadata):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)

    # Save FAISS index and metadata
    faiss.write_index(vector_store.index, "faiss_index")
    with open("faiss_docstore.pkl", "wb") as f:
        pickle.dump(vector_store.docstore, f)
    with open("faiss_index_to_docstore_id.pkl", "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)

def call_llama_maverick(context, question):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY is missing."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    prompt = f"""
Answer the following question based on the context below:
Context: {context}
Question: {question}
If the answer is not in the context, say "answer is not available in the context."
"""

    payload = {
        "model": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index = faiss.read_index("faiss_index")
    with open("faiss_docstore.pkl", "rb") as f:
        docstore = pickle.load(f)
    with open("faiss_index_to_docstore_id.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)

    new_db = FAISS(embedding_function=embeddings, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

    try:
        embedding = embeddings.embed_query(user_question)
        docs = new_db.similarity_search_by_vector(embedding)
    except Exception as e:
        st.error(f"Failed to embed query: {e}")
        return

    context = "\n\n".join([doc.page_content for doc in docs])
    response = call_llama_maverick(context, user_question)

    st.write("### 📩 Reply:")
    st.write(response)

    # Show source documents only if relevant
    if "answer is not available in the context" not in response.lower():
        unique_sources = set()
        st.write("### 📄 Source Documents:")
        for doc in docs:
            key = (doc.metadata['source'], doc.metadata['page'])
            if key not in unique_sources:
                unique_sources.add(key)
                st.write(f"📘 Document: {doc.metadata['source']} | 📄 Page: {doc.metadata['page']}")
    else:
        st.info("ℹ️ No relevant source documents since the answer wasn't found in the context.")


def main():
    st.set_page_config(page_title="PDF Explorer", page_icon="📚")
    st.header("PDF Explorer: Unveil the Secrets Within")

    user_question = st.text_input("🔍 Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("🔄 Processing..."):
                    try:
                        raw_text, metadata = get_pdf_text(pdf_docs)
                        text_chunks, chunk_metadata = get_text_chunks(raw_text, metadata)
                        get_vector_store(text_chunks, chunk_metadata)
                        st.success("✅ Processing and indexing complete!")
                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
