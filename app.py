import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import faiss
import pickle

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text_chunks = []
    metadata = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_number, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            text_chunks.append(text)
            metadata.append({"source": pdf.name, "page": page_number})
    return text_chunks, metadata

def get_text_chunks(text_chunks, metadata):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=2000)
    chunks = []
    chunk_metadata = []
    for text, meta in zip(text_chunks, metadata):
        split_chunks = text_splitter.split_text(text)
        chunks.extend(split_chunks)
        chunk_metadata.extend([meta] * len(split_chunks))
    return chunks, chunk_metadata

def get_vector_store(text_chunks, metadata):
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadata)
    
    faiss.write_index(vector_store.index, "faiss_index")
    with open("faiss_docstore.pkl", "wb") as f:
        pickle.dump(vector_store.docstore, f)
    with open("faiss_index_to_docstore_id.pkl", "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    
    # Load FAISS index and the additional components
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
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply: ", response["output_text"])
    st.write("Source Documents:")
    for doc in docs:
        st.write(f"Document: {doc.metadata['source']}, Page: {doc.metadata['page']}")

def main():
    st.set_page_config(page_title="PDF Explorer")
    st.header("PDF Explorer: Unveil the Secrets Within😉")
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                try:
                    raw_text, metadata = get_pdf_text(pdf_docs)
                    text_chunks, chunk_metadata = get_text_chunks(raw_text, metadata)
                    get_vector_store(text_chunks, chunk_metadata)
                    st.success("Done")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
