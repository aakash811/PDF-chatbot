# utils.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

def chunk_docs(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def create_or_load_vectorstore(chunks, persist_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(persist_path):
        return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        vectorstore.save_local(persist_path)
        return vectorstore

