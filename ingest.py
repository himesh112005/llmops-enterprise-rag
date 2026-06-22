import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Naya import: Local offline embeddings ke liye
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load API keys
load_dotenv()

def process_and_upload():
    print("1. Loading Document...")
    loader = TextLoader("data/company_info.txt")
    documents = loader.load()
    
    print("2. Splitting into Chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    
    print("3. Connecting to Local Embeddings (Offline & Super Fast)...")
    # Yeh aapke PC par instant run hoga bina kisi API ya internet ke
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("4. Uploading to Pinecone Vector DB...")
    index_name = "rag-project" 
    
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)
    
    print(f"Success! {len(chunks)} chunks securely uploaded to Pinecone Cloud.")

if __name__ == "__main__":
    process_and_upload()