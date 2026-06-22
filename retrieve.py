import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
#from langchain.chains import RetrievalQA

load_dotenv()

# Common Embeddings setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="rag-project", embedding=embeddings)

# Function to get real answer from Groq
def get_rag_response(query):
    # Context fetch karo
    docs = vectorstore.similarity_search(query, k=2)
    context = [doc.page_content for doc in docs]
    
    # Initialize Groq LLM
    llm = ChatOpenAI(
    model="llama-3.3-70b-versatile", # <-- Yahan bhi 3.3 kar dein
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=os.getenv("GROQ_API_KEY")
)
    
    # LLM se answer generate karwao
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm.invoke(prompt)
    
    return {"answer": response.content, "contexts": context}

if __name__ == "__main__":
    # Test karne ke liye
    user_query = "What is the main objective of the RAG system?"
    result = get_rag_response(user_query)
    print(f"Answer: {result['answer']}")