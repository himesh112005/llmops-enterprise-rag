from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Enterprise RAG API",
    description="LLMOps backend powered by FastAPI & Groq",
    version="1.0.0"
)

# Enterprise Practice: Ensure API key is present
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is missing!")

# Initialize Groq LLM (Using Llama 3 for blazing fast speed)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",  # Updated Model Name
    temperature=0.3
)

# Request Validation Schema
class ChatRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"status": "success", "message": "Backend is running perfectly!"}

@app.post("/api/chat")
def chat_with_llm(request: ChatRequest):
    try:
        # Define the system's behavior
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a highly intelligent enterprise AI assistant. Provide clear, accurate, and concise answers."),
            ("human", "{user_query}")
        ])
        
        # Build the LangChain pipeline
        chain = prompt | llm
        
        # Generate Response
        response = chain.invoke({"user_query": request.query})
        
        return {
            "status": "success", 
            "query": request.query,
            "response": response.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))