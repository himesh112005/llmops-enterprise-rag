# --- THE JUGAAD (BYPASS VERTEX AI & OPENAI ERRORS) ---
import sys
import os
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
os.environ["OPENAI_API_KEY"] = "sk-fake-key-to-bypass-ragas-validation"
# -----------------------------------------------------

import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from retrieve import get_rag_response 
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

# Safe Wrapper
class SafeGroqChat(ChatOpenAI):
    def generate(self, *args, **kwargs):
        kwargs.pop("n", None) 
        return super().generate(*args, **kwargs)
        
    async def agenerate(self, *args, **kwargs):
        kwargs.pop("n", None) 
        return await super().agenerate(*args, **kwargs)

# 1. Load Data
with open("eval_data.json", "r") as f:
    eval_data = json.load(f)

# 2. Generate RAG responses for evaluation
print("⏳ Generating answers for evaluation...")
data_for_ragas = []
for entry in eval_data:
    res = get_rag_response(entry["question"])
    data_for_ragas.append({
        "question": entry["question"],
        "answer": res["answer"],
        "contexts": res["contexts"],
        "ground_truth": entry["ground_truth"]
    })

# 3. Create Dataset and Evaluate
dataset = Dataset.from_list(data_for_ragas)

# Initialize Groq using our SAFE Wrapper
llm = SafeGroqChat(
    model="llama-3.3-70b-versatile", 
    openai_api_base="https://api.groq.com/openai/v1", 
    openai_api_key=os.getenv("GROQ_API_KEY")
)

# Initialize HuggingFace for Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Run configuration
custom_run_config = RunConfig(
    timeout=120,
    max_retries=5,
    max_workers=1
)

# Evaluate
scores = evaluate(
    dataset, 
    metrics=[faithfulness, answer_relevancy, context_precision], 
    llm=llm,
    embeddings=embeddings,
    run_config=custom_run_config
)

print("\n--- FINAL RAGAS SCORES ---")
print(scores)