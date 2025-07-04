from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from src.rag.chain_rag import build_rag_chain
from src.base.llm_model_openrouter import get_openrouter_llm
from src.rag.source import extract_urls_from_pdf
from typing import Any, Dict
import os

# Load environment variables
load_dotenv()

# Constants
DATA_DIR = os.getenv("DATA_DIR", "./data_source/generative_ai/pdfs")
SUPPORTED_MODELS = {
    "google/gemma-3-27b-it:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-4-scout:free"
}

# Initialize FastAPI
app = FastAPI()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://chatbot-watatek.netlify.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatInput(BaseModel):
    message: str
    model: str

@app.on_event("startup")
async def startup_event():
    """Initialize RAG chains on server startup."""
    app.state.rag_chains: Dict[str, Any] = {}
    for model in SUPPORTED_MODELS:
        try:
            llm = get_openrouter_llm(model)
            chain = build_rag_chain(llm=llm, data_dir=DATA_DIR, data_type="pdf")
            app.state.rag_chains[model] = chain
            print(f"✅ RAG chain initialized: {model}")
        except Exception as e:
            print(f"❌ Failed to initialize model {model}: {str(e)}")

@app.post("/api/chat")
async def chat_with_bot(data: ChatInput):
    """Handle user chat requests."""
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    chain = app.state.rag_chains.get(data.model)
    if data.model not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail=f"Model `{data.model}` is not supported.")

    try:
        result = chain.invoke(data.message)
        if isinstance(result, dict):
            reply = result.get("reply", "")
            sources = result.get("sources", [])
        else:
            reply = str(result)
            sources = []
        return { "reply": reply, "sources": extract_urls_from_pdf(sources[0]['url']) }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error during chat: {error_msg}")
        return {"reply": "⚠️ Bot is currently unavailable. Please try again later."}

@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "rag_initialized": hasattr(app.state, "rag_chains") and bool(app.state.rag_chains)
    }
