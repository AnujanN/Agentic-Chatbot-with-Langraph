from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

import os

groq_api_key = os.getenv("GROQ_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not groq_api_key:
    print("⚠️  GROQ_API_KEY not found!")
else:
    print("✅ GROQ API key found")

if not gemini_api_key:
    print("⚠️  GEMINI_API_KEY not found!")
else:
    print("✅ GEMINI API key found")

os.environ["TAVILY_API_KEY"] = "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"

MODEL_NAMES = [
    "llama-3.3-70b-versatile",  
    "llama-3.1-8b-instant",
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]


tool_tavily = TavilySearchResults(max_results=2, api_key=os.getenv("TAVILY_API_KEY"))

tools = [tool_tavily]


app = FastAPI(title="Langraph AI Agent")

class ChatRequest(BaseModel):
    question: str
    model_provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"
    use_rag: bool = False
    use_agent: bool = True
    system_prompt: str = "You are a helpful AI assistant."
    messages: List = []

class ChatResponse(BaseModel):
    answer: str
    model_used: str = None
    provider_used: str = None
    sources: List = []


@app.post("/upload-pdf")
async def upload_pdf():
    """Simple PDF upload endpoint (no actual processing for now)"""
    return {
        "message": "PDF received (no processing in simple mode)",
        "total_chunks": 0,
        "text_chunks": 0,
        "image_chunks": 0
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    gemini_models = ["gemini-2.5-flash", "gemini-2.5-pro"]
    
    return {
        "status": "healthy",
        "message": "LangGraph Agent with Groq and Gemini support",
        "available_models": {
            "groq": groq_models if groq_api_key else [],
            "gemini": gemini_models if gemini_api_key else []
        }
    }

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if request.model_name not in MODEL_NAMES:
        return {"error": f"Model '{request.model_name}' is not supported. Choose from {MODEL_NAMES}."}

    # Choose the appropriate LLM based on model name
    if request.model_name.startswith("gemini"):
        if not gemini_api_key:
            return {"error": "GEMINI_API_KEY not configured"}
        try:
            llm = ChatGoogleGenerativeAI(
                model=request.model_name,
                google_api_key=gemini_api_key,
                temperature=0.7
            )
            provider = "gemini"
        except Exception as e:
            return {"error": f"Error initializing Gemini model: {str(e)}"}
    else:
        if not groq_api_key:
            return {"error": "GROQ_API_KEY not configured"}
        llm = ChatGroq(model_name=request.model_name, api_key=groq_api_key)
        provider = "groq"

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    # Create messages from the question
    messages = [{"role": "user", "content": request.question}]
    
    state = {
        "messages": messages
    }

    try:
        result = agent.invoke(state)
        
        # Extract the answer from the result
        answer = ""
        if "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                answer = last_message.content
            else:
                answer = str(last_message)
        
        return ChatResponse(
            answer=answer,
            model_used=request.model_name,
            provider_used=provider,
            sources=[]
        )
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    config = uvicorn.Config(app, host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)
    
    # Run the server synchronously
    server.run()