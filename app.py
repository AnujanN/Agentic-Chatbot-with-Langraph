from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

class RequestState(BaseModel):
    model_name: str
    system_prompt: str 
    messages: List[str]

@app.post("/chat")
def chat_endpoint(request: RequestState):
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
        except Exception as e:
            return {"error": f"Error initializing Gemini model: {str(e)}"}
    else:
        if not groq_api_key:
            return {"error": "GROQ_API_KEY not configured"}
        llm = ChatGroq(model_name=request.model_name, api_key=groq_api_key)

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    state = {
        "messages": [{"role": "system", "content": request.system_prompt}]
                   + [{"role": "user", "content": msg} for msg in request.messages]
    }

    try:
        result = agent.invoke(state)
        return result
    except Exception as e:
        return {"error": f"Error processing request: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    config = uvicorn.Config(app, host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)
    
    # Run the server synchronously for standalone Python execution
    server.run()