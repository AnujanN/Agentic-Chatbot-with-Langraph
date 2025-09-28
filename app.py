from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

import os

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("⚠️  GROQ_API_KEY not found!")
else:
    print("✅ GROQ API key found")

os.environ["TAVILY_API_KEY"] = "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"

MODEL_NAMES = [
    "llama-3.3-70b-versatile",  
    "llama-3.1-8b-instant"       
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

    llm = ChatGroq(model_name=request.model_name, api_key=groq_api_key)

    agent = create_react_agent(
        model=llm,
        tools=tools
    )

    state = {
        "messages": [{"role": "system", "content": request.system_prompt}]
                   + [{"role": "user", "content": msg} for msg in request.messages]
    }

    result = agent.invoke(state)
    return result


if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    config = uvicorn.Config(app, host="127.0.0.1", port=8000)
    server = uvicorn.Server(config)
    
    # Run the server synchronously
    server.run()