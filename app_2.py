"""
Integrated Multimodal LangGraph Agent
Combines original LangGraph agent with multimodal RAG capabilities
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import fitz  # PyMuPDF
import torch
import faiss
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    model: str = "gemini_flash"

class ChatResponse(BaseModel):
    response: str
    model_used: str
    tools_used: List[str]
    context_sources: Dict[str, int]
    timestamp: str

# Global variables
vector_store = None
multimodal_index = None
multimodal_data = {}
multimodal_counter = 0
clip_model = None
clip_processor = None
embeddings = None

# Initialize FastAPI app
app = FastAPI(title="Integrated Multimodal LangGraph Agent", version="1.0.0")

def initialize_models():
    """Initialize all required models"""
    global embeddings, clip_model, clip_processor, multimodal_index
    
    try:
        # Text embeddings for documents
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Text embeddings model loaded")
        
        # CLIP model for multimodal content
        clip_model_id = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(clip_model_id)
        clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        print("✅ CLIP model loaded")
        
        # Initialize multimodal FAISS index
        clip_embedding_dim = 512
        multimodal_index = faiss.IndexFlatL2(clip_embedding_dim)
        print("✅ Multimodal vector store initialized")
        
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")

# Initialize models on startup
initialize_models()

def setup_llms():
    """Setup LLM clients"""
    llms = {
        "groq_llama_70b": ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"
        ) if os.getenv("GROQ_API_KEY") else None,
        
        "groq_llama_8b": ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="llama-3.1-8b-instant"
        ) if os.getenv("GROQ_API_KEY") else None,
        
        "gemini_flash": ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.5-flash"
        ) if os.getenv("GEMINI_API_KEY") else None,

        "gemini_pro": ChatGoogleGenerativeAI(
            google_api_key=os.getenv("GEMINI_API_KEY"),
            model="gemini-2.5-pro"
        ) if os.getenv("GEMINI_API_KEY") else None
    }
    return {k: v for k, v in llms.items() if v is not None}

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def create_web_search_tool():
    """Create web search tool"""
    try:
        # Use the newer langchain-tavily package to avoid deprecation warning
        from langchain_tavily import TavilySearchAPIWrapper
        search = TavilySearchAPIWrapper(
            api_key=os.getenv("TAVILY_API_KEY"),
            max_results=5,
            search_depth="advanced"
        )
        return Tool(
            name="web_search", 
            description="Search the web for current information and recent developments",
            func=lambda query: search.run(query)
        )
    except Exception as e:
        print(f"⚠️ Web search tool unavailable: {e}")
        return None

def create_document_retrieval_tool():
    """Create document retrieval tool"""
    def retrieve_documents(query: str) -> str:
        if vector_store is None:
            return "No documents have been uploaded yet. Please upload PDF documents first."
        
        try:
            docs = vector_store.similarity_search_with_score(query, k=3)
            if not docs:
                return "No relevant documents found for this query."
            
            results = []
            for doc, score in docs:
                results.append(f"Document: {doc.page_content}")
            
            return "Relevant document content:\n" + "\n---\n".join(results)
        except Exception as e:
            return f"Error retrieving documents: {e}"
    
    return Tool(
        name="document_retrieval",
        description="Retrieve relevant information from uploaded PDF documents",
        func=retrieve_documents
    )

def create_multimodal_retrieval_tool():
    """Create multimodal content retrieval tool"""
    def retrieve_multimodal_content(query: str) -> str:
        if multimodal_index.ntotal == 0:
            return "No images have been uploaded yet. Please upload images first."
        
        try:
            # Create query embedding
            inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
            with torch.no_grad():
                query_embedding = clip_model.get_text_features(**inputs).cpu().numpy()
            
            # Search multimodal index
            k = min(3, multimodal_index.ntotal)
            distances, indices = multimodal_index.search(query_embedding, k)
            
            results = []
            for idx in indices[0]:
                if idx != -1 and idx in multimodal_data:
                    content = multimodal_data[idx]
                    if content.get('description'):
                        results.append(f"Image: {content['description']}")
                    else:
                        results.append("Image: Visual content available")
            
            if not results:
                return "No relevant visual content found for this query."
            
            return "Relevant visual content:\n" + "\n---\n".join(results)
        except Exception as e:
            return f"Error retrieving visual content: {e}"
    
    return Tool(
        name="multimodal_retrieval",
        description="Retrieve relevant information from uploaded images and visual content",
        func=retrieve_multimodal_content
    )

def create_integrated_agent(model_key: str = "gemini_flash"):
    """Create LangGraph agent with all tools"""
    llms = setup_llms()
    
    if model_key not in llms:
        raise ValueError(f"Model {model_key} not available")
    
    llm = llms[model_key]
    
    # Create all available tools
    tools = []
    
    # Web search tool
    web_tool = create_web_search_tool()
    if web_tool:
        tools.append(web_tool)
    
    # Document retrieval tool
    doc_tool = create_document_retrieval_tool()
    tools.append(doc_tool)
    
    # Multimodal retrieval tool
    multimodal_tool = create_multimodal_retrieval_tool()
    tools.append(multimodal_tool)
    
    # Create the agent
    agent = create_react_agent(llm, tools)
    return agent

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    llms = setup_llms()
    return {
        "status": "healthy",
        "available_models": list(llms.keys()),
        "text_documents": "available" if vector_store else "none",
        "multimodal_content": multimodal_index.ntotal if multimodal_index else 0,
        "web_search": "available" if os.getenv("TAVILY_API_KEY") else "unavailable"
    }

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    global vector_store
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(file.file)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata={"source": file.filename}) for chunk in chunks]
        
        # Create or update vector store
        if vector_store is None:
            vector_store = LangchainFAISS.from_documents(documents, embeddings)
        else:
            vector_store.add_documents(documents)
        
        return {
            "message": f"PDF '{file.filename}' processed successfully",
            "chunks": len(chunks),
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...), description: str = ""):
    """Upload and process image with optional description"""
    global multimodal_counter
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        # Load image
        image = Image.open(file.file)
        
        # Create multimodal embedding
        if description:
            # Combined text and image embedding
            inputs = clip_processor(
                text=[description], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            )
            with torch.no_grad():
                text_features = clip_model.get_text_features(**inputs).cpu().numpy()
                image_features = clip_model.get_image_features(**inputs).cpu().numpy()
                multimodal_embedding = (text_features + image_features) / 2
        else:
            # Image-only embedding
            inputs = clip_processor(images=[image], return_tensors="pt", padding=True)
            with torch.no_grad():
                multimodal_embedding = clip_model.get_image_features(**inputs).cpu().numpy()
        
        # Add to FAISS index
        multimodal_index.add(multimodal_embedding)
        
        # Store metadata
        multimodal_data[multimodal_counter] = {
            "type": "image",
            "filename": file.filename,
            "description": description,
            "embedding_id": multimodal_counter
        }
        multimodal_counter += 1
        
        return {
            "message": f"Image '{file.filename}' processed successfully",
            "description": description,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Chat with the integrated multimodal agent"""
    try:
        # Create agent with selected model
        agent = create_integrated_agent(request.model)
        
        # Get available tools count for context
        tools_available = []
        if os.getenv("TAVILY_API_KEY"):
            tools_available.append("web_search")
        if vector_store:
            tools_available.append("document_retrieval")
        if multimodal_index and multimodal_index.ntotal > 0:
            tools_available.append("multimodal_retrieval")
        
        # Create enhanced prompt that mentions available capabilities
        enhanced_message = f"""You are an intelligent AI agent with access to multiple information sources:

Available tools: {', '.join(tools_available)}

User query: {request.message}

Please use the most appropriate tools to provide a comprehensive answer. If relevant documents or images are available, include that information in your response."""
        
        # Get response from agent
        response = agent.invoke({"messages": [("user", enhanced_message)]})
        
        # Extract tools used (this is a simplified extraction)
        tools_used = []
        response_text = str(response.get("messages", [{}])[-1].content if response.get("messages") else "")
        
        if "web_search" in str(response):
            tools_used.append("web_search")
        if vector_store and "document" in response_text.lower():
            tools_used.append("document_retrieval")
        if multimodal_index and multimodal_index.ntotal > 0 and "image" in response_text.lower():
            tools_used.append("multimodal_retrieval")
        
        # Model name mapping
        model_names = {
            "groq_llama_70b": "Groq Llama 3.3 70B",
            "groq_llama_8b": "Groq Llama 3.1 8B", 
            "gemini_flash": "Gemini 2.5 Flash",
            "gemini_pro": "Gemini 2.5 Pro"
        }
        
        return ChatResponse(
            response=response_text,
            model_used=model_names.get(request.model, request.model),
            tools_used=tools_used,
            context_sources={
                "text_documents": vector_store.index.ntotal if vector_store else 0,
                "images": multimodal_index.ntotal if multimodal_index else 0,
                "web_search": 1 if "web_search" in tools_used else 0
            },
            timestamp=str(os.times())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    llms = setup_llms()
    
    return {
        "models": {
            "available": list(llms.keys()),
            "total": len(llms)
        },
        "content": {
            "text_documents": vector_store.index.ntotal if vector_store else 0,
            "images": multimodal_index.ntotal if multimodal_index else 0
        },
        "tools": {
            "web_search": bool(os.getenv("TAVILY_API_KEY")),
            "document_retrieval": vector_store is not None,
            "multimodal_retrieval": multimodal_index is not None and multimodal_index.ntotal > 0
        },
        "api_keys": {
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "tavily": bool(os.getenv("TAVILY_API_KEY"))
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Integrated Multimodal LangGraph Agent...")
    print("📊 System Status:")
    
    # Check API keys
    api_keys = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY"))
    }
    
    for key, available in api_keys.items():
        status = "✅" if available else "❌"
        print(f"   {status} {key}: {'Available' if available else 'Missing'}")
    
    if not any(api_keys.values()):
        print("⚠️ No API keys found! Please check your .env file.")
    
    print("\n🌐 Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)