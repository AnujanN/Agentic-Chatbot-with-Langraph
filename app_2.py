"""
Enhanced Multimodal LangGraph Agent with CLIP-based RAG
Combines LangGraph ReAct agent with advanced multimodal capabilities
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
import io
import base64
from transformers import CLIPProcessor, CLIPModel
from langchain_tavily import TavilySearch
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

# Global variables for multimodal system
vector_store = None
faiss_index = None
multimodal_data = {}
image_data_store = {}
all_docs = []
clip_model = None
clip_processor = None
embeddings = None
counter = 0

# Initialize FastAPI app
app = FastAPI(title="Enhanced Multimodal LangGraph Agent", version="2.0.0")

def initialize_models():
    """Initialize CLIP model and embeddings"""
    global clip_model, clip_processor, embeddings, faiss_index
    
    try:
        # Initialize CLIP model for multimodal embeddings
        model_id = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(model_id)
        clip_processor = CLIPProcessor.from_pretrained(model_id)
        clip_model.eval()
        print("✅ CLIP model loaded successfully")
        
        # Initialize text embeddings for fallback
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Text embeddings loaded successfully")
        
        # Initialize FAISS index for multimodal content
        embedding_dim = 512  # CLIP embedding dimension
        faiss_index = faiss.IndexFlatL2(embedding_dim)
        print("✅ FAISS index initialized")
        
        return True
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return False

def embed_image(image_data):
    """Create CLIP embedding for an image"""
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data
    
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text):
    """Create CLIP embedding for text"""
    inputs = clip_processor(
        text=text, return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def process_pdf_multimodal(pdf_file):
    """Process PDF with advanced multimodal capabilities"""
    global all_docs, image_data_store
    
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    all_docs_local = []
    all_embeddings = []
    image_data_store_local = {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    for i, page in enumerate(doc):
        text = page.get_text()
        images = list(page.get_images(full=True))
        text_chunks = []
        
        # Process text content
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            for chunk in text_chunks:
                embedding = embed_text(chunk.page_content)
                all_embeddings.append(embedding)
                all_docs_local.append(chunk)
        
        # Process images (with optional OCR)
        if text_chunks or images:
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_id = f"page_{i}_img_{img_index}"
                    
                    # Store image as base64
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data_store_local[image_id] = img_base64
                    
                    # OCR: extract text from image (optional - graceful fallback)
                    ocr_text = ""
                    try:
                        import pytesseract
                        ocr_text = pytesseract.image_to_string(pil_image)
                    except ImportError:
                        ocr_text = "[OCR not available - install pytesseract for text extraction]"
                    except Exception:
                        ocr_text = "[OCR extraction failed]"
                    
                    # Create image embedding
                    embedding = embed_image(pil_image)
                    all_embeddings.append(embedding)
                    
                    # Create document with OCR text for better retrieval
                    image_doc = Document(
                        page_content=f"[Image: {image_id}] OCR: {ocr_text.strip()}",
                        metadata={"page": i, "type": "image", "image_id": image_id},
                    )
                    all_docs_local.append(image_doc)
                except Exception as e:
                    print(f"Error processing image {img_index} on page {i}: {e}")
                    continue
    
    doc.close()
    
    # Update global stores
    all_docs.extend(all_docs_local)
    image_data_store.update(image_data_store_local)
    
    return all_docs_local, np.array(all_embeddings), image_data_store_local

def build_vector_store_advanced(all_docs, embeddings_array):
    """Build vector store with pre-computed embeddings"""
    # Use text embeddings for compatibility with LangChain FAISS
    return LangchainFAISS.from_embeddings(
        text_embeddings=[
            (doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)
        ],
        embedding=embeddings,  # Use text embeddings for compatibility
        metadatas=[doc.metadata for doc in all_docs],
    )

def retrieve_multimodal_content(query, k=5):
    """Advanced multimodal retrieval with cross-modal matching"""
    if vector_store is None:
        return []
    
    # Create query embedding using CLIP
    query_embedding = embed_text(query)
    results = vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)
    
    # Enhance with page-based pairing
    text_docs = [doc for doc in results if doc.metadata.get("type") == "text"]
    text_pages = {doc.metadata.get("page") for doc in text_docs}
    
    # Find image docs from the same pages
    paired_image_docs = [
        doc for doc in all_docs
        if doc.metadata.get("type") == "image" and doc.metadata.get("page") in text_pages
    ]
    
    # Remove duplicates and combine
    image_ids_included = {
        doc.metadata.get("image_id") for doc in results 
        if doc.metadata.get("type") == "image"
    }
    new_image_docs = [
        doc for doc in paired_image_docs 
        if doc.metadata.get("image_id") not in image_ids_included
    ]
    
    return results + new_image_docs[:3]  # Limit additional images

def setup_llms():
    """Setup LLM clients"""
    llms = {}
    
    # Groq models
    if os.getenv("GROQ_API_KEY"):
        llms.update({
            "groq_llama_70b": ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama-3.3-70b-versatile"
            ),
            "groq_llama_8b": ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"), 
                model_name="llama-3.1-8b-instant"
            )
        })
    
    # Gemini models
    if os.getenv("GEMINI_API_KEY"):
        llms.update({
            "gemini_flash": ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GEMINI_API_KEY"),
                model="gemini-2.5-flash"
            ),
            "gemini_pro": ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GEMINI_API_KEY"),
                model="gemini-2.5-pro"
            )
        })
    
    return llms

def create_web_search_tool():
    """Create web search tool"""
    if not os.getenv("TAVILY_API_KEY"):
        return None
    
    try:
        search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))
        
        def search_web(query: str) -> str:
            try:
                results = search.search(query)
                if results:
                    return f"Web search results for '{query}':\\n" + "\\n".join([
                        f"- {result.get('title', 'No title')}: {result.get('content', 'No content')}"
                        for result in results[:3]
                    ])
                return "No web search results found."
            except Exception as e:
                return f"Web search error: {e}"
        
        return Tool(
            name="web_search",
            description="Search the web for current information and news",
            func=search_web
        )
    except Exception as e:
        print(f"Error creating web search tool: {e}")
        return None

def create_document_retrieval_tool():
    """Create document retrieval tool"""
    def retrieve_documents(query: str) -> str:
        if vector_store is None:
            return "No documents uploaded. Please upload PDF documents first."
        
        try:
            context_docs = retrieve_multimodal_content(query, k=5)
            
            if not context_docs:
                return "No relevant documents found."
            
            results = []
            for doc in context_docs:
                doc_type = doc.metadata.get("type", "text")
                page = doc.metadata.get("page", "unknown")
                
                if doc_type == "text":
                    preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    results.append(f"Text from page {page}: {preview}")
                elif doc_type == "image":
                    image_id = doc.metadata.get("image_id")
                    results.append(f"Image from page {page} ({image_id}): {doc.page_content}")
            
            return "Document retrieval results:\\n" + "\\n---\\n".join(results)
            
        except Exception as e:
            return f"Error retrieving documents: {e}"
    
    return Tool(
        name="document_retrieval",
        description="Retrieve information from uploaded PDF documents with text and images",
        func=retrieve_documents
    )

def create_image_analysis_tool():
    """Create image analysis tool"""
    def analyze_images(query: str) -> str:
        if faiss_index is None or faiss_index.ntotal == 0:
            return "No images uploaded. Please upload images first."
        
        try:
            # Search standalone images
            query_embedding = embed_text(query).reshape(1, -1)
            k = min(3, faiss_index.ntotal)
            distances, indices = faiss_index.search(query_embedding, k)
            
            results = []
            for idx in indices[0]:
                if idx != -1 and idx in multimodal_data:
                    content = multimodal_data[idx]
                    results.append(f"Image: {content.get('description', 'Visual content')}")
            
            return "Image analysis results:\\n" + "\\n".join(results) if results else "No relevant images found."
            
        except Exception as e:
            return f"Error analyzing images: {e}"
    
    return Tool(
        name="image_analysis",
        description="Analyze and search through uploaded standalone images",
        func=analyze_images
    )

def create_integrated_agent(model_key: str = "gemini_flash"):
    """Create LangGraph agent with all tools"""
    llms = setup_llms()
    
    if model_key not in llms:
        raise ValueError(f"Model {model_key} not available")
    
    llm = llms[model_key]
    
    # Create tools
    tools = []
    
    # Web search
    web_tool = create_web_search_tool()
    if web_tool:
        tools.append(web_tool)
    
    # Document retrieval
    tools.append(create_document_retrieval_tool())
    
    # Image analysis
    tools.append(create_image_analysis_tool())
    
    # Create agent
    agent = create_react_agent(llm, tools)
    return agent

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    llms = setup_llms()
    return {
        "status": "healthy",
        "available_models": list(llms.keys()),
        "text_documents": "available" if vector_store else "none",
        "images": faiss_index.ntotal if faiss_index else 0,
        "web_search": "available" if os.getenv("TAVILY_API_KEY") else "unavailable"
    }

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    global vector_store
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Process PDF with multimodal capabilities
        file.file.seek(0)
        all_docs_local, embeddings_array, image_data_store_local = process_pdf_multimodal(file.file)
        
        if len(all_docs_local) == 0:
            raise HTTPException(status_code=400, detail="No content found in PDF")
        
        # Build vector store
        if vector_store is None:
            vector_store = build_vector_store_advanced(all_docs_local, embeddings_array)
        else:
            vector_store.add_documents(all_docs_local)
        
        # Count documents
        num_text = sum(1 for doc in all_docs_local if doc.metadata.get("type") == "text")
        num_image = sum(1 for doc in all_docs_local if doc.metadata.get("type") == "image")
        
        return {
            "message": f"PDF '{file.filename}' processed successfully",
            "total_documents": len(all_docs_local),
            "text_chunks": num_text,
            "images_processed": num_image,
            "multimodal_features": True,
            "filename": file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...), description: str = ""):
    """Upload and process standalone image"""
    global counter
    
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
            multimodal_embedding = embed_image(image).reshape(1, -1)
        
        # Add to FAISS index
        faiss_index.add(multimodal_embedding)
        
        # Store metadata
        multimodal_data[counter] = {
            "type": "image",
            "filename": file.filename,
            "description": description,
            "embedding_id": counter
        }
        counter += 1
        
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
        # Create agent
        agent = create_integrated_agent(request.model)
        
        # Enhanced prompt
        enhanced_message = f"""You are an intelligent multimodal AI agent with access to multiple tools:

Available capabilities:
- Web search for current information
- Document retrieval from uploaded PDFs (including images with OCR)
- Image analysis from standalone uploads

User query: {request.message}

Use the most appropriate tools to provide a comprehensive answer."""
        
        # Get response
        response = agent.invoke({"messages": [("user", enhanced_message)]})
        
        # Extract response text
        response_text = str(response.get("messages", [{}])[-1].content if response.get("messages") else "")
        
        # Simple tool detection
        tools_used = []
        if "web" in response_text.lower():
            tools_used.append("web_search")
        if "document" in response_text.lower() or "pdf" in response_text.lower():
            tools_used.append("document_retrieval")
        if "image" in response_text.lower():
            tools_used.append("image_analysis")
        
        return ChatResponse(
            response=response_text,
            model_used=request.model,
            tools_used=tools_used,
            context_sources={
                "documents": len(all_docs),
                "images": faiss_index.ntotal if faiss_index else 0
            },
            timestamp=str(os.times())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/status")
async def get_system_status():
    """Get system status"""
    llms = setup_llms()
    
    return {
        "models": {
            "available": list(llms.keys()),
            "total": len(llms)
        },
        "content": {
            "documents": len(all_docs),
            "images": faiss_index.ntotal if faiss_index else 0
        },
        "features": {
            "multimodal_rag": True,
            "clip_embeddings": True,
            "pdf_processing": True,
            "ocr_support": True,
            "web_search": bool(os.getenv("TAVILY_API_KEY"))
        },
        "api_keys": {
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "gemini": bool(os.getenv("GEMINI_API_KEY")),
            "tavily": bool(os.getenv("TAVILY_API_KEY"))
        }
    }

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models when server starts"""
    success = initialize_models()
    if not success:
        print("⚠️ Warning: Some models failed to initialize")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Enhanced Multimodal LangGraph Agent...")
    print("🎯 Features: CLIP embeddings, PDF+Image processing, OCR, Web search")
    print("🔧 Tools: Document retrieval, Image analysis, Web search")
    
    # Check API keys
    api_keys = {
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY"))
    }
    
    print("\\n🔑 API Key Status:")
    for key, available in api_keys.items():
        status = "✅" if available else "❌"
        print(f"   {status} {key}: {'Available' if available else 'Missing'}")
    
    print("\\n🌐 Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)