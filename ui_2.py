"""
Enhanced UI for Multimodal LangGraph Agent
Clean interface with improved error handling and connection reliability
"""

import streamlit as st
import requests
from PIL import Image
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="🤖 Enhanced AI Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

def check_backend_with_retry(max_retries=3, delay=1):
    """Check backend with retry mechanism"""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=10)
            if response.status_code == 200:
                return True, response.json()
            else:
                return False, {"error": f"Backend returned status {response.status_code}"}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            return False, {"error": "Cannot connect to backend - is it running on port 8000?"}
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
            return False, {"error": "Backend connection timeout"}
        except Exception as e:
            return False, {"error": f"Backend check failed: {str(e)}"}
    
    return False, {"error": "Failed after multiple attempts"}

def upload_pdf(file):
    """Upload PDF file"""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload_pdf", files=files, timeout=30)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def upload_image(file, description=""):
    """Upload image file"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {"description": description}
        response = requests.post(f"{API_BASE_URL}/upload_image", files=files, data=data, timeout=30)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def send_message(message, model):
    """Send chat message"""
    try:
        payload = {"message": message, "model": model}
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=60)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_system_status():
    """Get detailed system status"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=10)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

# Main UI
st.title("🤖 Enhanced Multimodal AI Agent")
st.caption("Powered by LangGraph • CLIP Embeddings • Advanced RAG")

# Check backend connection
with st.spinner("🔍 Connecting to backend..."):
    backend_ok, health_data = check_backend_with_retry()

if not backend_ok:
    st.error("❌ **Backend Connection Failed**")
    error_msg = health_data.get("error", "Unknown connection error")
    st.error(f"**Details:** {error_msg}")
    
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Steps to fix:**
        1. Make sure the backend is running: `python app_2.py`
        2. Check if port 8000 is available
        3. Verify your .env file has the required API keys
        4. Try restarting both backend and UI
        """)
    
    if st.button("🔄 Retry Connection"):
        st.rerun()
    
    st.stop()

# Success - show backend info
st.success("✅ **Backend Connected Successfully**")

# Get detailed status
status_ok, status_data = get_system_status()
if status_ok:
    features = status_data.get("features", {})
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 Documents", status_data.get("content", {}).get("documents", 0))
    with col2:
        st.metric("🖼️ Images", status_data.get("content", {}).get("images", 0))
    with col3:
        web_status = "✅" if features.get("web_search") else "❌"
        st.metric("🌐 Web Search", web_status)

st.divider()

# Sidebar with enhanced options
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    models = health_data.get("available_models", [])
    if models:
        model_names = {
            "gemini_flash": "🚀 Gemini 2.5 Flash",
            "gemini_pro": "⭐ Gemini 2.5 Pro", 
            "groq_llama_70b": "🦙 Llama 3.3 70B",
            "groq_llama_8b": "⚡ Llama 3.1 8B"
        }
        selected_model = st.selectbox(
            "🧠 AI Model",
            options=models,
            format_func=lambda x: model_names.get(x, x)
        )
    else:
        st.error("No models available")
        selected_model = None
    
    st.divider()
    
    # Enhanced file uploads
    st.subheader("📁 Upload Content")
    
    # PDF upload
    pdf_file = st.file_uploader("📄 Upload PDF Document", type=['pdf'])
    if pdf_file and st.button("🔄 Process PDF", use_container_width=True):
        with st.spinner("Processing PDF with multimodal capabilities..."):
            success, result = upload_pdf(pdf_file)
            if success:
                st.success("✅ PDF processed successfully!")
                if 'total_documents' in result:
                    st.info(f"📊 Processed {result['text_chunks']} text chunks and {result['images_processed']} images")
            else:
                st.error(f"❌ Error: {result.get('error', 'Failed')}")
    
    # Image upload
    image_file = st.file_uploader("🖼️ Upload Image", type=['png', 'jpg', 'jpeg'])
    if image_file:
        image_desc = st.text_input("📝 Image description (optional)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Process Image", use_container_width=True):
                with st.spinner("Processing image..."):
                    success, result = upload_image(image_file, image_desc)
                    if success:
                        st.success("✅ Image processed!")
                    else:
                        st.error(f"❌ Error: {result.get('error', 'Failed')}")
        with col2:
            if st.button("👁️ Preview", use_container_width=True):
                st.image(image_file, caption="Uploaded Image", use_column_width=True)
    
    st.divider()
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        if st.button("📊 System Status", use_container_width=True):
            if status_ok:
                st.json(status_data)
            else:
                st.error("Cannot fetch system status")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()

# Main chat interface
st.subheader("💬 Chat Interface")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about documents, images, or search the web..."):
    if not selected_model:
        st.error("Please select a model first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking and using tools..."):
                success, response = send_message(prompt, selected_model)
                
                if success:
                    reply = response.get("response", "No response")
                    st.markdown(reply)
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    
                    # Show additional info
                    col1, col2 = st.columns(2)
                    with col1:
                        model_used = response.get("model_used", selected_model)
                        st.caption(f"🧠 Model: {model_used}")
                    with col2:
                        tools_used = response.get("tools_used", [])
                        if tools_used:
                            st.caption(f"🔧 Tools: {', '.join(tools_used)}")
                        else:
                            st.caption("🔧 Tools: None used")
                else:
                    error_msg = f"❌ Error: {response.get('error', 'Failed to get response')}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer with enhanced information
st.divider()
with st.expander("ℹ️ Agent Capabilities & Usage"):
    st.markdown("""
    ### 🎯 **This Enhanced AI Agent Can:**
    
    #### 📄 **Document Processing**
    - Upload and analyze PDF documents
    - Extract text from PDFs with intelligent chunking
    - Extract and process images embedded in PDFs
    - OCR text recognition from images (when available)
    - Cross-modal content pairing (text + images from same pages)
    
    #### 🖼️ **Image Analysis**
    - Upload and analyze standalone images
    - CLIP-based image embeddings for semantic search
    - Image description and content understanding
    - Visual similarity search across uploaded content
    
    #### 🌐 **Web Search**
    - Real-time web search for current information
    - News and factual information retrieval
    - Supplement document knowledge with web data
    
    #### 🧠 **Intelligent Agent Features**
    - **LangGraph ReAct Agent**: Intelligently chooses the best tools
    - **Multi-model Support**: Gemini and Llama models
    - **Context Awareness**: Understands when to use different data sources
    - **Multimodal RAG**: Combines text, images, and web search seamlessly
    
    ### 💡 **Best Practices:**
    - Upload relevant documents before asking questions
    - Be specific in your queries for better results
    - Use image descriptions for better searchability
    - The agent will automatically choose the best tools for your question
    """)

st.markdown("---")
st.markdown("🚀 **Enhanced Multimodal AI Agent** • Built with LangGraph, CLIP, and Advanced RAG")