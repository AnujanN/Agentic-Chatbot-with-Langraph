"""
Simple Integrated UI for Multimodal LangGraph Agent
Clean and minimalist design
"""

import streamlit as st
import requests
from PIL import Image

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="🤖 AI Agent",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

def check_backend():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return response.status_code == 200, response.json()
    except:
        return False, {}

def upload_pdf(file):
    """Upload PDF file"""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload_pdf", files=files)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def upload_image(file, description=""):
    """Upload image file"""
    try:
        files = {"file": (file.name, file, file.type)}
        data = {"description": description}
        response = requests.post(f"{API_BASE_URL}/upload_image", files=files, data=data)
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

# Main UI
st.title("🤖 AI Agent Chat")

# Check backend
backend_ok, health_data = check_backend()
if not backend_ok:
    st.error("❌ Backend not running! Start with: `python app_2.py`")
    st.stop()

# Simple sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    models = health_data.get("available_models", [])
    if models:
        model_names = {
            "gemini_flash": "Gemini Flash",
            "gemini_pro": "Gemini Pro", 
            "groq_llama_70b": "Llama 70B",
            "groq_llama_8b": "Llama 8B"
        }
        selected_model = st.selectbox(
            "AI Model",
            options=models,
            format_func=lambda x: model_names.get(x, x)
        )
    else:
        st.error("No models available")
        selected_model = None
    
    st.divider()
    
    # File uploads
    st.subheader("📁 Upload Files")
    
    # PDF upload
    pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
    if pdf_file and st.button("📄 Process PDF"):
        with st.spinner("Processing..."):
            success, result = upload_pdf(pdf_file)
            if success:
                st.success("✅ PDF uploaded!")
            else:
                st.error(f"❌ Error: {result.get('error', 'Failed')}")
    
    # Image upload
    image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    if image_file:
        image_desc = st.text_input("Image description (optional)")
        if st.button("🖼️ Process Image"):
            with st.spinner("Processing..."):
                success, result = upload_image(image_file, image_desc)
                if success:
                    st.success("✅ Image uploaded!")
                else:
                    st.error(f"❌ Error: {result.get('error', 'Failed')}")
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat interface
st.subheader("💬 Chat")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            success, response = send_message(prompt, selected_model)
            
            if success:
                reply = response.get("response", "No response")
                st.markdown(reply)
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
                # Show model used
                model_used = response.get("model_used", selected_model)
                st.caption(f"Model: {model_used}")
            else:
                error_msg = f"❌ Error: {response.get('error', 'Failed to get response')}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
with st.expander("ℹ️ How to use"):
    st.markdown("""
    **This AI agent can:**
    
    • 🌐 **Search the web** for current information
    • 📄 **Analyze PDFs** - Upload documents and ask questions
    • 🖼️ **Understand images** - Upload images and discuss them
    • 🤖 **Chat naturally** - Ask any question
    
    **Tips:**
    • Upload files first, then ask questions about them
    • Try different AI models for different tasks
    • Be specific in your questions for better results
    """)