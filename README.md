# Multimodal RAG Agent with LangGraph
Intelligent multimodal AI agent that chooses between web search, document retrieval, and image analysis.

## 🚀 Quick Setup
```bash
# Clone and setup
git clone https://github.com/AnujanN/Agentic-Chatbot-with-Langraph.git
cd "AI Agent Chatbot with Langraph"

# Create environment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure API keys
echo "GROQ_API_KEY=your_key_here" > .env
echo "GEMINI_API_KEY=your_key_here" >> .env
echo "TAVILY_API_KEY=your_key_here" >> .env

# Run backend
python app_2.py

# Run UI (new terminal)
streamlit run ui_2.py
```
Open http://localhost:8501 to start chatting!

## 🎯 Features
🧠 **Smart Agent**: LangGraph ReAct agent with intelligent tool selection  
📄 **Document Q&A**: Upload PDFs and ask questions  
🖼️ **Image Analysis**: CLIP embeddings + OCR for visual understanding  
🌐 **Web Search**: Real-time information via Tavily API  
🤖 **Multi-LLM**: Gemini and Groq (Llama) model support  
🎨 **Easy Upload**: Drag-and-drop interface for files  

## 🔧 Requirements
- Python 3.8+
- API keys: [Groq](https://console.groq.com), [Gemini](https://aistudio.google.com), [Tavily](https://tavily.com)

## 📁 File Structure
```
├── app_2.py           # FastAPI backend + LangGraph agent
├── ui_2.py            # Streamlit frontend
├── requirements.txt   # Dependencies
├── Dockerfile        # Container setup
└── .env              # API keys (create this)
```

## 🐳 Docker
```bash
docker build -t ai-agent .
docker run -p 8000:8000 -p 8501:8501 --env-file .env ai-agent
```
