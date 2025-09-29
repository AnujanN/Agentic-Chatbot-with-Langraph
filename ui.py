import streamlit as st
import requests

st.set_page_config(page_title="AI Agent Chatbot with Langraph", layout="centered")

API_URL = "http://127.0.0.1:8000/chat"

MODEL_NAMES = [
    "llama-3.3-70b-versatile",  
    "llama-3.1-8b-instant",
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

st.title("🤖 LangGraph AI Agent Chatbot")
st.write("Interact with the LangGraph-based agent using this interface.")

# Add connection status check
@st.cache_data(ttl=60)
def check_api_status():
    try:
        response = requests.get("http://127.0.0.1:8000/docs", timeout=5)
        return response.status_code == 200
    except:
        return False

api_status = check_api_status()
if api_status:
    st.success("✅ API Server is running")
else:
    st.error("❌ API Server is not running. Please start your FastAPI server first.")
    st.info("💡 Run your notebook cell with the FastAPI server to start it.")


given_system_prompt = st.text_area("Define you AI Agent:", height=10, placeholder="Type your system prompt here...")

selected_model = st.selectbox("Select Model:", MODEL_NAMES)

user_input = st.text_area("Your Message:", height=100, placeholder="Type your message here...")


if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # Create payload in the format expected by RequestState from notebook
        payload = {
            "model_name": selected_model,
            "system_prompt": given_system_prompt or "You are a helpful AI assistant.",
            "messages": [user_input]
        }
        
        try:
            # Show what we're sending (for debugging)
            with st.expander("🔍 Debug Info (Request Payload)"):
                st.json(payload)
            
            with st.spinner("🤔 Agent is thinking..."):
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()
            
            # Display the response
            st.success("Response from Agent:")
            
            # Handle ChatResponse format from FastAPI
            if isinstance(data, dict):
                if "answer" in data:
                    # This is the expected ChatResponse format
                    st.write(data["answer"])
                    
                    # Show model info if available
                    if "model_used" in data and "provider_used" in data:
                        st.info(f"🤖 Model: {data['provider_used']}/{data['model_used']}")
                
                elif "error" in data:
                    # Handle error responses
                    st.error(f"Error: {data['error']}")
                
                elif "messages" in data and data["messages"]:
                    # Fallback for old format
                    last_message = data["messages"][-1]
                    if isinstance(last_message, dict) and "content" in last_message:
                        st.write(last_message["content"])
                    else:
                        st.write(str(last_message))
                else:
                    st.write(str(data))
            else:
                st.write(str(data))
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the API: {e}")
        except Exception as e:
            st.error(f"Error processing response: {e}")