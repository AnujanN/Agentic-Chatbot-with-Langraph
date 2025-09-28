FROM python:3.13-slim

COPY . /app

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose both FastAPI and Streamlit ports
EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
 
