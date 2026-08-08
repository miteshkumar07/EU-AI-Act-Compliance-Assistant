FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for vector database extensions (like hnswlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace models so they are baked into the Docker image
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; \
from langchain_community.cross_encoders import HuggingFaceCrossEncoder; \
HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5', model_kwargs={'device': 'cpu'}); \
HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', model_kwargs={'device': 'cpu'})"

# Copy only the necessary files and folders for the deployment
COPY app.py .
COPY data/ ./data/
COPY framework_pipeline/ ./framework_pipeline/

COPY framework_chroma_db/ ./framework_chroma_db/
COPY framework_docstore/ ./framework_docstore/

# Set working directory to where the Streamlit app lives (root)
WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
