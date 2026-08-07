FROM python:3.12-slim

WORKDIR /app

# Install build dependencies for vector database extensions (like hnswlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
