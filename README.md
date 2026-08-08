# 🇪🇺 EU AI Act Compliance Assistant

Welcome to the **EU AI Act Compliance Assistant**! This project is an advanced, agentic Retrieval-Augmented Generation (RAG) system designed to navigate, interpret, and answer complex legal questions regarding the European Union's Artificial Intelligence Act.

Built as a professional portfolio project, this tool moves beyond basic semantic search. It utilizes a LangGraph-powered agent architecture with self-reflection, hallucination guardrails, and a highly optimized Hybrid Retriever to guarantee precise and legally accurate citations.

## 🚀 Features

*   **Agentic LangGraph Architecture:** The core logic isn't a simple straight-through pipeline. It features:
    *   **Intent Routing:** Differentiates between casual conversation and complex legal queries to optimize API usage.
    *   **Query Decomposition:** Breaks down complicated user questions into multiple targeted sub-queries for broader context retrieval.
    *   **Self-Reflection & Hallucination Checking:** A secondary LLM evaluator acts as a "judge" to verify that the generated answer is strictly grounded in the retrieved text. If a hallucination is detected, the agent is forced to regenerate its answer.
*   **Hybrid "Context-Enriched" Retrieval:**
    *   **Dense + Sparse:** Combines the semantic understanding of Dense Embeddings (`BAAI/bge-large-en-v1.5` in ChromaDB) with the exact keyword matching of BM25.
    *   **Reciprocal Rank Fusion (RRF):** Intelligently merges the results from both retrievers.
    *   **Cross-Encoder Reranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to aggressively rerank and filter the final document list, drastically improving precision and reducing LLM context-window latency.
    *   **Parent-Child Chunking:** Retrieves small, highly-relevant chunks but feeds the LLM the larger surrounding parent document to ensure legal context isn't lost.
*   **Premium Streamlit UI:** A custom-styled, dark-themed Streamlit interface featuring modern design elements, custom avatars, and a dedicated expandable section for inline legal citations.
*   **Production Ready (Google Cloud Run):** Fully Dockerized, utilizing pre-baked HuggingFace models for zero cold-start delays, and native Google Cloud Vertex AI integration for seamless enterprise-grade authentication.

## 🛠️ Tech Stack

*   **UI / Frontend:** Streamlit, Custom CSS
*   **Orchestration:** LangChain, LangGraph
*   **LLM Engine:** Google Cloud Vertex AI (`gemini-2.5-flash`)
*   **Vector Database:** ChromaDB (Local Pre-built)
*   **Embeddings & Reranking:** HuggingFace `sentence-transformers`
*   **Deployment:** Docker, Google Cloud Run

## ⚙️ Setup & Deployment

This application is designed for seamless, serverless deployment on **Google Cloud Run** using native **Vertex AI** authentication. No API keys are required!

### 1. Clone the repository
```bash
git clone https://github.com/miteshkumar07/EU-AI-Act-Compliance-Assistant.git
cd EU-AI-Act-Compliance-Assistant
```

*(Note: The main backend framework pipeline logic lives inside the `framework_pipeline/` directory.)*

### 2. Google Cloud Setup
1. Create a Google Cloud Project.
2. Enable the **Cloud Run API** and the **Agent Platform API** (Vertex AI).
3. Ensure your default Compute Engine Service Account has the **Vertex AI User** (`roles/aiplatform.user`) role. You can set this via the CLI:
   ```bash
   gcloud projects add-iam-policy-binding [YOUR_PROJECT_ID] \
       --member="serviceAccount:[YOUR_PROJECT_NUMBER]-compute@developer.gserviceaccount.com" \
       --role="roles/aiplatform.user"
   ```

### 3. Deploy to Cloud Run
Since the vector databases (`framework_chroma_db/` and `framework_docstore/`) are pre-built and included in the repository, and the HuggingFace models are pre-baked during the Docker build phase, deployment is instant and cold starts are eliminated.

Connect your GitHub repository directly to Google Cloud Run, and deploy! The `Dockerfile` handles everything automatically.

### 4. Run Locally (Optional)
If you prefer to run the application locally for testing:
1. Ensure Docker is installed.
2. Authenticate with Google Cloud locally using: `gcloud auth application-default login`.
3. Build and run the container:
   ```bash
   docker build -t ai-act-assistant .
   docker run -p 8501:8501 ai-act-assistant
   ```
4. Access the UI at `http://localhost:8501`.

## 🤝 Let's Connect!

Exploring the intersection of law and AI has been a fascinating challenge. If you'd like to discuss the technical implementation, share feedback, or explore potential collaborations, please reach out!

*   [LinkedIn](https://www.linkedin.com/in/mitesh-kumar0707/)
*   [GitHub](https://github.com/miteshkumar07)
*   [Portfolio Website](https://miteshkumar.com/)