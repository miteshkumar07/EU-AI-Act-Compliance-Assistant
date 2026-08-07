# 🇪🇺 EU AI Act Compliance Assistant

Welcome to the **EU AI Act Compliance Assistant**! This project is an advanced, agentic Retrieval-Augmented Generation (RAG) system designed to navigate, interpret, and answer complex legal questions regarding the European Union's Artificial Intelligence Act.

Built as a portfolio project, this tool moves beyond basic semantic search. It utilizes a LangGraph-powered agent architecture with self-reflection, hallucination guardrails, and a highly optimized Hybrid Retriever to guarantee precise and legally accurate citations.

## 🚀 Features

*   **Agentic LangGraph Architecture:** The core logic isn't a simple straight-through pipeline. It features:
    *   **Intent Routing:** Differentiates between casual conversation and complex legal queries.
    *   **Query Decomposition:** Breaks down complicated user questions into multiple targeted sub-queries for broader context retrieval.
    *   **Self-Reflection & Hallucination Checking:** A secondary LLM evaluator acts as a "judge" to verify that the generated answer is strictly grounded in the retrieved text. If a hallucination is detected, the agent is forced to regenerate its answer.
*   **Hybrid "Context-Enriched" Retrieval:**
    *   **Dense + Sparse:** Combines the semantic understanding of Dense Embeddings (`BAAI/bge-large-en-v1.5` in ChromaDB) with the exact keyword matching of BM25.
    *   **Reciprocal Rank Fusion (RRF):** Intelligently merges the results from both retrievers.
    *   **Cross-Encoder Reranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to aggressively rerank and filter the final document list, drastically improving precision and reducing LLM context-window latency.
    *   **Parent-Child Chunking:** Retrieves small, highly-relevant chunks but feeds the LLM the larger surrounding parent document to ensure legal context isn't lost.
*   **Premium Streamlit UI:** A custom-styled, dark-themed Streamlit interface featuring glassmorphism elements, custom avatars, and a dedicated expandable section for inline legal citations.
*   **Production Ready:** Fully Dockerized with HuggingFace caching optimized for fast restarts.

## 🛠️ Tech Stack

*   **UI / Frontend:** Streamlit, Custom CSS
*   **Orchestration:** LangChain, LangGraph
*   **LLM Engine:** Google Gemini (`gemini-2.5-flash`)
*   **Vector Database:** ChromaDB
*   **Embeddings & Reranking:** HuggingFace `sentence-transformers`
*   **Evaluation:** RAGAS (Retrieval Augmented Generation Assessment), Custom LLM-as-a-judge
*   **Deployment:** Docker, Docker Compose

## ⚙️ Local Setup & Installation

You can run this project locally with Docker. 

### Prerequisites
*   Docker & Docker Compose installed.
*   A Google Gemini API Key.
*   A HuggingFace Access Token.

### 1. Clone the repository
```bash
git clone https://github.com/miteshkumar07/EU-AI-Act-Compliance-Assistant.git
cd EU-AI-Act-Agent
```

*(Note: The main application code lives inside the `framework_pipeline` directory.)*

### 2. Set up Environment Variables
Create a `.env` file in the root directory and add your API keys:
```env
GOOGLE_API_KEY="your_gemini_api_key_here"
HF_TOKEN="your_huggingface_token_here"
```

### 3. Run with Docker Compose
Build and start the container:
```bash
docker-compose up --build
```
The first time you run this, it will download the HuggingFace embedding models. They are cached locally in a hidden `.hf_cache` folder so subsequent startups are blazing fast.

### 4. Access the UI
Open your browser and navigate to:
```text
http://localhost:8501
```

## 📊 Evaluation & Performance

Building a legal RAG requires strict adherence to facts. To ensure the highest quality, this pipeline was rigorously tested:
*   **RAGAS Metrics:** Evaluated for Answer Relevancy, Faithfulness, Context Precision, and Context Recall using the `ragas` framework.
*   **LLM-as-a-Judge:** Achieved a **4.15 / 5.0** correctness score on a custom benchmark suite of tricky EU AI Act edge-cases.
*   **Latency Optimization:** Replaced slow LLM-based document grading nodes with lightning-fast Cross-Encoder reranking, cutting response times significantly without sacrificing accuracy.

## 🤝 Let's Connect!

Exploring the intersection of law and AI has been a fascinating challenge. If you'd like to discuss the technical implementation, share feedback, or explore potential collaborations, please reach out!

*   [LinkedIn](#)
*   [GitHub](#)
*   [Portfolio](#)