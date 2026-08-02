import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

def build_ensemble_retriever():
    """
    Builds a Hybrid Retriever combining BM25 (Sparse/Keyword) 
    and ChromaDB (Dense/Semantic) using BAAI/bge-large-en-v1.5.
    """
    print("Loading Semantic (Dense) Retriever...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 1. Load the Baseline ChromaDB
    db = Chroma(
        collection_name="eu_ai_act_baseline",
        embedding_function=embeddings,
        persist_directory="../framework_chroma_db"
    )
    # Configure it to return the top 5 most semantically relevant chunks
    dense_retriever = db.as_retriever(search_kwargs={"k": 5})
    
    # 2. Build the BM25 Keyword Retriever
    print("Building Keyword (Sparse) Retriever...")
    # Extract all the text from Chroma to build the keyword index
    all_data = db.get()
    docs = []
    for text, metadata in zip(all_data['documents'], all_data['metadatas']):
        docs.append(Document(page_content=text, metadata=metadata))
        
    # BM25 operates entirely in memory on the raw text
    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = 5
    
    # 3. Combine them using Reciprocal Rank Fusion (RRF)
    print("Fusing Retrievers via Reciprocal Rank Fusion (RRF)...")
    # Weights: Give slightly more weight to semantic meaning (0.6) than exact keywords (0.4)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4]
    )
    
    return ensemble_retriever

if __name__ == "__main__":
    retriever = build_ensemble_retriever()
    
    # Test a tricky query that requires both concept matching and exact string matching
    query = "What does Article 14 say about human oversight?"
    print(f"\nTESTING HYBRID SEARCH:\nQuery: '{query}'")
    print("-" * 50)
    
    # Fire the search!
    results = retriever.invoke(query)
    
    print(f"Returned {len(results)} highly-ranked chunks.")
    print(f"TOP RESULT METADATA: {results[0].metadata}")
    print(f"TOP RESULT CONTENT:\n{results[0].page_content[:400]}...")
