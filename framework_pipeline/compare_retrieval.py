import os
import warnings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress HuggingFace token warnings for cleaner output
warnings.filterwarnings("ignore")

def compare_methods(query: str):
    print(f"\n{'='*70}")
    print(f"🧪 COMPARING CHUNKING METHODS FOR QUERY:")
    print(f"'{query}'")
    print(f"{'='*70}")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # ---------------------------------------------------------
    # 1. Baseline Retrieval (Recursive Character)
    # ---------------------------------------------------------
    print("\n\n" + "-"*50)
    print("--- 1. BASELINE (Recursive Character) ---")
    print("-" * 50)
    baseline_db = Chroma(
        collection_name="eu_ai_act_baseline",
        embedding_function=embeddings,
        persist_directory="../framework_chroma_db"
    )
    # Fetch top 2 most similar chunks
    baseline_results = baseline_db.similarity_search(query, k=2)
    for i, res in enumerate(baseline_results):
        print(f"\nResult {i+1} (Metadata: {res.metadata}):\n{res.page_content[:400]}...\n[Total Length: {len(res.page_content)} chars]")


    # ---------------------------------------------------------
    # 2. Semantic Chunking Retrieval
    # ---------------------------------------------------------
    print("\n\n" + "-"*50)
    print("--- 2. SEMANTIC CHUNKING ---")
    print("-" * 50)
    semantic_db = Chroma(
        collection_name="eu_ai_act_semantic_chunks",
        embedding_function=embeddings,
        persist_directory="../framework_chroma_db"
    )
    # Fetch top 2 most similar chunks
    semantic_results = semantic_db.similarity_search(query, k=2)
    for i, res in enumerate(semantic_results):
        print(f"\nResult {i+1} (Metadata: {res.metadata}):\n{res.page_content[:400]}...\n[Total Length: {len(res.page_content)} chars]")


    # ---------------------------------------------------------
    # 3. Context-Enriched (ParentDocumentRetriever)
    # ---------------------------------------------------------
    print("\n\n" + "-"*50)
    print("--- 3. CONTEXT-ENRICHED (Parent-Child) ---")
    print("-" * 50)
    child_db = Chroma(
        collection_name="eu_ai_act_context_chunks",
        embedding_function=embeddings,
        persist_directory="../framework_chroma_db"
    )
    fs = LocalFileStore("../framework_docstore/eu_ai_act")
    store = create_kv_docstore(fs)
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    context_retriever = ParentDocumentRetriever(
        vectorstore=child_db,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    # Fetch top 2 parent documents based on child chunk matches
    context_results = context_retriever.invoke(query)[:2]
    for i, res in enumerate(context_results):
        print(f"\nResult {i+1} (Metadata: {res.metadata}):\n{res.page_content[:400]}...\n[Total Length: {len(res.page_content)} chars]")


if __name__ == "__main__":
    test_query = "Under what exact conditions is the use of real-time remote biometric identification systems in publicly accessible spaces by law enforcement considered an exception to the prohibited AI practices, and who must authorize it?"
    compare_methods(test_query)
