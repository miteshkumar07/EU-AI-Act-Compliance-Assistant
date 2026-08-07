import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def inspect_collection(collection_name, description):
    print(f"\n{'='*60}")
    print(f"🔍 INSPECTING: {description}")
    print(f"Collection Name: {collection_name}")
    print(f"{'='*60}")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory="../framework_chroma_db"
    )
    
    # Get all data from the collection
    collection_data = db.get()
    
    total_chunks = len(collection_data['ids'])
    print(f"Total Chunks in Database: {total_chunks}")
    
    if total_chunks > 0:
        print("\n--- SAMPLE CHUNK ---")
        # Print the first document and its metadata
        print(f"Metadata: {collection_data['metadatas'][0]}")
        print("-" * 20)
        
        content = collection_data['documents'][0]
        # Truncate content for display purposes if it's too long
        if len(content) > 500:
            print(f"{content[:500]}...\n[TRUNCATED... Total Length: {len(content)} chars]")
        else:
            print(content)
            
if __name__ == "__main__":
    inspect_collection("eu_ai_act_baseline", "1. Baseline (Recursive Character)")
    inspect_collection("eu_ai_act_semantic_chunks", "2. Semantic Chunking")
    inspect_collection("eu_ai_act_context_chunks", "3. Context-Enriched (Child Chunks)")
