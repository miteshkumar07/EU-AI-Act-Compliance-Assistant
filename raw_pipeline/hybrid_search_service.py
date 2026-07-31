import os
import json
import chromadb
from rank_bm25 import BM25Okapi

def legal_tokenize(text: str) -> list:
    return text.lower().replace(".", "").replace(",", "").replace("(", "").replace(")", "").split()

class HybridRetriever:
    def __init__(self, chroma_path: str = "./chroma_db", collection_name: str = "document_aware_by_article"):
        print("[HYBRID] Initializing Hybrid Search Engine...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection(name=collection_name)
        
        # Pull all raw text and metadata blocks from ChromaDB
        all_data = self.collection.get()
        self.documents = all_data['documents']
        self.metadatas = all_data['metadatas']
        self.ids = all_data['ids']
        
        # Build the sparse BM25 index over our localized corpus
        tokenized_corpus = [legal_tokenize(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"    -> BM25 sparse index successfully built over {len(self.documents)} blocks.")

    def dense_vector_search_indices(self, query: str, k: int = 15) -> list:
        """Returns the internal corpus indices of the top-k vector matches."""
        from retrieval_service import retrieve_top_k
        chunks = retrieve_top_k(query=query, collection_name="document_aware_by_article", k=k)
        
        matched_indices = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if chunk_text in self.documents:
                matched_indices.append(self.documents.index(chunk_text))
        return matched_indices

    def sparse_lexical_search_indices(self, query: str, k: int = 15) -> list:
        """Returns the internal corpus indices of the top-k keyword matches."""
        tokenized_query = legal_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair document position indices with their lexical scores
        doc_indices_with_scores = list(enumerate(scores))
        ranked_indices = sorted(doc_indices_with_scores, key=lambda x: x[1], reverse=True)
        
        # Return position indices that scored higher than 0
        return [idx for idx, score in ranked_indices[:k] if score > 0]

    def rank_fusion_search(self, query: str, final_k: int = 2, constant_factor: int = 60) -> list:
        """Implements Reciprocal Rank Fusion (RRF) using stable document indices."""
        dense_ranks = self.dense_vector_search_indices(query, k=15)
        sparse_ranks = self.sparse_lexical_search_indices(query, k=15)
        
        rrf_scores = {}
        
        # Calculate scores based on position rank in Vector Search
        for rank, doc_idx in enumerate(dense_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (constant_factor + (rank + 1)))
            
        # Calculate and blend scores based on position rank in Keyword Search
        for rank, doc_idx in enumerate(sparse_ranks):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (constant_factor + (rank + 1)))
            
        if not rrf_scores:
            return []

        # Sort mathematically by highest combined rank score
        sorted_fused_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_fused_indices = [doc_idx for doc_idx, score in sorted_fused_indices[:final_k]]
        
        # Re-map the winning positions back into structured metadata dictionaries
        final_chunks = []
        for idx in top_fused_indices:
            final_chunks.append({
                "text": self.documents[idx],
                "section": self.metadatas[idx].get("section", "N/A"),
                "source": self.metadatas[idx].get("source", "Unknown")
            })
        return final_chunks

if __name__ == "__main__":
    try:
        retriever = HybridRetriever()
        test_query = "What are the specific penalties or administrative fines under Article 5?"
        
        print(f"\n[TESTING HYBRID RANK FUSION QUERY]: '{test_query}'", flush=True)
        print("[1/2] Fetching dense and sparse candidates...", flush=True)
        results = retriever.rank_fusion_search(test_query, final_k=2)
        
        print(f"[2/2] Fusion complete. Found {len(results)} matches:", flush=True)
        for i, chunk in enumerate(results):
            print(f"\n--- [Fused Match {i+1}] Section: {chunk['section']} ---", flush=True)
            print(chunk['text'][:300] + "...", flush=True)
            
    except Exception as e:
        print(f"\n❌ [CRITICAL ERROR DURING FUSION RUNTIME]: {e}", flush=True)