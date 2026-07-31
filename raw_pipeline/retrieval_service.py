import chromadb
from chromadb.utils import embedding_functions

def retrieve_top_k(query: str, collection_name: str, k: int = 3):
    """
    Takes a raw query string, hits a specific collection, and returns the top-k matched chunks.
    """
    # 1. Connect to our persistent directory
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 2. Re-attach the exact same embedding function we used to index the data
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # 3. Load the specific collection
    collection = client.get_collection(name=collection_name, embedding_function=embedding_func)
    
    # 4. Perform the query (This automatically embeds your query and runs the top-k graph search)
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    
    # 5. Format the output cleanly for downstream steps
    formatted_results = []
    
    # Chroma nested arrays structure: results['documents'][0] because we sent a list of 1 query text
    for i in range(len(results['documents'][0])):
        formatted_results.append({
            "text": results['documents'][0][i],
            "id": results['ids'][0][i],
            "source": results['metadatas'][0][i]['source'],
            "section": results['metadatas'][0][i].get('section', 'N/A')
        })
        
    return formatted_results

# --- Quick Code Test ---
if __name__ == "__main__":
    test_query = "What are the obligations of providers of general-purpose AI models?"
    print(f"Testing retrieval for: '{test_query}'\n")
    
    matches = retrieve_top_k(test_query, collection_name="fixed_length_1000_200", k=2)
    for idx, match in enumerate(matches):
        print(f"--- MATCH {idx+1} | Source: {match['source']} | Section: {match['section']} ---")
        print(match['text'] + "...\n")