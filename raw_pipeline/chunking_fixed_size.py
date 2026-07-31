import chromadb
from chromadb.utils import embedding_functions

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    
    # We loop through the text, jumping forward by (chunk_size - overlap) each time
    while start < text_length:
        # Slice the string from 'start' to 'start + chunk_size'
        end = start + chunk_size
        chunk = text[start:end]
        
        chunks.append(chunk)
        
        # Move the starting point forward, but step back by the overlap amount
        # so the next chunk repeats the last 200 characters of this one.
        start += (chunk_size - overlap)
        
    return chunks


# 1. Read both files
with open('data/processed/clean_eu_ai_act.txt', 'r', encoding='utf-8') as f:
    act_text = f.read()

with open('data/processed/clean_eu_prohibited_ai.txt', 'r', encoding='utf-8') as f:
    guide_text = f.read()

# 2. Chunk them separately 
raw_act_chunks = chunk_text(act_text, chunk_size=1000, overlap=200)
raw_guide_chunks = chunk_text(guide_text, chunk_size=1000, overlap=200)

# 3. Upgrade them to Dictionaries with Metadata
act_chunks_with_metadata = [
    {"text": chunk, "source": "EU AI Act"} for chunk in raw_act_chunks
]

guide_chunks_with_metadata = [
    {"text": chunk, "source": "Prohibited AI Guidelines"} for chunk in raw_guide_chunks
]

# 4. NOW we combine them
all_chunks = act_chunks_with_metadata + guide_chunks_with_metadata

print(f"Total chunks created: {len(all_chunks)}")


##### Saving the chunks to chromaDB 

# 1. Setup the Chroma Client
# Using a persistent directory means your vectors will be saved to your hard drive
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Setup the Embedding Function
# This tells Chroma to use the exact same model you tested on Day 2
model_name = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# 3. Create (or get) the collection
collection = client.get_or_create_collection(
    name="fixed_length_1000_200",
    embedding_function=embedding_func
)

# 4. Prepare your data for Chroma
# Chroma needs separate lists for documents, IDs, and metadata
# IDs must be unique strings
ids = [f"id_{i}" for i in range(len(all_chunks))]
documents = [chunk["text"] for chunk in all_chunks]
metadatas = [{"source": chunk["source"]} for chunk in all_chunks]

# 5. Add to the collection
# Chroma will use the embedding_func to automatically convert documents to vectors!
collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print("Storage complete! Vectors are now indexed in ChromaDB.")