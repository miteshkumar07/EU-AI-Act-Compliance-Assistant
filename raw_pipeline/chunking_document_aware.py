import chromadb
from chromadb.utils import embedding_functions
import re

def chunk_text_by_article(text):
    # Split whenever we see "Article " followed by numbers, using a word boundary
    # Wrapping it in parenthesis captures the article title so we don't lose it
    raw_splits = re.split(r'\b(Article \d+)\b', text)
    
    chunks = []
    
    # The first element [0] is everything before Article 1 (the preamble)
    if raw_splits[0].strip():
        chunks.append(f"Preamble\n{raw_splits[0].strip()}")
        
    # Loop through the rest and pair the article titles with their bodies
    for i in range(1, len(raw_splits), 2):
        header = raw_splits[i].strip()
        body = raw_splits[i+1].strip() if (i+1) < len(raw_splits) else ""
        
        full_chunk = f"{header}\n{body}"
        chunks.append(full_chunk)
            
    return chunks
# ---------------------------------------------------
# STEP 1: Read the processed text files
# ---------------------------------------------------
with open('data/processed/clean_eu_ai_act.txt', 'r', encoding='utf-8') as f:
    act_text = f.read()

with open('data/processed/clean_eu_prohibited_ai.txt', 'r', encoding='utf-8') as f:
    guide_text = f.read()

# ---------------------------------------------------
# STEP 2: Chunk documents separately
# ---------------------------------------------------
raw_act_chunks = chunk_text_by_article(act_text)
# Note: Guidelines use section headers like "2.1.", "3.1." instead of "Article"
# For now, we will split them by paragraphs using double newlines
raw_guide_chunks = [p.strip() for p in guide_text.split('\n\n') if p.strip()]

# ---------------------------------------------------
# STEP 3: Build lists of Dictionaries with Metadata
# ---------------------------------------------------
all_chunks = []

# Process AI Act Chunks dynamically
for chunk in raw_act_chunks:
    match = re.search(r'Article \d+', chunk)
    article_label = match.group(0) if match else "Preamble/Recitals"
    
    all_chunks.append({
        "text": chunk,
        "source": "EU AI Act",
        "section": article_label
    })

# Process Guidelines Chunks dynamically
for chunk in raw_guide_chunks:
    # Hunt for section numbers like 3.1 or 5.2 at the start of paragraphs
    match = re.search(r'^\d+\.\d+\.?', chunk)
    section_label = f"Section {match.group(0)}" if match else "General Guidance"
    
    all_chunks.append({
        "text": chunk,
        "source": "Prohibited AI Guidelines",
        "section": section_label
    })

print(f"Total structured chunks created: {len(all_chunks)}")

# ---------------------------------------------------
# STEP 4: Save to ChromaDB
# ---------------------------------------------------
client = chromadb.PersistentClient(path="./chroma_db")

model_name = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

collection = client.get_or_create_collection(
    name="document_aware_by_article",
    embedding_function=embedding_func
)

# Prepare unpack arrays for Chroma
ids = [f"doc_aware_{i}" for i in range(len(all_chunks))]
documents = [chunk["text"] for chunk in all_chunks]
metadatas = [
    {"source": chunk["source"], "section": chunk["section"]} 
    for chunk in all_chunks
]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas
)

print("Storage complete! Structure-aware vectors are indexed under 'document_aware_by_article'.")