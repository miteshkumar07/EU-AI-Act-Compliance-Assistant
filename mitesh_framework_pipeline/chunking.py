import re
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
#import time
from langchain_experimental.text_splitter import SemanticChunker
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever




from dotenv import load_dotenv
load_dotenv()

def process_eu_ai_act(md_file_path, chunk_size=1000, chunk_overlap=100):
    """
    Processes the EU AI Act by separating it into Recitals, Articles, and Annexes
    before running the recursive character chunking.
    """
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Find the start of the Articles and Annexes
    articles_md = re.search(r'# _Article 1_', md_text)
    annex_md = re.search(r'## _ANNEX I_', md_text)
    
    if not articles_md or not annex_md:
        raise ValueError("Could not find the start of the Articles or Annexes in the text.")
        
    start_of_article = articles_md.start()
    start_of_annex = annex_md.start()
    
    # Split the text into the three massive parent sections
    recital_text = md_text[:start_of_article]
    article_text = md_text[start_of_article:start_of_annex]
    annex_text = md_text[start_of_annex:]

    # Extract filename for citations
    filename = os.path.basename(md_file_path)

    # Store in LangChain Document format with metadata
    docs = [
        Document(page_content=recital_text, metadata={"section": "recitals", "source": filename}),
        Document(page_content=article_text, metadata={"section": "articles", "source": filename}),
        Document(page_content=annex_text, metadata={"section": "annex", "source": filename})
    ]

    print("Chunking using the Recursive Character Chunking method...")
    # Chunk the documents while preserving metadata
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    eu_ai_act_chunks_baseline = text_splitter.split_documents(docs)
    eu_ai_act_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    persist_dir: str = "../framework_chroma_db"
    # Initialize empty Chroma DB
    eu_ai_act_embeddings_vectors = Chroma(
        collection_name="eu_ai_act_baseline",
        embedding_function=eu_ai_act_embeddings,
        persist_directory=persist_dir
    )
    
    # Add documents in batches to avoid rate limits (100 requests per minute)
    batch_size = 50
    for i in range(0, len(eu_ai_act_chunks_baseline), batch_size):
        batch = eu_ai_act_chunks_baseline[i:i + batch_size]
        eu_ai_act_embeddings_vectors.add_documents(batch)
        print(f"Added {min(i + batch_size, len(eu_ai_act_chunks_baseline))}/{len(eu_ai_act_chunks_baseline)} chunks for AI Act...")
        if i + batch_size < len(eu_ai_act_chunks_baseline):
            #time.sleep(32) # Wait 32 seconds before next batch to respect rate limits
            pass
    print("Saved the embeddings for the Recursive Character Chunking method to chroma db...")
##########################################################################
    print("Chunking using the Semantic Chunking method...")
    # Chunk the documents while preserving metadata
    semantic_chunker = SemanticChunker(eu_ai_act_embeddings)
    eu_ai_act_chunks_semantic = semantic_chunker.split_documents(docs)

    persist_dir: str = "../framework_chroma_db"
    # Initialize empty Chroma DB
    eu_ai_act_embeddings_vectors = Chroma(
        collection_name="eu_ai_act_semantic_chunks",
        embedding_function=eu_ai_act_embeddings,
        persist_directory=persist_dir
    )
    
    # Add documents in batches to avoid rate limits (100 requests per minute)
    batch_size = 50
    for i in range(0, len(eu_ai_act_chunks_semantic), batch_size):
        batch = eu_ai_act_chunks_semantic[i:i + batch_size]
        eu_ai_act_embeddings_vectors.add_documents(batch)
        print(f"Added {min(i + batch_size, len(eu_ai_act_chunks_semantic))}/{len(eu_ai_act_chunks_semantic)} chunks for AI Act...")
        if i + batch_size < len(eu_ai_act_chunks_semantic):
            #time.sleep(32) # Wait 32 seconds before next batch to respect rate limits
            pass
    print("Saved the embeddings for the Semantic Chunking method to chroma db...")

##########################################################################
    print("Chunking using the Context-Enriched Chunking (ParentDocumentRetriever) method...")
    
    # 1. Splitters: Large parents, small children
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 2. Storage for Child Chunks (Vector DB)
    eu_ai_act_embeddings_vectors_context = Chroma(
        collection_name="eu_ai_act_context_chunks",
        embedding_function=eu_ai_act_embeddings,
        persist_directory=persist_dir
    )

    # 3. Storage for Parent Chunks (Local Disk)
    fs = LocalFileStore("../framework_docstore/eu_ai_act")
    store = create_kv_docstore(fs)
    
    # 4. The Retriever that manages both
    retriever = ParentDocumentRetriever(
        vectorstore=eu_ai_act_embeddings_vectors_context,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    # Add the documents to the retriever (it will automatically chunk and route them to both stores)
    print("Building the ParentDocumentRetriever for the AI Act...")
    retriever.add_documents(docs)
    
    print("Saved the embeddings for the Context-Enriched Chunking method to chroma db...")
##########################################################################
            
    return eu_ai_act_embeddings_vectors

def process_prohibited_ai_guidelines(md_file_path, chunk_size=1000, chunk_overlap=100):
    """
    Processes the Prohibited AI Guidelines by cleaning up PDF-conversion artifacts
    and using a hierarchical Markdown splitter.
    """
    with open(md_file_path, "r", encoding="utf-8") as f:
        proh_md_content = f.read()

    # --- PREPROCESSING / CLEANING PHASE ---
    # Fix 1: Remove fake headers from "For example," boxes
    proh_md_content = re.sub(r'# For example,', r'For example,', proh_md_content)

    # Fix 2: Remove fake headers from "Article X AI Act provides:" quote boxes
    proh_md_content = re.sub(r'# (\*\*_*Article \d+.*\*\*)', r'\1', proh_md_content)

    # Fix 3: Upgrade sub-sections (e.g., "# **2.1." -> "## **2.1.")
    proh_md_content = re.sub(r'# \*\*(\d+\.\d+\.)', r'## **\1', proh_md_content)

    # Fix 4: Upgrade sub-sub-sections (e.g., "# **2.5.1." -> "### **2.5.1.")
    proh_md_content = re.sub(r'# \*\*(\d+\.\d+\.\d+\.)', r'### **\1', proh_md_content)

    # Fix 5: Upgrade lettered sections (e.g., "# **_a)" -> "#### **_a)")
    proh_md_content = re.sub(r'# \*\*_([a-z]\))', r'#### **_\1', proh_md_content)
    proh_md_content = re.sub(r'# _([a-z]\))', r'#### _\1', proh_md_content)

    # --- LANGCHAIN CHUNKING PHASE ---
    headers_to_split_on = [
        ("#", "Main_Section"),    
        ("##", "Sub_Section"),
        ("###", "Sub_Sub_Section"),
        ("####", "Letter_Section")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False 
    )

    # 1. Create the hierarchical Parent Chunks
    filename = os.path.basename(md_file_path)
    md_header_splits = markdown_splitter.split_text(proh_md_content)
    for doc in md_header_splits:
        doc.metadata["source"] = filename
        
    print("Chunking using the Recursive Character Chunking method...")

    # 2. Chop them down to the specified size if they are too long
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    eu_ai_act_proh_chunks = text_splitter.split_documents(md_header_splits)
    eu_ai_act_proh_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    persist_dir: str = "../framework_chroma_db"
    # Initialize empty Chroma DB
    eu_ai_act_proh_embeddings_vectors = Chroma(
        collection_name="eu_ai_act_proh_baseline",
        embedding_function=eu_ai_act_proh_embeddings,
        persist_directory=persist_dir
    )
    
    # Add documents in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(eu_ai_act_proh_chunks), batch_size):
        batch = eu_ai_act_proh_chunks[i:i + batch_size]
        eu_ai_act_proh_embeddings_vectors.add_documents(batch)
        print(f"Added {min(i + batch_size, len(eu_ai_act_proh_chunks))}/{len(eu_ai_act_proh_chunks)} chunks for Prohibited Guidelines...")
        if i + batch_size < len(eu_ai_act_proh_chunks):
            #time.sleep(32) # Wait 32 seconds before next batch to respect rate limits
            pass
    print("Saved the embeddings for the Recursive Character Chunking method to chroma db...")
    ##########################################################################




    print("Chunking using the Semantic Chunking method...")
    # Chunk the documents while preserving metadata
    # FIX: Pass the embedding model, not the text string!
    semantic_chunker = SemanticChunker(eu_ai_act_proh_embeddings)
    eu_ai_act_chunks_semantic = semantic_chunker.split_documents(md_header_splits)

    persist_dir: str = "../framework_chroma_db"
    # Initialize empty Chroma DB
    # FIX: Use a unique collection name for the guidelines, and pass the embedding model!
    eu_ai_act_proh_embeddings_vectors_semantic = Chroma(
        collection_name="eu_ai_act_proh_semantic_chunks",
        embedding_function=eu_ai_act_proh_embeddings,
        persist_directory=persist_dir
    )
    
    # Add documents in batches
    batch_size = 50
    for i in range(0, len(eu_ai_act_chunks_semantic), batch_size):
        batch = eu_ai_act_chunks_semantic[i:i + batch_size]
        eu_ai_act_proh_embeddings_vectors_semantic.add_documents(batch)
        print(f"Added {min(i + batch_size, len(eu_ai_act_chunks_semantic))}/{len(eu_ai_act_chunks_semantic)} chunks for Prohibited Guidelines (Semantic)...")
        
    print("Saved the embeddings for the Semantic Chunking method to chroma db...")
    
##########################################################################
    print("Chunking using the Context-Enriched Chunking (ParentDocumentRetriever) method...")
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    eu_ai_act_proh_embeddings_vectors_context = Chroma(
        collection_name="eu_ai_act_proh_context_chunks",
        embedding_function=eu_ai_act_proh_embeddings,
        persist_directory=persist_dir
    )

    fs = LocalFileStore("../framework_docstore/eu_ai_act_proh")
    store = create_kv_docstore(fs)
    
    retriever = ParentDocumentRetriever(
        vectorstore=eu_ai_act_proh_embeddings_vectors_context,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    print("Building the ParentDocumentRetriever for the Prohibited Guidelines...")
    retriever.add_documents(md_header_splits)
    
    print("Saved the embeddings for the Context-Enriched Chunking method to chroma db...")
##########################################################################

    # Return the baseline vectors so the print statements at the bottom still work
    return eu_ai_act_proh_embeddings_vectors

if __name__ == "__main__":
    ai_act_path = "../data/processed/clean_eu_ai_act.md"
    proh_ai_path = "../data/processed/clean_eu_proh_ai_act.md"
    
    print("Processing EU AI Act...")
    ai_act_db = process_eu_ai_act(ai_act_path)
    print(f"Success! Created {ai_act_db._collection.count()} structured chunks for the AI Act.\n")
    
    print("Processing Prohibited AI Guidelines...")
    proh_ai_db = process_prohibited_ai_guidelines(proh_ai_path)
    print(f"Success! Created {proh_ai_db._collection.count()} perfectly structured chunks for the Guidelines.")
