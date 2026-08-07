import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.documents import Document
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_ensemble_retriever():
    """
    Builds a Hybrid Retriever combining BM25 (Sparse/Keyword) 
    and ChromaDB Context-Enriched Retriever using BAAI/bge-large-en-v1.5.
    """
    print("Loading Semantic (Dense) Context-Enriched Retriever...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 1. Load the Context-Enriched ChromaDB and Parent Store
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    child_db = Chroma(
        collection_name="eu_ai_act_context_chunks",
        embedding_function=embeddings,
        persist_directory=os.path.join(base_dir, "framework_chroma_db")
    )
    fs = LocalFileStore(os.path.join(base_dir, "framework_docstore", "eu_ai_act"))
    store = create_kv_docstore(fs)
    
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    dense_retriever = ParentDocumentRetriever(
        vectorstore=child_db,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 20}
    )
    
    # 2. Build the BM25 Keyword Retriever
    print("Building Keyword (Sparse) Retriever...")
    # Extract all the text from the parent docstore to build the keyword index
    docs = []
    for key in store.yield_keys():
        doc = store.mget([key])[0]
        if doc:
            docs.append(doc)
        
    # BM25 operates entirely in memory on the raw text
    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = 20
    
    # 3. Combine them using Reciprocal Rank Fusion (RRF)
    print("Fusing Retrievers via Reciprocal Rank Fusion (RRF)...")
    # Weights: Give slightly more weight to semantic meaning (0.6) than exact keywords (0.4)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4]
    )
    cross_encoder = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        model_kwargs={'device': 'cpu'}
    )
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=7)
    ensemble_retriever = ContextualCompressionRetriever(
        base_retriever=ensemble_retriever,
        base_compressor=compressor
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
