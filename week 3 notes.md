# Chunking Strategies
Here is a step-by-step breakdown of exactly how each of the three chunking methods we built processes your legal documents:
  ──────
  ### 1. The Baseline: Recursive Character Chunking
  This is the industry standard. It prioritizes keeping chunks at a specific size without slicing words in half.

  How it worked in your code:

  1. Set the Rules: We told the RecursiveCharacterTextSplitter to create chunks of exactly 1,000 characters, with a 100-character overlap (so no
  context is lost at the edges).
  2. Attempt 1 (Paragraphs): It looks at the massive AI Act and tries to chop it up wherever it sees a double-return (a new paragraph).
  3. Attempt 2 (Sentences): If a paragraph is longer than 1,000 characters, it recursively moves to the next rule: it looks for periods to split
  the text by individual sentences.
  4. Attempt 3 (Words): If a single sentence is somehow longer than 1,000 characters (common in legal texts!), it splits by spaces to separate
  words.
  5. Embed and Store: It takes those perfectly sized 1,000-character blocks, converts them to vectors using the BAAI model, and saves them in the
  eu_ai_act_baseline table in ChromaDB.

  The Result: Clean, uniform blocks of text. Good for general search, but sometimes cuts legal context in half.
  ──────
  ### 2. The Advanced Method: Semantic Chunking

  This is a purely AI-driven approach. It ignores character limits completely and relies entirely on "meaning."

  How it worked in your code:

  1. Sentence Isolation: The SemanticChunker ignores paragraphs and immediately breaks the entire EU AI Act into hundreds of individual sentences.
  2. Sentence Embedding: Before making a single chunk, it asks the local BAAI model to embed every single sentence into a mathematical vector.
  3. Math Comparison: It compares the vector of Sentence 1 against Sentence 2. If the math says they mean the exact same thing, it glues them
  together into a chunk.
  4. The Cut-off: It continues comparing sentences (Sentence 2 vs 3, 3 vs 4). The moment the math detects a sudden "jump" in meaning—signaling the
  topic has changed from "Definitions" to "Penalties"—it slices the chunk right there and starts a new one.
  5. Store: It saves these dynamic, variable-length chunks into the eu_ai_act_semantic_chunks table.

  The Result: Chunks that perfectly encapsulate a single idea or topic, regardless of how long or short they are.
  ──────
  ### 3. The Legal Champion: Context-Enriched Chunking

  This is the most complex method, using the ParentDocumentRetriever. It is specifically designed to solve the "Lost Context" problem in legal
  documents (e.g., when a small chunk says "The Provider must pay a fine," but doesn't explain who the provider is).

  How it worked in your code:

  1. Dual Splitters: We created two splitters: a massive "Parent" splitter (2,000 characters) and a tiny "Child" splitter (400 characters).
  2. Parent Storage: The script chops the AI Act into massive 2,000-character parent chunks. It gives each parent a unique ID and saves the raw
  text directly to a local folder on your Mac (framework_docstore).
  3. Child Shredding: It then takes those massive parents and shreds them further into 400-character child chunks. It tags every child with the
  unique ID of its parent.
  4. Vector Storage: It takes only the tiny 400-character child chunks, embeds them, and puts them in the Chroma Database
  (eu_ai_act_context_chunks).
  5. The Magic Retrieval: Later, when a user asks a question, the AI will search the tiny chunks to find a highly accurate match. But instead of
  returning the tiny chunk, the ParentDocumentRetriever intercepts it, looks up the Parent ID, and returns the massive 2,000-character document
  from your local folder to the LLM.

  The Result: High-precision search (using tiny chunks) combined with massive context window retrieval (returning parent chunks). Perfect for legal
  documents.
  ──────